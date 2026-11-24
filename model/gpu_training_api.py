"""Internal API for the existing run_training.sh launcher."""

import json
import os
import re
import secrets
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "run_training.sh"
JOB_ROOT = ROOT / "runs" / "api"
TRAIN_PYTHON = Path(
    os.environ.get(
        "GPU_TRAINING_PYTHON",
        str(Path.home() / "miniforge3/envs/defect/bin/python"),
    )
)
MODEL_DIRS = {"anomaly": "이상치 모델", "classifier": "분류 모델"}
JOB_ID = re.compile(r"[0-9]{20}_(anomaly|classifier)_[0-9a-f]{12}")
SAVE_MODEL_LINE = re.compile(
    r"(?m)^(SAVE_MODEL[ \t]*:[ \t]*)(?:true|false)([ \t]*(?:#.*)?)$"
)
LOCK = threading.Lock()
PROCESSES: dict[str, subprocess.Popen] = {}
QUEUED: set[str] = set()


def require_token(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("GPU_TRAINING_TOKEN")
    if not token:
        raise HTTPException(503, "GPU API authentication is not configured")
    expected = f"Bearer {token}".encode()
    if authorization is None or not secrets.compare_digest(
        authorization.encode(), expected
    ):
        raise HTTPException(401, "Invalid credentials")


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class TrainingRequest(BaseModel):
    model: Literal["anomaly", "classifier", "both"]
    fold: int | None = Field(default=None, ge=0, le=9)
    register_model: bool = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def job_path(job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(404, "Job not found")
    path = JOB_ROOT / job_id
    if not path.is_dir():
        raise HTTPException(404, "Job not found")
    return path


def metadata(path: Path) -> dict:
    try:
        return json.loads((path / "job.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_metadata(path: Path, values: dict) -> None:
    temporary = path / "job.json.tmp"
    temporary.write_text(json.dumps(values), encoding="utf-8")
    temporary.replace(path / "job.json")


def training_result(path: Path) -> dict:
    values = {}
    try:
        with (path / "result.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    values.update(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return values


def exit_code(path: Path) -> int | None:
    try:
        return int((path / "training.log.exit").read_text().strip())
    except (OSError, ValueError):
        return None


def describe_job(path: Path) -> dict:
    stored = metadata(path)
    result = training_result(path)
    process = PROCESSES.get(path.name)
    code = exit_code(path)
    stopping = (path / "stop-requested").is_file()
    if stored.get("state") == "cancelled":
        state = "cancelled"
    elif code == 0:
        state = "succeeded"
    elif code == 3:
        state = "busy"
    elif code is not None:
        stopped = stopping and code in (-15, -2, 130, 143)
        state = "cancelled" if stopped else "failed"
    elif path.name in QUEUED:
        state = "queued"
    elif process is None:
        state = "cancelled" if stored and not stored.get("started_at") else "unknown"
    else:
        state = "stopping" if stopping else "running"

    submitted, model, _ = path.name.split("_")
    started_at = stored.get("started_at")
    if not stored:
        started_at = datetime.strptime(submitted, "%Y%m%d%H%M%S%f").replace(
            tzinfo=timezone.utc
        ).isoformat()
    finished_at = stored.get("finished_at")
    if not finished_at and code is not None:
        finished_at = datetime.fromtimestamp(
            (path / "training.log.exit").stat().st_mtime, timezone.utc
        ).isoformat()
    duration = None
    if started_at and (finished_at or state in ("running", "stopping")):
        end = datetime.fromisoformat(finished_at) if finished_at else utc_now()
        duration = max(0, int((end - datetime.fromisoformat(started_at)).total_seconds()))
    requested = stored.get("register_model")
    registration = result.get("registration_status")
    if registration is None and requested is not None:
        registration = (
            "not_requested" if not requested else
            "pending" if state in ("queued", "running", "stopping") else
            "unconfirmed"
        )

    return {
        "job_id": path.name,
        "model": model,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
        "status": state,
        "exit_code": code,
        "fold": stored.get("fold"),
        "register_model": requested,
        "registration_status": registration,
        "registered_model_name": result.get("registered_model_name"),
        "registered_model_version": result.get("registered_model_version"),
        "run_id": result.get("run_id"),
        "experiment_id": result.get("experiment_id"),
        "batch_id": stored.get("batch_id"),
    }


def prepare_job(model: str, fold: int | None, register_model: bool, batch_id: str | None):
    model_dir = ROOT / MODEL_DIRS[model]
    required_files = (SCRIPT, model_dir / "config.yaml", model_dir / "train.py")
    if not all(path.is_file() for path in required_files):
        raise HTTPException(503, "Training files are missing")
    try:
        original = (model_dir / "config.yaml").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise HTTPException(503, "Could not read model config") from None
    config_value = "true" if register_model else "false"
    config, matches = SAVE_MODEL_LINE.subn(
        lambda match: match[1] + config_value + match[2], original
    )
    keys = re.findall(r"(?m)^SAVE_MODEL[ \t]*:", original)
    if matches != 1 or len(keys) != 1:
        raise HTTPException(503, "Expected one top-level SAVE_MODEL boolean")

    job_id = f"{utc_now():%Y%m%d%H%M%S%f}_{model}_{secrets.token_hex(6)}"
    path = JOB_ROOT / job_id
    job_config = model_dir / f".config.api.{job_id}.yaml"
    try:
        path.mkdir(mode=0o700, parents=True)
        save_metadata(path, {
            "model": model,
            "fold": fold,
            "register_model": register_model,
            "batch_id": batch_id,
            "started_at": None,
        })
        fd = os.open(job_config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(config)
    except OSError:
        job_config.unlink(missing_ok=True)
        if path.is_dir():
            (path / "training.log.exit").write_text("1\n")
        raise HTTPException(503, "Could not prepare training files") from None
    QUEUED.add(job_id)
    return path, job_config


def cancel_queued(job: tuple[Path, Path], reason: str) -> None:
    path, config = job
    values = metadata(path)
    values.update(state="cancelled", finished_at=utc_now().isoformat())
    save_metadata(path, values)
    (path / "training.log").write_text(reason + "\n", encoding="utf-8")
    config.unlink(missing_ok=True)
    QUEUED.discard(path.name)


def launch_job(job: tuple[Path, Path]) -> subprocess.Popen:
    path, config = job
    values = metadata(path)
    model = values["model"]
    args = ["bash", str(SCRIPT), model, "--config", config.name]
    args.extend(["--experiment", f"retraining-{model}"])
    if values["fold"] is not None:
        args.extend(["--fold", str(values["fold"])])
    train_env = dict(os.environ)
    train_env.pop("GPU_TRAINING_TOKEN", None)
    train_env.update(
        PYTHON=str(TRAIN_PYTHON),
        LOG_FILE=str(path / "training.log"),
        TRAINING_RESULT_FILE=str(path / "result.jsonl"),
    )
    values["started_at"] = utc_now().isoformat()
    save_metadata(path, values)
    QUEUED.discard(path.name)
    try:
        process = subprocess.Popen(args, cwd=ROOT, env=train_env, start_new_session=True)
    except OSError:
        config.unlink(missing_ok=True)
        (path / "training.log.exit").write_text("1\n")
        raise HTTPException(503, "Could not start the training process") from None
    PROCESSES[path.name] = process
    return process


def finish_jobs(jobs: list[tuple[Path, Path]], process: subprocess.Popen) -> None:
    for index, (path, config) in enumerate(jobs):
        code = process.wait()
        with LOCK:
            if exit_code(path) is None:
                (path / "training.log.exit").write_text(f"{code}\n")
            values = metadata(path)
            values["finished_at"] = utc_now().isoformat()
            save_metadata(path, values)
            config.unlink(missing_ok=True)
            PROCESSES.pop(path.name, None)
            if index + 1 == len(jobs):
                return
            following = jobs[index + 1]
            if code != 0 or (path / "stop-requested").is_file():
                cancel_queued(following, "앞선 학습이 중단되었거나 실패했습니다.")
                return
            try:
                process = launch_job(following)
            except HTTPException as error:
                (following[0] / "training.log").write_text(str(error.detail), encoding="utf-8")
                return


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "gpu-training"}


@app.post("/jobs", status_code=202, dependencies=[Depends(require_token)])
def start_job(payload: TrainingRequest) -> dict:
    if payload.model == "anomaly" and payload.fold is not None:
        raise HTTPException(422, "fold is only supported for classifier")

    if not os.access(TRAIN_PYTHON, os.X_OK):
        raise HTTPException(503, "Training Python is not executable")

    models = ("anomaly", "classifier") if payload.model == "both" else (payload.model,)
    batch_id = secrets.token_hex(12) if len(models) > 1 else None
    jobs = []
    with LOCK:
        if PROCESSES:
            raise HTTPException(409, "A job is already running")
        try:
            for model in models:
                fold = (payload.fold if payload.fold is not None else 0) if model == "classifier" else None
                jobs.append(prepare_job(model, fold, payload.register_model, batch_id))
            process = launch_job(jobs[0])
        except HTTPException:
            for job in jobs:
                if job[0].name in QUEUED:
                    cancel_queued(job, "학습 시작 요청을 처리하지 못했습니다.")
            raise
        threading.Thread(target=finish_jobs, args=(jobs, process), daemon=True).start()
        return describe_job(jobs[0][0])

@app.get("/jobs", dependencies=[Depends(require_token)])
def list_jobs() -> list[dict]:
    if not JOB_ROOT.is_dir():
        return []
    with LOCK:
        paths = (p for p in JOB_ROOT.iterdir() if p.is_dir() and JOB_ID.fullmatch(p.name))
        return [describe_job(path) for path in sorted(paths, reverse=True)[:30]]


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def read_job(job_id: str) -> dict:
    with LOCK:
        return describe_job(job_path(job_id))


@app.get("/jobs/{job_id}/log", dependencies=[Depends(require_token)])
def read_log(job_id: str) -> dict:
    path = job_path(job_id) / "training.log"
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 64 * 1024))
            log = stream.read(64 * 1024).decode("utf-8", errors="replace")
    except FileNotFoundError:
        log = ""
    return {"job_id": job_id, "log": log}


@app.post("/jobs/{job_id}/stop", status_code=202, dependencies=[Depends(require_token)])
def stop_job(job_id: str) -> dict:
    with LOCK:
        path = job_path(job_id)
        process = PROCESSES.get(job_id)
        if process is None or process.poll() is not None:
            raise HTTPException(409, "Job is no longer running in this API process")
        if not (path / "training.log").exists():
            raise HTTPException(409, "Job is still starting; retry")
        marker = path / "stop-requested"
        if not marker.exists():
            marker.touch()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                marker.unlink(missing_ok=True)
                raise HTTPException(409, "Job has exited") from None
        return describe_job(path)
