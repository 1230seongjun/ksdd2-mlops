"""운영에서 학습했던 모델을 공개 읽기 전용 버킷에서 받아 로컬 MLflow 에 같은 버전 번호로 등록합니다.

로컬 실행(deploy/local/ksdd2.yaml)의 model-import 컨테이너가 Pod 를 띄울 때마다 실행합니다.
  - 목록(models.json)에 적힌 파일만 키 없이 HTTP GET 으로 받습니다. 운영 버킷에는 아무것도 쓰지 않습니다.
  - 받은 파일은 로컬 MLflow(ksdd2-mlartifacts 볼륨)에 저장되므로, 한 번 받은 뒤에는 인터넷 없이도 씁니다.
  - 운영에서 지운 번호는 빈 버전을 넣었다 지워서 건너뜁니다 → 번호가 운영과 같고,
    로컬에서 재학습한 모델은 그다음 번호(예: v9)로 로컬에만 등록됩니다.
  - 이미 등록돼 있으면 건너뛰고, 중간에 끊겼으면 이어서 등록합니다.

models.json 은 운영 MLflow 에서 export_models.py 로 만듭니다.
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from mlflow import MlflowClient
from mlflow.entities import Metric, Param

EXPERIMENT = "운영 모델 (불러옴)"


def wait_for_mlflow(client: MlflowClient) -> None:
    for _ in range(90):
        try:
            client.search_experiments(max_results=1)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("MLflow 에 연결하지 못했습니다")


def download(url: str, target: Path, size: int, sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as file:
                digest = hashlib.sha256()
                while chunk := response.read(1 << 20):
                    file.write(chunk)
                    digest.update(chunk)
            if target.stat().st_size != size or digest.hexdigest() != sha256:
                raise ValueError(f"내려받은 파일이 목록과 다릅니다: {url}")
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * attempt)


def new_run(client: MlflowClient, experiment_id: str, run: dict) -> str:
    created = client.create_run(experiment_id, run_name=run.get("name") or None,
                                tags={"imported_from": "production"})
    run_id = created.info.run_id
    params = [Param(key, str(value)[:6000]) for key, value in (run.get("params") or {}).items()]
    now = int(time.time() * 1000)
    metrics = [Metric(key, float(value), now, 0) for key, value in (run.get("metrics") or {}).items()]
    for start in range(0, max(len(params), len(metrics)), 100):
        client.log_batch(run_id, metrics=metrics[start:start + 100], params=params[start:start + 100])
    return run_id


def register(client: MlflowClient, name: str, run_id: str, expected: int, tags=None, description=None):
    artifact_uri = client.get_run(run_id).info.artifact_uri
    version = client.create_model_version(name, f"{artifact_uri}/model", run_id=run_id,
                                          tags=tags or {}, description=description or None)
    for _ in range(60):
        version = client.get_model_version(name, version.version)
        if version.status != "PENDING_REGISTRATION":
            break
        time.sleep(1)
    if int(version.version) != expected:
        raise RuntimeError(f"{name}: v{expected} 자리에 v{version.version} 이 등록됐습니다. "
                           "로컬에서 먼저 학습한 모델이 있으면 podman kube down --force 로 비운 뒤 다시 실행하세요.")
    return version


def import_model(client: MlflowClient, base_url: str, experiment_id: str, model: dict) -> None:
    name = model["name"]
    wanted = {int(item["version"]): item for item in model["versions"]}
    if not wanted:
        return
    last = max(wanted)
    existing = [int(v.version) for v in client.search_model_versions(f"name = '{name}'")]
    if not existing:
        try:
            client.create_registered_model(name, description=model.get("description") or None)
        except Exception:
            pass   # 이미 있음 (버전은 모두 지워진 상태)
    start = max(existing, default=0) + 1
    if start > last:
        print(f"[{name}] 이미 v{last} 까지 등록돼 있어 건너뜁니다")
        return
    print(f"[{name}] v{start} ~ v{last} 불러오기")
    for number in range(start, last + 1):
        item = wanted.get(number)
        if item is None:
            # 운영에서 지운 번호: 빈 버전을 만들었다 지워 번호만 맞춘다
            run_id = new_run(client, experiment_id, {"name": f"{name} v{number} (운영에서 삭제된 번호)"})
            with tempfile.TemporaryDirectory() as folder:
                Path(folder, "README.txt").write_text("운영에서 삭제된 버전 번호를 맞추기 위한 자리입니다.\n", encoding="utf-8")
                client.log_artifacts(run_id, folder, "model")
            client.set_terminated(run_id)
            register(client, name, run_id, number)
            client.delete_model_version(name, str(number))
            print(f"  v{number}: 운영에서 삭제된 번호 — 건너뜀")
            continue
        with tempfile.TemporaryDirectory() as folder:
            for file in item["files"]:
                url = f"{base_url}/{quote(name)}/{number}/{quote(file['path'])}"
                download(url, Path(folder, file["path"]), int(file["size"]), file["sha256"])
            run_id = new_run(client, experiment_id, item.get("run") or {})
            client.log_artifacts(run_id, folder, "model")
        client.set_terminated(run_id)
        register(client, name, run_id, number, item.get("tags"), item.get("description"))
        for alias in item.get("aliases") or []:
            client.set_registered_model_alias(name, alias, str(number))
        size = sum(int(f["size"]) for f in item["files"]) / 1e6
        print(f"  v{number}: 등록 ({len(item['files'])}개 파일, {size:.1f}MB)")


def main() -> int:
    manifest_path = Path(sys.argv[1] if len(sys.argv) > 1 else "models.json")
    if not manifest_path.exists():
        print(f"{manifest_path} 가 없어 운영 모델을 불러오지 않습니다 (재학습하면 v1 부터 로컬에 등록)")
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_url = (manifest.get("base_url") or "").rstrip("/")
    if not base_url:
        print("models.json 에 base_url 이 없어 운영 모델을 불러오지 않습니다")
        return 0
    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    wait_for_mlflow(client)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    for model in manifest.get("models", []):
        import_model(client, base_url, experiment_id, model)
    print("운영 모델 불러오기 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
