"""추론 API 를 띄우기 전에 한 번 실행하는 준비 단계 (deploy/app/start-api.sh 가 호출).

1. DB 와 MLflow 가 응답할 때까지 기다린다 (계층마다 Pod 가 따로 떠서 순서가 보장되지 않기 때문).
2. 이미지에 들어 있는 운영 모델(GitHub Release)을 로컬 MLflow 에 운영 때와 같은 버전 번호로 등록한다.
   - 이미 그 번호까지 등록돼 있으면 건너뛴다 (API 를 다시 시작해도 중복 등록되지 않음).
   - 앞 번호는 빈 버전을 만들었다 지워서 번호만 맞춘다. 그래서 로컬에서 재학습한 모델은 다음 번호(예: v9)가 된다.
3. 데모 관리자 계정을 만든다 (이미 있으면 비밀번호만 데모 값으로 맞춤).
4. 기본으로 쓸 모델을 고른다: 이상 탐지는 가장 최근 버전, 분류는 그 버전의 classifier_version 태그(없으면 가장 최근 버전).
   추론 API 는 MODEL_URI 를 숫자 버전으로만 받으므로(db.model_version) 여기서 숫자로 정해 넘긴다.

마지막 줄에 셸에서 eval 할 export 문을 출력한다. 진행 상황은 표준 오류로 출력한다.
"""
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from mlflow import MlflowClient

ASSETS = Path(os.environ.get("ASSETS_DIR", "/opt/ksdd2"))
DETECTOR, CLASSIFIER = "Zero-Defects", "Zero-Defects-Classifier"


def log(message: str) -> None:
    print(f"[bootstrap] {message}", file=sys.stderr, flush=True)


def wait(name: str, check, seconds: int = 180) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            check()
            log(f"{name} 준비됨")
            return
        except Exception as error:   # 아직 뜨는 중
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name} 에 연결하지 못했습니다: {error}") from error
            time.sleep(2)


def register(client: MlflowClient, name: str, folder: Path, experiment_id: str, expected: int, tags=None):
    run = client.create_run(experiment_id, run_name=f"{name} v{expected}", tags={"source": "github-release"})
    client.log_artifacts(run.info.run_id, str(folder), "model")
    client.set_terminated(run.info.run_id)
    version = client.create_model_version(
        name, f"{run.info.artifact_uri}/model", run_id=run.info.run_id, tags=tags or {}
    )
    if int(version.version) != expected:
        raise RuntimeError(
            f"{name}: v{expected} 자리에 v{version.version} 이 등록됐습니다. "
            "로컬에서 먼저 학습한 모델이 있으면 deploy/down.sh --volumes 로 비운 뒤 다시 실행하세요."
        )


def import_models(client: MlflowClient) -> None:
    manifest_path = ASSETS / "assets.json"
    models = json.loads(manifest_path.read_text(encoding="utf-8")).get("models", []) if manifest_path.exists() else []
    models = [m for m in models if (ASSETS / "models" / m["file"]).exists()]
    if not models:
        log("이미지에 운영 모델이 없어 건너뜀 (재학습하면 v1 부터 등록)")
        return
    experiment = client.get_experiment_by_name("운영 모델 (GitHub Release)")
    experiment_id = experiment.experiment_id if experiment else client.create_experiment("운영 모델 (GitHub Release)")
    for model in sorted(models, key=lambda m: m["name"] != CLASSIFIER):   # 분류 모델 먼저: 이상 탐지 태그가 분류 버전을 가리킴
        name, target = model["name"], int(model["version"])
        existing = [int(v.version) for v in client.search_model_versions(f"name = '{name}'")]
        if max(existing, default=0) >= target:
            log(f"{name} v{target}: 이미 등록됨")
            continue
        if not existing:
            try:
                client.create_registered_model(name)
            except Exception:
                pass   # 버전은 모두 지워졌지만 이름은 남아 있는 경우
        with tempfile.TemporaryDirectory() as folder:
            for number in range(max(existing, default=0) + 1, target):
                Path(folder, "placeholder.txt").write_text("운영 버전 번호를 맞추기 위한 자리\n", encoding="utf-8")
                register(client, name, Path(folder), experiment_id, number)
                client.delete_model_version(name, str(number))
            model_dir = Path(folder, "model")
            with tarfile.open(ASSETS / "models" / model["file"]) as tar:
                tar.extractall(model_dir, filter="data")
            register(client, name, model_dir, experiment_id, target, model.get("tags"))
        log(f"{name} v{target}: 등록")


def seed_admin() -> None:
    email, password = os.environ.get("ADMIN_EMAIL"), os.environ.get("ADMIN_PASSWORD")
    if not (email and password):
        return
    import db
    from auth import hash_password
    with db.connect_db() as conn:
        conn.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (%s, %s, 'admin') "
            "ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash, role = 'admin'",
            (email, hash_password(password)),
        )
    log(f"데모 관리자 계정: {email}")


def default_models(client: MlflowClient) -> tuple[str, str]:
    def ready(name):
        return {int(v.version): v for v in client.search_model_versions(f"name = '{name}'") if v.status == "READY"}

    detectors, classifiers = ready(DETECTOR), ready(CLASSIFIER)
    detector = max(detectors, default=1)
    tagged = detectors[detector].tags.get("classifier_version") if detectors else None
    classifier = int(tagged) if tagged and tagged.isdigit() and int(tagged) in classifiers else max(classifiers, default=1)
    return f"models:/{DETECTOR}/{detector}", f"models:/{CLASSIFIER}/{classifier}"


def main() -> None:
    import db
    client = MlflowClient(os.environ["MLFLOW_TRACKING_URI"])
    wait("DB", lambda: db.connect_db().close())
    wait("MLflow", lambda: client.search_experiments(max_results=1))
    try:
        import_models(client)
    except Exception as error:   # 모델이 없어도 화면 · 로그인 · 재학습은 동작하므로 API 는 띄운다
        log(f"운영 모델 등록 실패: {error}")
    seed_admin()
    detector, classifier = default_models(client)
    log(f"기본 모델: {detector} + {classifier}")
    print(f"export MODEL_URI={detector} CLASSIFIER_MODEL_URI={classifier}")


if __name__ == "__main__":
    main()
