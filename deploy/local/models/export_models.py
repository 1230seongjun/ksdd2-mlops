"""운영 MLflow 의 등록 모델을 공개 버킷에 올릴 파일과 목록(models.json)으로 내보냅니다. (팀원이 한 번만 실행)

운영 MLflow 와 그 Object Storage 에 접근할 수 있는 곳(mlops 서버, 또는 SSH 터널을 연 PC)에서 실행합니다.
운영 쪽에서는 읽기만 합니다.

  MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python deploy/local/models/export_models.py \
      --out ./models-public \
      --base-url https://<공개 버킷 주소>

결과
  models-public/<모델 이름>/<버전>/...   공개 읽기 버킷에 그대로 올릴 모델 파일
  models-public/models.json             deploy/local/models/models.json 으로 복사해 커밋 (키 · 비밀값 없음)

이어서 할 일
  1) NHN Object Storage 에 모델 전용 버킷을 만들고 공개 읽기로 설정 (운영 · 데이터셋 버킷은 private 유지)
  2) models-public/ 안의 모델 폴더들을 그 버킷 최상위에 올림
  3) models.json 을 deploy/local/models/ 에 복사해 커밋 → 로컬 실행 시 model-import 가 받아 등록
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

MODEL_NAMES = ["Zero-Defects", "Zero-Defects-Classifier"]

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--base-url", required=True, help="공개 읽기 버킷 주소 (모델 폴더들이 이 아래에 놓임)")
parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
args = parser.parse_args()

client = MlflowClient()
manifest = {"base_url": args.base_url.rstrip("/"), "models": []}

for name in args.models:
    registered = client.get_registered_model(name)
    aliases = {}
    for alias, version in (registered.aliases or {}).items():
        aliases.setdefault(int(version), []).append(alias)
    versions = sorted(client.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version))
    entry = {"name": name, "description": registered.description or "", "versions": []}
    for version in versions:
        number = int(version.version)
        target = args.out / name / str(number)
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{number}", dst_path=str(target))
        files = []
        for path in sorted(p for p in target.rglob("*") if p.is_file()):
            files.append({
                "path": path.relative_to(target).as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        run = {}
        if version.run_id:
            info = client.get_run(version.run_id)
            run = {"name": info.info.run_name, "params": info.data.params, "metrics": info.data.metrics}
        entry["versions"].append({
            "version": number,
            "tags": dict(version.tags),
            "description": version.description or "",
            "aliases": aliases.get(number, []),
            "run": run,
            "files": files,
        })
        print(f"{name} v{number}: {len(files)}개 파일")
    manifest["models"].append(entry)

args.out.mkdir(parents=True, exist_ok=True)
(args.out / "models.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"완료: {args.out / 'models.json'}")
