"""운영에서 쓰던 MLflow 모델을 GitHub Release 에 올릴 압축 파일로 만든다. (팀이 한 번만 실행)

모델 폴더(MLmodel 파일이 있는 폴더)를 그대로 묶는다. 처음 실행할 때 deploy/app/bootstrap.py 가
이 묶음을 로컬 MLflow 에 운영 때와 같은 버전 번호로 등록한다.

모델 폴더를 얻는 방법은 둘 중 하나:
  --from-mlflow models:/Zero-Defects/8   운영 MLflow 에 접속할 수 있을 때 (MLFLOW_TRACKING_URI 와 저장소 키 필요)
  --from-dir <폴더>                      Object Storage 에서 모델 폴더를 직접 내려받았을 때

사용 예:
  python deploy/app/pack_model.py --name Zero-Defects --version 8 --from-dir ./zero-defects-v8 \
      --tag ui_selectable=true --tag classifier_version=3
출력: <이름>-v<버전>.tar.gz 와 deploy/app/assets.json 의 models 목록에 넣을 항목
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, choices=["Zero-Defects", "Zero-Defects-Classifier"])
parser.add_argument("--version", required=True, type=int, help="운영 MLflow 에서의 버전 번호 (로컬에도 같은 번호로 등록)")
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--from-mlflow", metavar="MODEL_URI")
source.add_argument("--from-dir", type=Path)
parser.add_argument("--tag", action="append", default=[], help="모델 버전 태그 (예: ui_selectable=true)")
args = parser.parse_args()

if args.from_mlflow:
    import mlflow
    folder = Path(mlflow.artifacts.download_artifacts(artifact_uri=args.from_mlflow))
else:
    folder = args.from_dir
if not (folder / "MLmodel").is_file():
    raise SystemExit(f"{folder} 에 MLmodel 파일이 없습니다 (MLflow 모델 폴더가 아님)")

out = Path(f"{args.name.lower()}-v{args.version}.tar.gz")
with tarfile.open(out, "w:gz") as tar:
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        tar.add(path, arcname=path.relative_to(folder).as_posix())

entry = {
    "name": args.name,
    "version": args.version,
    "file": out.name,
    "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "tags": dict(tag.split("=", 1) for tag in args.tag),
}
print(f"만듦: {out} ({out.stat().st_size / 1e6:.1f}MB)")
print("deploy/app/assets.json 의 models 목록에 넣을 항목:")
print(json.dumps(entry, ensure_ascii=False, indent=2))
