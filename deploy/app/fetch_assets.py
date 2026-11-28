"""app 이미지를 빌드할 때 GitHub Release 에서 부분 데이터셋과 모델 파일을 받는다.

왜 빌드할 때 받는가
  - NHN Object Storage 는 부트캠프 계정이 끝나면 사라지므로, 지원 시점에도 동작하도록 GitHub Release 에 둔다.
  - 빌드할 때 받아 이미지에 넣으면, 빌드한 뒤에는 인터넷 없이 실행된다.
  - sha256 을 확인해, 누가 빌드해도 같은 데이터 · 모델로 실행된다.

목록은 deploy/app/assets.json 하나에서 관리한다. sha256 이 비어 있으면 아직 올리지 않은 것으로 보고 건너뛴다.

사용: python fetch_assets.py <assets.json> <저장 폴더>
  <저장 폴더>/dataset/normal, anomaly    부분 데이터셋 (압축을 푼 상태)
  <저장 폴더>/models/<파일 이름>          모델 묶음 (처음 실행할 때 deploy/app/bootstrap.py 가 MLflow 에 등록)
"""
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path


def download(url: str, target: Path, sha256: str) -> None:
    print(f"받는 중: {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, open(target, "wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        target.unlink()
        sys.exit(f"sha256 이 다릅니다: {target.name} (기대 {sha256}, 실제 {digest.hexdigest()})")


def main() -> None:
    manifest_path, root = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest["release_url"].rstrip("/")

    dataset_dir = root / "dataset"
    (dataset_dir / "normal").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "anomaly").mkdir(parents=True, exist_ok=True)
    dataset = manifest.get("dataset") or {}
    if dataset.get("sha256"):
        archive = root / dataset["file"]
        download(f"{base}/{dataset['file']}", archive, dataset["sha256"])
        with tarfile.open(archive) as tar:
            tar.extractall(dataset_dir, filter="data")   # 압축 안의 경로가 폴더 밖으로 나가지 못하게
        archive.unlink()
        counts = {name: len(list((dataset_dir / name).glob("*.png"))) for name in ("normal", "anomaly")}
        print(f"부분 데이터셋: 정상 {counts['normal']}장, 결함 {counts['anomaly']}장")
    else:
        print("부분 데이터셋: assets.json 에 sha256 이 없어 건너뜀 (재학습하려면 데이터가 필요)")

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model in manifest.get("models", []):
        if not model.get("sha256"):
            print(f"모델 {model.get('name')} v{model.get('version')}: sha256 이 없어 건너뜀")
            continue
        download(f"{base}/{model['file']}", models_dir / model["file"], model["sha256"])

    shutil.copy(manifest_path, root / "assets.json")   # 실행할 때 bootstrap.py 가 같은 목록을 읽는다


if __name__ == "__main__":
    main()
