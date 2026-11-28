"""부분 데이터셋을 GitHub Release 에 올릴 압축 파일로 만든다. (팀이 한 번만 실행)

입력 폴더 구조: <폴더>/normal/*.png, <폴더>/anomaly/*.png  (184×597, 원본 GT 마스크로 정상 · 결함만 나눈 이미지)
팀이 만든 결함 유형 라벨 · 바운딩 박스는 공개 여부가 불확실해 넣지 않는다 (docs/local-run-design.md 4절).

출력: ksdd2-subset.tar.gz 와 sha256 → deploy/app/assets.json 의 dataset 항목에 적는다.
압축 안에 KolektorSDD2 출처와 라이선스(CC BY-NC-SA 4.0)를 함께 넣는다 (라이선스의 출처 표시 · 같은 라이선스 조건).

사용: python deploy/app/pack_dataset.py <폴더> [--out ksdd2-subset.tar.gz]
"""
import argparse
import hashlib
import io
import tarfile
from pathlib import Path

from PIL import Image

WIDTH, HEIGHT = 184, 597   # 추론 API 가 받는 이미지 크기
NOTICE = """KolektorSDD2 부분 데이터셋 (불량제로 로컬 실행용)

원본: KolektorSDD2, ViCoS Lab, University of Ljubljana
      https://www.vicos.si/resources/kolektorsdd2/
라이선스: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

변경 사항: 원본 이미지의 가운데를 184×597 로 자르고, 원본 GT 마스크에 결함 픽셀이 있으면 anomaly/, 없으면 normal/ 로 나눴습니다.
이 묶음도 같은 라이선스(CC BY-NC-SA 4.0)로 배포하며, 비상업적 용도로만 사용할 수 있습니다.
"""

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--out", type=Path, default=Path("ksdd2-subset.tar.gz"))
args = parser.parse_args()

files = []
for name in ("normal", "anomaly"):
    images = sorted((args.folder / name).glob("*.png"))
    if not images:
        raise SystemExit(f"{args.folder / name} 에 png 이미지가 없습니다")
    for path in images:
        with Image.open(path) as image:
            if image.size != (WIDTH, HEIGHT):
                raise SystemExit(f"{path}: {image.size} — 추론 API 는 {WIDTH}×{HEIGHT} 만 받습니다")
    files += [(path, f"{name}/{path.name}") for path in images]
    print(f"{name}: {len(images)}장")

with tarfile.open(args.out, "w:gz") as tar:
    data = NOTICE.encode("utf-8")
    info = tarfile.TarInfo("LICENSE-DATASET.txt")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
    for path, name in files:
        tar.add(path, arcname=name)

sha256 = hashlib.sha256(args.out.read_bytes()).hexdigest()
print(f"\n만듦: {args.out} ({args.out.stat().st_size / 1e6:.1f}MB)")
print("deploy/app/assets.json 의 dataset 항목:")
print(f'  "file": "{args.out.name}",\n  "sha256": "{sha256}"')
