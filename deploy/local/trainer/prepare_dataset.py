"""KolektorSDD2 를 로컬 학습용으로 정리합니다.

원저작자 페이지(https://www.vicos.si/resources/kolektorsdd2/)에서 직접 내려받은 zip 을 그대로 넣으면,
  1) 이미지 가운데를 184×597 로 자르고 (검사 API 입력 규격, 팀 데이터 v2 와 같은 방식)
  2) 자른 마스크에 결함 픽셀이 있으면 anomaly, 없으면 normal 로 나눠
  3) --out 폴더(normal/, anomaly/)에 저장하고 (이어서 upload_dataset.py 로 Object Storage 버킷에 올림),
  4) 학습에 쓰지 않은 정상 · 결함 샘플 몇 장을 /samples 에 따로 꺼내 둡니다 (검사 화면에 올려 볼 이미지).

사용 예 (Pod 를 띄운 상태에서):
  podman cp ./KolektorSDD2.zip ksdd2-trainer:/tmp/ksdd2.zip
  podman exec ksdd2-trainer python /app/prepare_dataset.py /tmp/ksdd2.zip --out /tmp/ksdd2 --samples /tmp/samples
  podman exec ksdd2-trainer python /app/upload_dataset.py /tmp/ksdd2
"""
import argparse
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 184, 597


def crop(image: Image.Image) -> Image.Image | None:
    width, height = image.size
    if width < WIDTH or height < HEIGHT:
        return None
    left, top = (width - WIDTH) // 2, (height - HEIGHT) // 2
    return image.crop((left, top, left + WIDTH, top + HEIGHT))


def find_images(root: Path) -> tuple[list[Path], list[Path]]:
    normal, anomaly = [], []
    for mask_path in sorted(root.rglob("*_GT.png")):
        image_path = mask_path.with_name(mask_path.name.replace("_GT", ""))
        if not image_path.exists():
            continue
        with Image.open(mask_path) as mask:
            cropped = crop(mask.convert("L"))
        if cropped is None:
            continue
        (anomaly if np.asarray(cropped).max() > 0 else normal).append(image_path)
    return normal, anomaly


def save(paths: list[Path], folder: Path, prefix: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with Image.open(path) as image:
            crop(image.convert("RGB")).save(folder / f"{prefix}{path.parent.name}_{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="KolektorSDD2 zip 파일 또는 압축을 푼 폴더")
    parser.add_argument("--out", default="/data/ksdd2", help="학습 데이터 저장 위치 (기존 내용은 지움)")
    parser.add_argument("--samples", default="/samples", help="검사 화면에 올려 볼 샘플 저장 위치")
    parser.add_argument("--sample-count", type=int, default=3, help="정상 · 결함 각각 꺼내 둘 샘플 수")
    parser.add_argument("--limit-normal", type=int, help="빠른 확인용: 정상 이미지 수 제한")
    parser.add_argument("--limit-anomaly", type=int, help="빠른 확인용: 결함 이미지 수 제한")
    args = parser.parse_args()

    source = Path(args.source)
    with tempfile.TemporaryDirectory() as temporary:
        if source.is_file():
            print("압축 푸는 중:", source)
            zipfile.ZipFile(source).extractall(temporary)
            root = Path(temporary)
        else:
            root = source

        normal, anomaly = find_images(root)
        if not normal or not anomaly:
            raise SystemExit("KolektorSDD2 이미지와 마스크(*_GT.png)를 찾지 못했습니다. zip 파일을 확인하세요.")

        generator = random.Random(42)
        generator.shuffle(normal)
        generator.shuffle(anomaly)
        count = args.sample_count
        samples_normal, normal = normal[:count], normal[count:]
        samples_anomaly, anomaly = anomaly[:count], anomaly[count:]
        if args.limit_normal:
            normal = normal[:args.limit_normal]
        if args.limit_anomaly:
            anomaly = anomaly[:args.limit_anomaly]

        out = Path(args.out)
        shutil.rmtree(out, ignore_errors=True)
        save(normal, out / "normal")
        save(anomaly, out / "anomaly")

        samples = Path(args.samples)
        if samples.is_dir():
            save(samples_normal, samples, "normal_")
            save(samples_anomaly, samples, "defect_")
            print(f"샘플 {len(samples_normal) + len(samples_anomaly)}장 → {samples}")

    print(f"학습 데이터 → {out}: 정상 {len(normal)}장 / 결함 {len(anomaly)}장")


if __name__ == "__main__":
    main()
