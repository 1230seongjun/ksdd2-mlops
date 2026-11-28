"""재학습 직전에 학습 폴더를 새로 만든다: 기본 부분 데이터셋 + 작업자가 확정한 검사 이미지.

model/run_training.sh 가 PREPARE_DATASET_CMD 로 호출한다 (로컬 실행에서만 설정됨).

왜 작업자가 확정한 이미지만 쓰는가
  모델이 혼자 판정한 이미지까지 학습에 넣으면, 모델이 자신의 오판을 정답으로 다시 배운다.
  그래서 재검토에서 작업자가 확정한(review_status = 'finalized') 이미지만 쓴다.
    - 정상으로 확정 → normal/   (모델이 불량으로 잘못 본 정상: 이상 탐지 학습에 가장 도움이 되는 이미지)
    - 결함으로 확정 → anomaly/  (이상 탐지는 결함을 검증 · 테스트에만 쓴다)

학습 폴더는 매번 지우고 다시 만들어, 어떤 데이터로 학습했는지가 이 실행의 로그 한 곳에 남게 한다.

사용: python build_dataset.py <anomaly|classifier>
환경변수
  DATASET_ROOT         학습 폴더 (학습 코드 config.py 도 같은 값을 읽는다)
  BASE_DATASET_DIR     이미지에 들어 있는 부분 데이터셋 (normal/, anomaly/)
  INFERENCE_IMAGE_DIR  추론 API 가 저장한 검사 이미지 (<요청 ID>.png)
  INFERENCE_DB_*       검사 이력 DB (추론 API 와 같은 값)
"""
import os
import shutil
import sys
from pathlib import Path

import psycopg

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def reviewed_images(image_dir: Path) -> list[tuple[str, Path]]:
    conninfo = {
        "host": os.environ["INFERENCE_DB_HOST"],
        "port": os.environ.get("INFERENCE_DB_PORT", "5432"),
        "dbname": os.environ["INFERENCE_DB_NAME"],
        "user": os.environ["INFERENCE_DB_USER"],
        "password": os.environ["INFERENCE_DB_PASSWORD"],
    }
    with psycopg.connect(**conninfo) as conn:
        rows = conn.execute(
            "SELECT id::text, reviewed_label FROM inference_requests "
            "WHERE review_status = 'finalized' AND reviewed_label IN ('normal', 'defect')"
        ).fetchall()
    found = []
    for request_id, label in rows:
        image = next((image_dir / f"{request_id}{s}" for s in (".png", ".jpg") if (image_dir / f"{request_id}{s}").exists()), None)
        if image is not None:   # 이미지 저장을 켜기 전에 검사한 요청은 이미지가 없다
            found.append(("normal" if label == "normal" else "anomaly", image))
    return found


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else ""
    if model != "anomaly":
        # 결함 분류 학습에는 팀이 만든 결함 유형 라벨이 필요한데, 공개 여부가 불확실해 부분 데이터셋에 넣지 않았다
        sys.exit("결함 분류 모델은 학습 데이터(팀 라벨)가 공개 데이터셋에 없어 로컬에서 학습하지 않습니다. 이상 탐지 모델을 선택하세요.")

    base = Path(os.environ["BASE_DATASET_DIR"])
    target = Path(os.environ["DATASET_ROOT"]) / "anomaly"
    if target.exists():
        shutil.rmtree(target)
    counts = {}
    for folder in ("normal", "anomaly"):
        (target / folder).mkdir(parents=True)
        files = [p for p in (base / folder).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
        for path in files:
            shutil.copy2(path, target / folder / path.name)
        counts[folder] = [len(files), 0]

    for folder, image in reviewed_images(Path(os.environ["INFERENCE_IMAGE_DIR"]) / "requests"):
        shutil.copy2(image, target / folder / f"review_{image.name}")
        counts[folder][1] += 1

    for folder, (from_base, from_review) in counts.items():
        print(f"[학습 데이터] {folder}: 기본 {from_base}장 + 작업자 확정 검사 이미지 {from_review}장")
    if counts["normal"][0] + counts["normal"][1] == 0:
        sys.exit("정상 이미지가 없어 학습할 수 없습니다 (부분 데이터셋이 이미지에 들어 있는지 확인: deploy/app/assets.json)")


if __name__ == "__main__":
    main()
