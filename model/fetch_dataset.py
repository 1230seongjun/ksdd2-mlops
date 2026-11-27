"""팀 데이터셋을 오브젝트 스토리지 버킷에서 한 번 내려받아 로컬 폴더에 둔다.

사용: python fetch_dataset.py <anomaly|classifier> <받을 폴더>
  (보통은 scripts/local.sh fetch-data 가 학습 서버 컨테이너 안에서 두 번 실행한다)

각 모델 폴더의 config.yaml 에 적힌 버킷 위치를 그대로 쓰고, 받을 곳만 바꾼다.
받은 뒤에는 학습할 때 DATASET_ROOT 로 이 폴더를 읽으므로 버킷 키가 더 필요 없다.
접근 키는 환경변수 OBJECT_STORAGE_ACCESS_KEY, OBJECT_STORAGE_SECRET_KEY 로 넣는다.
"""

import dataclasses
import os
import sys
from pathlib import Path

MODEL_DIRS = {"anomaly": "이상치 모델", "classifier": "분류 모델"}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in MODEL_DIRS:
        sys.exit("사용법: python fetch_dataset.py <anomaly|classifier> <받을 폴더>")
    model, target = sys.argv[1], Path(sys.argv[2]).resolve()

    model_dir = Path(__file__).resolve().parent / MODEL_DIRS[model]
    sys.path.insert(0, str(model_dir))   # 모델 폴더의 config.py, storage.py 를 그대로 쓴다
    os.environ.pop("DATASET_ROOT", None)   # 버킷 설정을 읽어야 하므로 로컬 폴더 설정은 끈다
    from config import load_config
    from storage import download_dataset

    config = load_config(model_dir / "config.yaml")
    changes = {"data_source": "object_storage", "cache_dir": target}
    if "bbox_json" in config.__dataclass_fields__:   # 분류 모델은 바운딩 박스 파일도 같은 폴더에 받는다
        changes["bbox_json"] = target / "coco.json"
    config = dataclasses.replace(config, **changes)

    target.mkdir(parents=True, exist_ok=True)
    download_dataset(config)
    print(f"{model} 데이터 받음: {target}")


if __name__ == "__main__":
    main()
