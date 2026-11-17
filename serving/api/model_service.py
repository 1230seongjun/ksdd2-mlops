import json
import math
import re
import sys
import threading
import yaml
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import mlflow.pyfunc
from mlflow.artifacts import download_artifacts
import pandas as pd
from pandas.api.types import is_bool


# 모델마다 같은 이름의 코드(config, data, model)를 담고 있어,
# 모델을 올릴 때마다 먼저 올린 모델의 모듈을 비워 서로 섞이지 않게 한다.
SHARED_MODULES = ("config", "data", "model", "storage", "evaluation")
load_lock = threading.Lock()


def clear_shared_modules():
    for name in SHARED_MODULES:
        sys.modules.pop(name, None)


# 학습 PC의 절대경로가 기록된 model_code_path는 파일명만 남기고,
# artifacts 경로의 역슬래시는 슬래시로 바꿔 로드한다.
@lru_cache(maxsize=8)
def load_model(tracking_uri, model_uri):
    if not isinstance(model_uri, str) or not model_uri.strip():
        raise ValueError("MODEL_URI가 설정되지 않았습니다.")

    mlflow.set_tracking_uri(tracking_uri)
    local = Path(download_artifacts(artifact_uri=model_uri))
    mlmodel = local / "MLmodel"
    info = yaml.safe_load(mlmodel.read_text(encoding="utf-8"))
    flavor = info.get("flavors", {}).get("python_function", {})
    changed = False

    code_path = flavor.get("model_code_path")
    if code_path:
        flavor["model_code_path"] = re.split(r"[\\/]", code_path)[-1]
        changed = True

    for artifact in (flavor.get("artifacts") or {}).values():
        if "path" in artifact:
            artifact["path"] = artifact["path"].replace("\\", "/")
            changed = True

    if changed:
        mlmodel.write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    with load_lock:
        clear_shared_modules()
        try:
            return mlflow.pyfunc.load_model(str(local))
        finally:
            clear_shared_modules()

def validate_number(value, name):
    if is_bool(value) or not isinstance(value, Real):
        raise ValueError(f"{name}은 숫자여야 합니다.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{name}은 유한한 숫자여야 합니다.")

    return number


def validate_prediction(result):
    if not isinstance(result, pd.DataFrame):
        raise ValueError("모델 출력은 DataFrame이어야 합니다.")

    if len(result) != 1:
        raise ValueError("예측 결과는 정확히 1행이어야 합니다.")

    if not result.columns.is_unique:
        raise ValueError("모델 출력에 중복 컬럼이 있습니다.")

    required = {"score", "threshold", "is_anomaly"}
    if not required.issubset(result.columns):
        raise ValueError("모델 출력 필수 컬럼이 누락되었습니다.")

    # 컬럼별로 추출하여 행 단위 자료형 변환을 피한다.
    score = validate_number(result["score"].iloc[0], "score")
    threshold = validate_number(
        result["threshold"].iloc[0], "threshold"
    )
    is_anomaly = result["is_anomaly"].iloc[0]

    # 문자열 "False"나 숫자 0/1을 임의로 불리언으로 변환하지 않는다.
    if not is_bool(is_anomaly):
        raise ValueError("is_anomaly는 불리언이어야 합니다.")

    return {
        "anomaly_score": score,
        "threshold": threshold,
        "is_anomaly": bool(is_anomaly),
    }


def build_tensor_input(image_bytes):
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("이미지는 비어 있지 않은 bytes여야 합니다.")

    with Image.open(BytesIO(image_bytes)) as image:
        if image.size != (184, 597):
            raise ValueError("이미지는 가로 184px, 세로 597px이어야 합니다.")

        rgb_image = image.convert("RGB")
        image_array = np.array(rgb_image, dtype=np.uint8)
    return np.expand_dims(image_array, axis=0)

def validate_tensor_prediction(result):
    if not isinstance(result, pd.DataFrame):
        raise ValueError("모델 출력은 DataFrame이어야 합니다.")

    if not result.columns.is_unique:
        raise ValueError("모델 출력에 중복 컬럼이 있습니다.")

    required = {"anomaly_score", "is_defect", "threshold"}
    if not required.issubset(result.columns):
        raise ValueError("새 모델 출력 필수 컬럼이 누락되었습니다.")

    # 필요한 컬럼만 선택하고 기존 검증 함수의 이름에 맞춘다.
    normalized = result[
        ["anomaly_score", "is_defect", "threshold"]
    ].rename(columns={
        "anomaly_score": "score",
        "is_defect": "is_anomaly",
    })

    return validate_prediction(normalized)

# 분류 모델의 색 이름을 화면의 결함 종류 id로 바꾼다. 청색은 하나로 합친다.
DEFECT_TYPES = {
    "갈적색결함": "brown",
    "흑색 결함": "black",
    "밝은 청색 결함": "blue",
    "어두운 청색 결함": "blue",
    "청색 결함": "blue",
    "표면 결함": "surface",
}


def parse_confidence(result):
    if "confidence" not in result.columns:
        return None

    value = result["confidence"].iloc[0]

    if is_bool(value):
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None

    return confidence


def parse_regions(result):
    if "regions" not in result.columns:
        return []

    value = result["regions"].iloc[0]

    try:
        regions = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return []

    return regions if isinstance(regions, list) else []

def build_boxes(regions, image_width, image_height):
    boxes = []

    for region in regions:
        if not isinstance(region, dict):
            continue

        bbox = region.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if not all(isinstance(value, Real) and not is_bool(value) for value in bbox):
            continue

        x, y, w, h = bbox
        if not all(math.isfinite(value) for value in bbox):
            continue
        if w <= 0 or h <= 0:
            continue

        left = max(0, x)
        top = max(0, y)
        right = min(image_width, x + w)
        bottom = min(image_height, y + h)

        if right <= left or bottom <= top:
            continue

        x, y = left, top
        w, h = right - left, bottom - top

        score = region.get("score")
        if not (isinstance(score, Real) and not is_bool(score) and 0 <= score <= 1):
            score = None

        boxes.append({
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "score": float(score) if score is not None else None,
        })

    return boxes

def parse_classification(result):
    if not isinstance(result, pd.DataFrame) or len(result) != 1:
        raise ValueError("분류 모델 출력이 올바르지 않습니다.")

    if "color" not in result.columns:
        raise ValueError("분류 모델 출력 필수 컬럼이 누락되었습니다.")

    color = str(result["color"].iloc[0])
    defect_type = DEFECT_TYPES.get(color)

    return {
        "defect_type": defect_type,
        "defect_subtype": color if defect_type else None,
        "confidence": parse_confidence(result),
        "regions": parse_regions(result),
    }


# 모델이 약속한 입력·출력 형식(계약)별로 입력 만들기와 출력 해석을 묶는다.
@dataclass(frozen=True)
class Contract:
    kind: str
    prepare: Callable
    parse: Callable


CONTRACTS = {
    "tensor_v1": Contract("detector", build_tensor_input, validate_tensor_prediction),
    "classifier_v1": Contract("classifier", build_tensor_input, parse_classification),
}


def run_model(model, contract_name, image_bytes):
    contract = CONTRACTS[contract_name]
    return contract.parse(model.predict(contract.prepare(image_bytes)))
