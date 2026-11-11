import base64
import math
from numbers import Real
from io import BytesIO
from mlflow import MlflowClient

import numpy as np
from PIL import Image
import mlflow
import mlflow.pyfunc
import pandas as pd
from pandas.api.types import is_bool


# MLflow 서버에 연결하고 지정된 모델을 로드한다.
def load_model(tracking_uri, model_uri):
    if not isinstance(model_uri, str) or not model_uri.strip():
        raise ValueError("MODEL_URI가 설정되지 않았습니다.")

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)


# 숫자 문자열·불리언·NaN·무한대를 거부한다.
def validate_number(value, name):
    if is_bool(value) or not isinstance(value, Real):
        raise ValueError(f"{name}은 숫자여야 합니다.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{name}은 유한한 숫자여야 합니다.")

    return number


# 모델 출력을 API에서 사용하는 형식으로 검증·변환한다.
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


# 이미지 1장을 모델 입력 계약에 맞춰 전달한다.
def predict_image(model, image_bytes):
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("이미지는 비어 있지 않은 bytes여야 합니다.")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    model_input = pd.DataFrame({
        "image_base64": [image_base64]
    })

    result = model.predict(model_input)
    return validate_prediction(result)

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

# 새 모델의 입력 변환 → 추론 → 출력 검증을 연결한다.
def predict_tensor_image(model, image_bytes):
    model_input = build_tensor_input(image_bytes)
    result = model.predict(model_input)
    return validate_tensor_prediction(result)

def get_model_candidates(tracking_uri, model_name):
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(
        f"name = '{model_name}'"
    )

    candidates = []

    for version in versions:
        if version.status != "READY":
            continue

        metrics = {}
        if version.run_id:
            run = client.get_run(version.run_id)
            metrics = run.data.metrics
        candidates.append({
            "version": int(version.version),
            "run_id": version.run_id,
            "recall": metrics.get("val_recall"),
            "precision": metrics.get("val_precision"),
        })

    return candidates

def select_latest_candidate(candidates):
    if not candidates:
        raise ValueError("선택할 수 있는 모델 버전이 없습니다.")

    return max(candidates, key=lambda candidate: candidate["version"])

def select_best_recall_candidate(candidates):
    eligible = []

    for candidate in candidates:
        recall = candidate.get("recall")

        if (
            is_bool(recall)
            or not isinstance(recall, Real)
            or not math.isfinite(recall)
            or not 0 <= recall <= 1
        ):
            continue

        eligible.append(candidate)

    if not eligible:
        raise ValueError("유효한 Recall이 있는 모델이 없습니다.")

    def ranking(candidate):
        precision = candidate.get("precision")

        if (
            is_bool(precision)
            or not isinstance(precision, Real)
            or not math.isfinite(precision)
            or not 0 <= precision <= 1
        ):
            precision = -1

        return (
            candidate["recall"],
            precision,
            candidate["version"],
        )

    return max(eligible, key=ranking)
