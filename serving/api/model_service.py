import base64

import mlflow
import mlflow.pyfunc
import pandas as pd


# MLflow 서버에 연결하고 저장된 모델을 불러오는 함수
def load_model(tracking_uri, model_uri):

    mlflow.set_tracking_uri(tracking_uri)

    # 모델 로드 후 FastAPI에서 사용하도록 반환
    return mlflow.pyfunc.load_model(model_uri)


# 이미지 데이터를 전달받아 실제 모델 추론을 수행하는 함수
def predict_image(model, image_bytes):

    # 이미지 바이트를 Base64 문자열로 변환
    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    # MLflow 모델 입력 형식에 맞게 변환
    model_input = pd.DataFrame({
        "image_base64": [image_base64]
    })

    # 저장된 모델로 추론 실행
    result = model.predict(model_input)

    # 모델 출력 형식 검증
    if not isinstance(result, pd.DataFrame):
        raise ValueError("모델 출력 형식 오류")

    if len(result) != 1:
        raise ValueError("예측 결과 개수 오류")

    required = {"score", "threshold", "is_anomaly"}

    if not required.issubset(result.columns):
        raise ValueError("모델 출력 컬럼 오류")

    # 모델이 반환한 결과 추출
    row = result.iloc[0]

    return {
        "anomaly_score": float(row["score"]),
        "threshold": float(row["threshold"]),
        "is_anomaly": bool(row["is_anomaly"])
    }

