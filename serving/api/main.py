import os
import time
import asyncio
import logging

from io import BytesIO
from contextlib import asynccontextmanager

from PIL import Image, UnidentifiedImageError

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import Response

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from starlette.concurrency import run_in_threadpool

from model_service import load_model, run_model


logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

MODEL_URI = os.getenv("MODEL_URI", "").strip()

# 결함 종류 분류 모델은 선택 사항이다. 비어 있거나 로드에 실패하면 종류 없이 동작한다.
CLASSIFIER_MODEL_URI = os.getenv("CLASSIFIER_MODEL_URI", "").strip()

# 신규 모델은 운영 모델과 나란히 올려 두고 고를 수 있게 한다. 비어 있으면 선택지에 나오지 않는다.
CHALLENGER_MODEL_URI = os.getenv("CHALLENGER_MODEL_URI", "").strip()
CHALLENGER_CLASSIFIER_MODEL_URI = os.getenv(
    "CHALLENGER_CLASSIFIER_MODEL_URI",
    ""
).strip()

# 신규 모델은 분류까지 올라와야 선택할 수 있고, 운영 모델은 분류 없이도 동작한다.
MODEL_SETS = {
    "champion": {
        "label": "운영 모델",
        "detector_uri": MODEL_URI,
        "classifier_uri": CLASSIFIER_MODEL_URI,
        "require_classifier": False,
    },
}

if CHALLENGER_MODEL_URI:
    MODEL_SETS["challenger"] = {
        "label": "신규 모델",
        "detector_uri": CHALLENGER_MODEL_URI,
        "classifier_uri": CHALLENGER_CLASSIFIER_MODEL_URI,
        "require_classifier": True,
    }

MAX_FILE_SIZE = 5 * 1024 * 1024

DETECTOR_CONTRACT = "tensor_v1"
CLASSIFIER_CONTRACT = "classifier_v1"


def set_status(app: FastAPI, key: str):
    spec = MODEL_SETS[key]
    state = app.state.sets[key]

    if state["detector"] is not None:
        needs_classifier = spec["require_classifier"] and spec["classifier_uri"]

        if not needs_classifier or state["classifier"] is not None:
            return "ready"

    return "failed" if state["failed"] else "loading"


async def load_part(app: FastAPI, key: str, role: str, uri: str):
    started = time.perf_counter()

    try:
        app.state.sets[key][role] = await run_in_threadpool(
            load_model,
            MLFLOW_TRACKING_URI,
            uri
        )

        logger.info(
            "Model loaded: %s %s %s (%.0fs)",
            key,
            role,
            uri,
            time.perf_counter() - started
        )

    except Exception:
        app.state.sets[key]["failed"] = True
        logger.exception("Model loading failed: %s %s %s", key, role, uri)


# 한 번에 하나씩 올려 다운로드와 메모리 사용이 겹치지 않게 한다.
async def load_in_background(app: FastAPI):
    for key, spec in MODEL_SETS.items():
        for role in ("detector", "classifier"):
            uri = spec[f"{role}_uri"]
            state = app.state.sets[key]

            if uri and state[role] is None and not state["failed"]:
                await load_part(app, key, role, uri)


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.sets = {
        key: {"detector": None, "classifier": None, "failed": False}
        for key in MODEL_SETS
    }
    started = time.perf_counter()

    try:
        app.state.sets["champion"]["detector"] = load_model(
            MLFLOW_TRACKING_URI,
            MODEL_URI
        )

        logger.info(
            "Model loaded: champion detector %s (%.0fs)",
            MODEL_URI,
            time.perf_counter() - started
        )

    except Exception:
        # API 서버 자체는 실행 유지
        app.state.sets["champion"]["failed"] = True
        logger.exception("Model loading failed")

    # 나머지 모델은 서버 시작(/ready)을 늦추지 않도록 백그라운드에서 올린다.
    loading_task = asyncio.create_task(load_in_background(app))

    yield

    loading_task.cancel()

    app.state.sets = {}


app = FastAPI(
    title="DefectCatch API",
    lifespan=lifespan
)


predict_requests = Counter(
    "defectcatch_predict_requests_total",
    "Number of HTTP responses for prediction requests",
    ["status_code"],
)

predict_duration = Histogram(
    "defectcatch_predict_request_duration_seconds",
    "Time spent handling prediction requests in seconds",
)


@app.middleware("http")
async def record_predict_metrics(
    request: Request,
    call_next
):

    if request.url.path != "/predict":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)

        status_code = response.status_code

        return response

    finally:
        duration = time.perf_counter() - start_time

        predict_requests.labels(
            status_code=str(status_code)
        ).inc()

        predict_duration.observe(duration)


@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        headers={
            "Content-Type": CONTENT_TYPE_LATEST
        },
    )


@app.get("/health")
def health():

    champion = app.state.sets["champion"]
    loaded = champion["detector"] is not None

    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "classifier_loaded": champion["classifier"] is not None
    }

@app.get("/ready")
def ready():
    if app.state.sets["champion"]["detector"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    return {"status": "ready"}


@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "id": key,
                "label": spec["label"],
                "model_name": spec["detector_uri"],
                "classifier_name": spec["classifier_uri"] or None,
                "status": set_status(app, key),
            }
            for key, spec in MODEL_SETS.items()
        ]
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_choice: str = Form("champion"),
):
    if model_choice not in MODEL_SETS:
        raise HTTPException(
            status_code=422,
            detail="지원하지 않는 모델 선택값입니다."
        )

    if file.content_type not in (
        "image/jpeg",
        "image/png"
    ):
        raise HTTPException(
            status_code=415,
            detail="JPEG 또는 PNG만 지원합니다."
        )

    # 최대 허용 크기보다 1바이트 더 읽어 초과 여부 확인
    data = await file.read(MAX_FILE_SIZE + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="빈 파일입니다."
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="파일 크기 초과"
        )

    try:
        with Image.open(BytesIO(data)) as image:

            if image.format not in ("JPEG", "PNG"):
                raise ValueError("지원하지 않는 이미지 형식")

            image.verify()

            # image.size 순서는 (가로, 세로)
            if image.size != (184, 597):
                width, height = image.size

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "이미지 크기가 맞지 않습니다. "
                        "필요 크기: 가로 184px × 세로 597px, "
                        f"입력 크기: 가로 {width}px × 세로 {height}px"
                    ),
                )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="올바른 이미지가 아닙니다."
        )

    status = set_status(app, model_choice)

    if status == "loading":
        raise HTTPException(
            status_code=503,
            detail="모델을 불러오는 중입니다. 잠시 후 다시 시도해 주세요.",
            headers={"Retry-After": "30"},
        )

    if status != "ready":
        raise HTTPException(
            status_code=503,
            detail="선택한 모델을 불러오지 못했습니다."
        )

    spec = MODEL_SETS[model_choice]
    state = app.state.sets[model_choice]
    selected_model = state["detector"]
    classifier = state["classifier"]
    selected_model_name = spec["detector_uri"]

    try:
        result = await run_in_threadpool(
            run_model,
            selected_model,
            DETECTOR_CONTRACT,
            data,
        )
    except Exception:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="모델 추론 실패"
        )

    prediction = (
        "defect"
        if result["is_anomaly"]
        else "normal"
    )

    # 분류가 실패해도 판정 결과는 그대로 반환한다.
    defect = {"defect_type": None, "defect_subtype": None}
    classified = False

    if result["is_anomaly"] and classifier is not None:
        try:
            defect = await run_in_threadpool(
                run_model,
                classifier,
                CLASSIFIER_CONTRACT,
                data,
            )
            classified = True
        except Exception:
            logger.exception("Defect classification failed")

    return {
        "filename": file.filename,
        "model_name": selected_model_name,
        "prediction": prediction,
        "anomaly_score": result["anomaly_score"],
        "threshold": result["threshold"],
        "is_anomaly": result["is_anomaly"],
        "defect_type": defect["defect_type"],
        "defect_subtype": defect["defect_subtype"],
        "classifier_name": spec["classifier_uri"] if classified else None
    }
