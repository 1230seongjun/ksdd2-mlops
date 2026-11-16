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

from model_service import (
    classify_defect,
    get_model_candidates,
    load_model,
    predict_tensor_image,
    select_best_recall_candidate,
    select_latest_candidate,
)


logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

MODEL_URI = os.getenv("MODEL_URI", "").strip()

# 결함 종류 분류 모델은 선택 사항이다. 비어 있거나 로드에 실패하면 종류 없이 동작한다.
CLASSIFIER_MODEL_URI = os.getenv("CLASSIFIER_MODEL_URI", "").strip()

MAX_FILE_SIZE = 5 * 1024 * 1024


async def load_classifier(app: FastAPI):
    try:
        app.state.classifier = await run_in_threadpool(
            load_model,
            MLFLOW_TRACKING_URI,
            CLASSIFIER_MODEL_URI
        )

        logger.info("Classifier loaded: %s", CLASSIFIER_MODEL_URI)

    except Exception:
        logger.exception("Classifier loading failed")


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = None
    app.state.classifier = None

    try:
        app.state.model = load_model(
            MLFLOW_TRACKING_URI,
            MODEL_URI
        )

        logger.info("Model loaded: %s", MODEL_URI)

    except Exception:
        # API 서버 자체는 실행 유지
        logger.exception("Model loading failed")

    # 분류 모델은 서버 시작(/ready)을 늦추지 않도록 백그라운드에서 올린다.
    classifier_task = None

    if CLASSIFIER_MODEL_URI:
        classifier_task = asyncio.create_task(load_classifier(app))

    yield

    if classifier_task is not None:
        classifier_task.cancel()

    app.state.model = None
    app.state.classifier = None


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

    loaded = app.state.model is not None

    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "classifier_loaded": app.state.classifier is not None
    }

@app.get("/ready")
def ready():
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    return {"status": "ready"}
@app.get("/models")
async def list_models():
    try:
        candidates = await run_in_threadpool(
            get_model_candidates,
            MLFLOW_TRACKING_URI,
            "Zero-Defects",
        )

        latest = select_latest_candidate(candidates)
        best = select_best_recall_candidate(candidates)

        return {
            "model_name": "Zero-Defects",
            "latest": latest,
            "best_recall": best,
        }

    except Exception:
        logger.exception("Model candidate lookup failed")
        raise HTTPException(
            status_code=503,
            detail="모델 후보를 조회할 수 없습니다."
        )

def load_selected_candidate(model_choice):
    """선택 기준에 맞는 등록 모델 버전을 찾아 로드한다."""
    candidates = get_model_candidates(
        MLFLOW_TRACKING_URI,
        "Zero-Defects",
    )

    if model_choice == "latest":
        candidate = select_latest_candidate(candidates)
    elif model_choice == "best_recall":
        candidate = select_best_recall_candidate(candidates)
    else:
        raise ValueError("지원하지 않는 모델 선택값입니다.")

    model_uri = f"models:/Zero-Defects/{candidate['version']}"
    model = load_model(MLFLOW_TRACKING_URI, model_uri)

    return model, model_uri

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_choice: str = Form("current"),
):
    if model_choice not in ("current", "latest", "best_recall"):
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

    if model_choice == "current":
        if app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="현재 모델이 로드되지 않았습니다."
            )

        selected_model = app.state.model
        selected_model_name = MODEL_URI

    else:
        try:
            selected_model, selected_model_name = await run_in_threadpool(
                load_selected_candidate,
                model_choice,
            )

        except Exception:
            logger.exception("Selected model loading failed")

            raise HTTPException(
                status_code=503,
                detail="선택한 모델을 불러오지 못했습니다."
            )

    try:
        result = await run_in_threadpool(
            predict_tensor_image,
            selected_model,
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

    if result["is_anomaly"] and app.state.classifier is not None:
        try:
            defect = await run_in_threadpool(
                classify_defect,
                app.state.classifier,
                data,
            )
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
        "defect_subtype": defect["defect_subtype"]
    }
