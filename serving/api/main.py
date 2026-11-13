
import os
import time
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

# 별도 파일에서 MLflow 모델 관련 함수 가져오기
from model_service import (
    get_model_candidates,
    load_model,
    predict_image,
    predict_tensor_image,
    select_best_recall_candidate,
    select_latest_candidate,
)

# ==========================================
# 1. 기본 설정
# ==========================================

# 서버 오류와 실행 상태를 기록하는 로거
logger = logging.getLogger(__name__)

# MLflow 서버 주소
# 환경변수가 없으면 기본 주소 사용
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

# 모델 선택은 배포 환경변수로 지정한다.
MODEL_URI = os.getenv("MODEL_URI", "").strip()

# 이미지 업로드 최대 크기: 5MB
MAX_FILE_SIZE = 5 * 1024 * 1024


# ==========================================
# 2. FastAPI 시작 및 모델 로드
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # 서버 시작 전 모델 초기화
    app.state.model = None

    try:
        # model_service.py에서 MLflow 모델 로드
        # 서버 시작 시 한 번만 실행
        app.state.model = load_model(
            MLFLOW_TRACKING_URI,
            MODEL_URI
        )

        logger.info("Model loaded: %s", MODEL_URI)

    except Exception:
        # 모델 로딩 실패 시 오류 기록
        # API 서버 자체는 실행 유지
        logger.exception("Model loading failed")

    # FastAPI 요청 처리 시작
    yield

    # 서버 종료 시 모델 참조 해제
    app.state.model = None


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="DefectCatch API",
    lifespan=lifespan
)


# ==========================================
# 3. Prometheus 메트릭 정의
# ==========================================

# /predict HTTP 응답 수 측정
# status_code별로 요청 수 기록
predict_requests = Counter(
    "defectcatch_predict_requests_total",
    "Number of HTTP responses for prediction requests",
    ["status_code"],
)

# /predict 전체 처리 시간 측정
predict_duration = Histogram(
    "defectcatch_predict_request_duration_seconds",
    "Time spent handling prediction requests in seconds",
)


# ==========================================
# 4. Prometheus 미들웨어
# ==========================================

@app.middleware("http")
async def record_predict_metrics(
    request: Request,
    call_next
):

    # /predict 요청만 측정
    if request.url.path != "/predict":
        return await call_next(request)

    # 요청 시작 시간
    start_time = time.perf_counter()
    status_code = 500

    try:
        # 실제 API 요청 처리
        response = await call_next(request)

        # HTTP 응답 상태 코드 저장
        status_code = response.status_code

        return response

    finally:
        # 전체 처리 시간 계산
        duration = time.perf_counter() - start_time

        # 상태 코드별 요청 수 증가
        predict_requests.labels(
            status_code=str(status_code)
        ).inc()

        # 요청 처리 시간 기록
        predict_duration.observe(duration)


# ==========================================
# 5. Prometheus 메트릭 제공
# ==========================================

@app.get("/metrics")
def metrics():

    # Prometheus가 수집할 메트릭 반환
    return Response(
        content=generate_latest(),
        headers={
            "Content-Type": CONTENT_TYPE_LATEST
        },
    )


# ==========================================
# 6. Health Check
# ==========================================

@app.get("/health")
def health():

    # MLflow 모델이 메모리에 로드됐는지 확인
    loaded = app.state.model is not None

    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded
    }

@app.get("/ready")
def ready():
    # 모델이 없으면 추론 준비가 안 된 상태
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    # 모델 객체가 있으면 준비 상태로 응답
    return {"status": "ready"}
# ==========================================
# 7. 이미지 예측 API
# ==========================================
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

    # 2. 이미지 파일 형식 확인
    if file.content_type not in (
        "image/jpeg",
        "image/png"
    ):
        raise HTTPException(
            status_code=415,
            detail="JPEG 또는 PNG만 지원합니다."
        )

    # 3. 이미지 파일 읽기
    # 최대 허용 크기보다 1바이트 더 읽어 초과 여부 확인
    data = await file.read(MAX_FILE_SIZE + 1)

    # 빈 파일 검사
    if not data:
        raise HTTPException(
            status_code=400,
            detail="빈 파일입니다."
        )

    # 5MB 초과 파일 차단
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="파일 크기 초과"
        )

    # 4. 실제 이미지 파일인지 확인
    try:
        with Image.open(BytesIO(data)) as image:

            # JPEG / PNG 형식 검사
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("지원하지 않는 이미지 형식")

            # 이미지 데이터 유효성 검사
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

    # 1. 요청에 사용할 모델 선택
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

    # 5. MLflow 모델 추론
    try:
        result = await run_in_threadpool(
            predict_tensor_image,
            selected_model,
            data,
        )
    except Exception:
        # 추론 실패 시 로그 기록
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="모델 추론 실패"
        )

    # 6. 정상 / 불량 판정
    prediction = (
        "defect"
        if result["is_anomaly"]
        else "normal"
    )

    # 7. 검사 결과를 JSON으로 웹에 반환
    return {
        "filename": file.filename,
        "model_name": selected_model_name,
        "prediction": prediction,
        "anomaly_score": result["anomaly_score"],
        "threshold": result["threshold"],
        "is_anomaly": result["is_anomaly"]
    }

