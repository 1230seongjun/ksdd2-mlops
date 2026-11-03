import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

app = FastAPI(title="DefectCatch API")


# 검사 API의 HTTP 응답 수
predict_requests = Counter(
    "defectcatch_predict_requests_total",
    "Number of HTTP responses for prediction requests",
    ["status_code"],
)

# 검사 API의 전체 요청 처리 시간
predict_duration = Histogram(
    "defectcatch_predict_request_duration_seconds",
    "Time spent handling prediction requests in seconds",
)


@app.middleware("http")
async def record_predict_metrics(request: Request, call_next):
    # /predict 요청만 측정한다.
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

        predict_requests.labels(status_code=str(status_code)).inc()
        predict_duration.observe(duration)


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="JPEG 또는 PNG 파일만 지원합니다."
        )

    data = await file.read(5 * 1024 * 1024 + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="빈 파일입니다."
        )

    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="파일 크기 초과"
        )

    return {
        "filename": file.filename,
        "model_name": "dummy",
        "prediction": "not_evaluated",
        "anomaly_score": None
    }
