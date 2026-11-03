from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="DefectCatch API")


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
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기 초과")

    return {
        "filename": file.filename,
        "model_name": "dummy",
        "prediction": "not_evaluated",
        "anomaly_score": None
    }
