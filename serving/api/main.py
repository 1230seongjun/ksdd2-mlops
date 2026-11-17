import os
import time
import uuid
import asyncio
import logging
import rc

from io import BytesIO
from contextlib import asynccontextmanager

from PIL import Image, UnidentifiedImageError

from fastapi import (
    BackgroundTasks,
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

from mlflow import MlflowClient

import db
from model_service import build_boxes, load_model, run_model
from fastapi import Cookie, Depends, HTTPException, Response
from pydantic import BaseModel
from auth import (
    authenticate_user,
    create_invitation,
    create_session,
    get_session_user,
    hash_password,
    register_worker,
    revoke_session
)
from psycopg.errors import UniqueViolation

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
    app.state.registry_models = {}
    app.state.registry_load_task = None
    app.state.registry_prepare_lock = asyncio.Lock()
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
    if app.state.registry_load_task is not None:
        app.state.registry_load_task.cancel()
    app.state.sets = {}


app = FastAPI(
    title="DefectCatch API",
    lifespan=lifespan
)

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
def login(payload: LoginRequest, response: Response):
    with db.connect_db() as conn:
        user_id = authenticate_user(conn, payload.email, payload.password)
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="이메일 또는 비밀번호가 올바르지 않습니다.",
            )

        session_token = create_session(conn, user_id)

    response.set_cookie(
        key="defectcatch_session",
        value=session_token,
        max_age=8 * 60 * 60,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return {"ok": True}

def get_current_user(
    defectcatch_session: str | None = Cookie(default=None),
) -> tuple[int, str]:
    if defectcatch_session is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    with db.connect_db() as conn:
        user = get_session_user(conn, defectcatch_session)

    if user is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    return user

@app.get("/auth/me")
def read_current_user(
    user: tuple[int, str] = Depends(get_current_user),
):
    user_id, role = user

    with db.connect_db() as conn:
        row = conn.execute(
            "SELECT email FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(401, "로그인이 필요합니다.")

    return {
        "id": user_id,
        "role": role,
        "account_name": row[0].split("@", 1)[0],
    }

@app.post("/auth/logout")
def logout(
    response: Response,
    defectcatch_session: str | None = Cookie(default=None),
):
    if defectcatch_session is not None:
        with db.connect_db() as conn:
            revoke_session(conn, defectcatch_session)

    response.delete_cookie(
        key="defectcatch_session",
        path="/",
        secure=True,
        httponly=True,
        samesite="lax",
    )
    return {"ok": True}

class InvitationRequest(BaseModel):
    email: str


@app.post("/auth/invitations", status_code=201)
def issue_invitation(
    payload: InvitationRequest,
    response: Response,
    user: tuple[int, str] = Depends(get_current_user),
):
    user_id, role = user
    if role != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    email = payload.email.strip()
    local_part, _, domain = email.partition("@")
    if (
        not local_part
        or not domain
        or "@" in domain
        or len(email) > 254
        or any(char.isspace() for char in email)
    ):
        raise HTTPException(status_code=422, detail="이메일 형식을 확인하세요.")

    with db.connect_db() as conn:
        token = create_invitation(conn, user_id, email)

    if token is None:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    response.headers["Cache-Control"] = "no-store"
    return {"invite_token": token}

class RegisterRequest(BaseModel):
    invite_token: str
    password: str


@app.post("/auth/register", status_code=201)
def register(payload: RegisterRequest):
    if not 15 <= len(payload.password) <= 128:
        raise HTTPException(
            status_code=422,
            detail="비밀번호는 15~128자로 입력하세요.",
        )

    password_hash = hash_password(payload.password)

    try:
        with db.connect_db() as conn:
            user_id = register_worker(conn, payload.invite_token, password_hash)
    except UniqueViolation as exc:
        raise HTTPException(
            status_code=409,
            detail="이미 등록된 계정입니다.",
        ) from exc

    if user_id is None:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않거나 이미 사용된 초대입니다.",
        )

    return {"id": user_id}

@app.get("/inference/history")
def read_inference_history(
    response: Response,
    user: tuple[int, str] = Depends(get_current_user),
):
    user_id, role = user
    with db.connect_db() as conn:
        records = db.list_inference_requests(conn, user_id, role)

    response.headers["Cache-Control"] = "no-store"
    return {"records": records}

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

def ready_model_versions(client, model_name, *, ready_only=True):
    versions = {}
    page_token = None

    while True:
        page = client.search_model_versions(
            filter_string=f"name = '{model_name}'",
            max_results=100,
            page_token=page_token,
        )

        for model in page:
            if not ready_only or model.status == "READY":
                versions[model.version] = model

        page_token = page.token
        if not page_token:
            return versions

@app.get("/models/versions")
def list_model_versions():
    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
        registry_uri=MLFLOW_TRACKING_URI,
    )

    def options(model_name):
        versions = ready_model_versions(client, model_name, ready_only=False)
        return [
            {"version": version, "status": versions[version].status}
            for version in sorted(versions, key=int, reverse=True)
        ]

    return {
        "detectors": options("Zero-Defects"),
        "classifiers": options("Zero-Defects-Classifier"),
    }

def registry_model_options():
    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
        registry_uri=MLFLOW_TRACKING_URI,
    )
    detectors = ready_model_versions(client, "Zero-Defects")
    classifiers = ready_model_versions(client, "Zero-Defects-Classifier")
    configured_pairs = {
        (spec["detector_uri"], spec["classifier_uri"])
        for spec in MODEL_SETS.values()
    }

    options = []
    for detector_version in sorted(detectors, key=int, reverse=True):
        tags = detectors[detector_version].tags
        classifier_version = tags.get("classifier_version")
        if tags.get("ui_selectable") != "true" or classifier_version not in classifiers:
            continue

        detector_uri = f"models:/Zero-Defects/{detector_version}"
        classifier_uri = f"models:/Zero-Defects-Classifier/{classifier_version}"
        if (detector_uri, classifier_uri) in configured_pairs:
            continue

        options.append({
            "id": f"registry-d{detector_version}-c{classifier_version}",
            "label": f"Detector v{detector_version} + Classifier v{classifier_version}",
            "model_name": detector_uri,
            "classifier_name": classifier_uri,
            "status": "available",
        })

    return options

@app.get("/models")
def list_models():
    models = [
        {
            "id": key,
            "label": spec["label"],
            "model_name": spec["detector_uri"],
            "classifier_name": spec["classifier_uri"] or None,
            "status": set_status(app, key),
        }
        for key, spec in MODEL_SETS.items()
    ]

    try:
        models.extend(registry_model_options())
    except Exception:
        logger.exception("MLflow 모델 목록 조회 실패")

    return {"models": models}

async def load_registry_pair(selection):
    try:
        selection["detector"] = await run_in_threadpool(
            load_model, MLFLOW_TRACKING_URI, selection["model_name"]
        )
        selection["classifier"] = await run_in_threadpool(
            load_model, MLFLOW_TRACKING_URI, selection["classifier_name"]
        )
        selection["status"] = "ready"
    except Exception:
        selection["status"] = "failed"
        selection["retry_at"] = time.monotonic() + 30
        logger.exception("선택 모델 로드 실패: %s", selection["id"])


@app.post("/models/{model_choice}/prepare")
async def prepare_model(
    model_choice: str,
    user: tuple[int, str] = Depends(get_current_user),
):
    async with app.state.registry_prepare_lock:
        prepared = app.state.registry_models
        current = prepared.get(model_choice)
        if current is not None and (
            current["status"] != "failed" or time.monotonic() < current["retry_at"]
        ):
            return {"status": current["status"]}

        task = app.state.registry_load_task
        if task is not None and not task.done():
            return {"status": "busy"}

        match = re.fullmatch(r"registry-d([1-9][0-9]*)-c([1-9][0-9]*)", model_choice)
        if match is None:
            raise HTTPException(422, "모델 선택값 형식이 올바르지 않습니다.")

        detector_version, classifier_version = match.groups()
        client = MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI,
            registry_uri=MLFLOW_TRACKING_URI,
        )

        try:
            detectors = await run_in_threadpool(
                ready_model_versions, client, "Zero-Defects"
            )
            classifiers = await run_in_threadpool(
                ready_model_versions, client, "Zero-Defects-Classifier"
            )
        except Exception as exc:
            logger.exception("MLflow 모델 버전 조회 실패")
            raise HTTPException(503, "모델 버전을 확인하지 못했습니다.") from exc

        if detector_version not in detectors or classifier_version not in classifiers:
            raise HTTPException(409, "선택한 모델 버전이 없거나 아직 준비되지 않았습니다.")

        option = {
            "id": model_choice,
            "model_name": f"models:/Zero-Defects/{detector_version}",
            "classifier_name": f"models:/Zero-Defects-Classifier/{classifier_version}",
        }

        prepared.pop(model_choice, None)
        if len(prepared) >= 2:
            del prepared[next(iter(prepared))]

        selection = {**option, "status": "loading"}
        prepared[model_choice] = selection
        app.state.registry_load_task = asyncio.create_task(load_registry_pair(selection))
        return {"status": "loading"}

@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_choice: str = Form("champion"),
    user: tuple[int, str] = Depends(get_current_user),
):
    user_id, _ = user
    started_at = time.perf_counter()

    registry_pair = app.state.registry_models.get(model_choice)

    if model_choice not in MODEL_SETS and registry_pair is None:
        if model_choice.startswith("registry-"):
            raise HTTPException(409, "선택한 모델을 먼저 준비해 주세요.")
        raise HTTPException(422, "지원하지 않는 모델 선택값입니다.")

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
            
            width, height = image.size
            if image.size != (184, 597):

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

    status = (
        registry_pair["status"]
        if registry_pair is not None
        else set_status(app, model_choice)
    )

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

    if registry_pair is not None:
        state = registry_pair
        selected_model_name = registry_pair["model_name"]
        classifier_name = registry_pair["classifier_name"]
    else:
        spec = MODEL_SETS[model_choice]
        state = app.state.sets[model_choice]
        selected_model_name = spec["detector_uri"]
        classifier_name = spec["classifier_uri"]

    selected_model = state["detector"]
    classifier = state["classifier"]

    detector_started = time.perf_counter()

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

    detector_ms = round((time.perf_counter() - detector_started) * 1000)

    prediction = (
        "defect"
        if result["is_anomaly"]
        else "normal"
    )

    # 분류가 실패해도 판정 결과는 그대로 반환한다.
    defect = {
        "defect_type": None,
        "defect_subtype": None,
        "confidence": None,
        "regions": [],
    }
    classified = False
    classifier_ms = None

    if result["is_anomaly"] and classifier is not None:
        classifier_started = time.perf_counter()

        try:
            defect = await run_in_threadpool(
                run_model,
                classifier,
                CLASSIFIER_CONTRACT,
                data,
            )
            classified = True
            classifier_ms = round(
                (time.perf_counter() - classifier_started) * 1000
            )
        except Exception:
            logger.exception("Defect classification failed")

    request_id = str(uuid.uuid4())

    # 저장은 응답을 보낸 뒤에 하고, 실패해도 검사 결과에는 영향이 없다.
    if db.ENABLED:
        background_tasks.add_task(
            db.save_inference,
            {
                "id": request_id,
                "user_id": user_id,
                "filename": file.filename or "",
                "model_choice": "challenger" if registry_pair is not None else model_choice,
                "prediction": prediction,
                "total_latency_ms": round(
                    (time.perf_counter() - started_at) * 1000
                ),
                "detector": {
                    "model_name": selected_model_name,
                    "anomaly_score": result["anomaly_score"],
                    "is_defect": result["is_anomaly"],
                    "latency_ms": detector_ms,
                },
                "classifier": {
                    "model_name": classifier_name,
                    "defect_type": defect["defect_type"],
                    "confidence": defect["confidence"],
                    "regions": defect["regions"],
                    "latency_ms": classifier_ms,
                } if classified else None,
            },
        )

    return {
        "id": request_id,
        "filename": file.filename,
        "model_name": selected_model_name,
        "prediction": prediction,
        "anomaly_score": result["anomaly_score"],
        "threshold": result["threshold"],
        "is_anomaly": result["is_anomaly"],
        "defect_type": defect["defect_type"],
        "defect_subtype": defect["defect_subtype"],
        "classifier_name": classifier_name if classified else None,
        "confidence": defect["confidence"],
        "boxes": build_boxes(defect["regions"], width, height),
    }
