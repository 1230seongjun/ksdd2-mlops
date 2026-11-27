"""검사 이미지를 파일로 남기는 저장소.

추론할 때마다 이미지가 쌓이면 클라우드 저장 비용이 계속 늘어나서, 운영(NHN Cloud)에서는 끈 채로 두었습니다.
환경변수로 켜면 검사한 원본 이미지를 요청 ID 이름으로 저장합니다.

  <INFERENCE_IMAGE_DIR>/requests/<요청 ID>.png|.jpg   (받은 바이트 그대로)

판정 · 작업자 재분류 · 결함 유형 · 박스 좌표는 이미 DB(inference_requests 등)에 저장되므로 여기서는 이미지만 저장합니다.
파일 이름의 요청 ID 가 inference_requests.id 와 같아서, 재학습 데이터를 만들 때 DB 행과 바로 이어 붙일 수 있습니다.

환경변수
  INFERENCE_IMAGE_STORE  none(기본, 저장 안 함) | local
  INFERENCE_IMAGE_DIR    local 일 때 저장 폴더 (기본 /data/inference)

저장에 실패해도 검사 응답과 DB 기록에는 영향을 주지 않습니다 (로그와 지표로만 남김).
"""
import logging
import os
from pathlib import Path
from uuid import UUID

from prometheus_client import Counter

logger = logging.getLogger(__name__)

BACKEND = os.getenv("INFERENCE_IMAGE_STORE", "none").strip().lower() or "none"
ROOT = Path(os.getenv("INFERENCE_IMAGE_DIR", "/data/inference").strip() or "/data/inference")
ENABLED = BACKEND == "local"

if BACKEND not in ("none", "local"):
    logger.warning("알 수 없는 INFERENCE_IMAGE_STORE=%s → 이미지를 저장하지 않습니다", BACKEND)

store_total = Counter(
    "defectcatch_image_store_total",
    "Number of inference image writes to the image store",
    ["result"],
)

EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg"}


def save_image(request_id: str, data: bytes, content_type: str) -> None:
    """검사 직후(응답 뒤 백그라운드) 호출: 원본 이미지를 <요청 ID>.<확장자> 로 저장한다."""
    if not ENABLED:
        return
    try:
        # 요청 ID 는 UUID 만 허용해서 경로 조작을 막는다
        name = str(UUID(str(request_id))) + EXTENSIONS.get(content_type, ".png")
        folder = ROOT / "requests"
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f".{name}.tmp"
        temporary.write_bytes(data)
        temporary.replace(folder / name)
        store_total.labels("ok").inc()
    except Exception:
        store_total.labels("error").inc()
        logger.exception("Inference image save failed: %s", request_id)
