"""검사한 원본 이미지를 파일로 저장한다 (재검토 결과와 함께 재학습 데이터로 쓰기 위해).

운영(NHN Cloud)에서는 검사할 때마다 이미지가 Object Storage 에 쌓여 비용이 계속 늘어나서 저장하지 않았다.
로컬 실행에서만 켜서, 작업자가 확정한 이미지를 재학습에 쓴다 (deploy/app/build_dataset.py).

  <INFERENCE_IMAGE_DIR>/requests/<요청 ID>.png|.jpg   (받은 바이트 그대로)

판정 · 작업자 재분류 결과는 이미 검사 이력 DB(inference_requests)에 저장되므로 여기서는 이미지만 저장한다.
파일 이름의 요청 ID 가 inference_requests.id 와 같아서 둘을 이어 붙일 수 있다.

환경변수
  INFERENCE_IMAGE_STORE  none(기본, 저장 안 함) | local
  INFERENCE_IMAGE_DIR    local 일 때 저장 폴더 (기본 /data/inference)

저장에 실패해도 검사 응답과 DB 기록에는 영향을 주지 않는다 (로그와 지표로만 남김).
"""
import logging
import os
from pathlib import Path
from uuid import UUID

from prometheus_client import Counter

logger = logging.getLogger(__name__)

ENABLED = os.getenv("INFERENCE_IMAGE_STORE", "none").strip().lower() == "local"
ROOT = Path(os.getenv("INFERENCE_IMAGE_DIR", "/data/inference").strip() or "/data/inference")
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg"}

store_total = Counter(
    "defectcatch_image_store_total",
    "Number of inference image writes to the image store",
    ["result"],
)


def save_image(request_id: str, data: bytes, content_type: str) -> None:
    """검사 응답을 보낸 뒤(백그라운드) 원본 이미지를 <요청 ID>.<확장자> 로 저장한다."""
    if not ENABLED:
        return
    try:
        name = str(UUID(str(request_id))) + EXTENSIONS.get(content_type, ".png")   # UUID 만 허용해 경로 조작을 막는다
        folder = ROOT / "requests"
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f".{name}.tmp"
        temporary.write_bytes(data)
        temporary.replace(folder / name)   # 학습 데이터를 만들 때 쓰다 만 파일을 읽지 않게
        store_total.labels("ok").inc()
    except Exception:
        store_total.labels("error").inc()
        logger.exception("Inference image save failed: %s", request_id)
