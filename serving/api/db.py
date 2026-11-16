import json
import logging
import os
import re

import psycopg
from prometheus_client import Counter
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

DB_HOST = os.getenv("INFERENCE_DB_HOST", "").strip()
DB_PORT = os.getenv("INFERENCE_DB_PORT", "").strip() or "5432"
DB_NAME = os.getenv("INFERENCE_DB_NAME", "").strip()
DB_USER = os.getenv("INFERENCE_DB_USER", "").strip()
DB_PASSWORD = os.getenv("INFERENCE_DB_PASSWORD", "")

# 접속 정보가 없으면 저장하지 않는다. DB가 준비되기 전에도 배포할 수 있다.
ENABLED = bool(DB_HOST and DB_NAME and DB_USER)

save_total = Counter(
    "defectcatch_inference_save_total",
    "Number of inference result saves",
    ["result"],
)


def model_version(model_uri):
    match = re.fullmatch(r"models:/[^/]+/(\d+)", model_uri)

    if match is None:
        raise ValueError(f"숫자 버전이 없는 모델 URI입니다: {model_uri}")

    return int(match.group(1))


def save_inference(record):
    detector = record["detector"]
    classifier = record["classifier"]

    try:
        with psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=3,
        ) as conn:
            conn.execute(
                "INSERT INTO inference_requests "
                "(id, filename, model_choice, prediction, total_latency_ms) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    record["id"],
                    record["filename"],
                    record["model_choice"],
                    record["prediction"],
                    record["total_latency_ms"],
                ),
            )
            conn.execute(
                "INSERT INTO detector_results "
                "(request_id, model_name, model_version, anomaly_score, "
                "is_defect, latency_ms) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    record["id"],
                    detector["model_name"],
                    model_version(detector["model_name"]),
                    detector["anomaly_score"],
                    detector["is_defect"],
                    detector["latency_ms"],
                ),
            )

            if classifier is not None:
                conn.execute(
                    "INSERT INTO classifier_results "
                    "(request_id, model_name, model_version, defect_type, "
                    "confidence, regions, latency_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        record["id"],
                        classifier["model_name"],
                        model_version(classifier["model_name"]),
                        classifier["defect_type"],
                        classifier["confidence"],
                        Jsonb(
                            classifier["regions"],
                            dumps=lambda obj: json.dumps(
                                obj, ensure_ascii=False
                            ),
                        ),
                        classifier["latency_ms"],
                    ),
                )

        save_total.labels("ok").inc()

    except Exception:
        save_total.labels("error").inc()
        logger.exception("Inference save failed: %s", record["id"])
