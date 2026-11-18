import json
import logging
import os
import re

import psycopg
from prometheus_client import Counter
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row

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

def connect_db():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=3,
    )

def save_inference(record):
    detector = record["detector"]
    classifier = record["classifier"]

    try:
        with connect_db() as conn:
            conn.execute(
                "INSERT INTO inference_requests "
                "(id, user_id, filename, model_choice, prediction, total_latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    record["id"],
                    record["user_id"],
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
    
def list_inference_requests(conn, user_id, role, *, filename=None, prediction=None,
                            start=None, end=None, offset=0, limit=50):
    conditions = """
            WHERE (%(role)s = 'admin' OR r.user_id = %(user_id)s)
              AND (%(filename)s::text IS NULL OR
                   strpos(lower(r.filename), lower(%(filename)s::text)) > 0)
              AND (%(prediction)s::text IS NULL OR r.prediction = %(prediction)s)
              AND (%(start)s::timestamptz IS NULL OR r.created_at >= %(start)s)
              AND (%(end)s::timestamptz IS NULL OR r.created_at < %(end)s)
    """
    params = dict(role=role, user_id=user_id, filename=filename,
                  prediction=prediction, start=start, end=end, offset=offset, limit=limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                r.id,
                r.filename,
                r.model_choice,
                r.prediction,
                r.created_at,
                r.user_id AS worker_id,
                split_part(u.email, '@', 1) AS worker_account,
                r.total_latency_ms,
                r.reviewed_label,
                r.reviewed_at,
                d.model_name AS detector_name,
                d.model_version AS detector_version,
                d.anomaly_score,
                d.is_defect,
                d.latency_ms AS detector_latency_ms,
                c.model_name AS classifier_name,
                c.model_version AS classifier_version,
                c.defect_type,
                c.confidence,
                c.regions,
                c.latency_ms AS classifier_latency_ms
            FROM inference_requests AS r
            LEFT JOIN users AS u ON u.id = r.user_id
            LEFT JOIN detector_results AS d ON d.request_id = r.id
            LEFT JOIN classifier_results AS c ON c.request_id = r.id
            """ + conditions + """
            ORDER BY r.created_at DESC, r.id DESC
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            params,
        )
        records = cur.fetchall()
        cur.execute("SELECT count(*) AS total FROM inference_requests AS r " + conditions, params)
        return records, cur.fetchone()["total"]

