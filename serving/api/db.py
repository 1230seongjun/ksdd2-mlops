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

def insert_inference(conn, record, created_at=None):
    detector = record["detector"]
    classifier = record["classifier"]
    conn.execute(
        """
        INSERT INTO inference_requests
            (id, user_id, filename, model_choice, prediction, total_latency_ms, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, CURRENT_TIMESTAMP))
        """,
        (
            record["id"],
            record["user_id"],
            record["filename"],
            record["model_choice"],
            record["prediction"],
            record["total_latency_ms"],
            created_at,
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
            "confidence, regions, latency_ms) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                record["id"],
                classifier["model_name"],
                model_version(classifier["model_name"]),
                classifier["defect_type"],
                classifier["confidence"],
                Jsonb(classifier["regions"]),
                classifier["latency_ms"],
            ),
        )


def save_inference(record):
    try:
        with connect_db() as conn:
            insert_inference(conn, record)
        save_total.labels("ok").inc()
    except Exception:
        save_total.labels("error").inc()
        logger.exception("Inference save failed: %s", record["id"])


def stage_inference_review(record):
    try:
        with connect_db() as conn:
            conn.execute(
                "INSERT INTO inference_review_queue (request_id, user_id, payload) VALUES (%s, %s, %s)",
                (record["id"], record["user_id"], Jsonb(record)),
            )
        return True
    except Exception:
        logger.exception("Review staging failed: %s", record["id"])
        return False


def finalize_inference_review(request_id, user_id, label, defect_type, boxes):
    with connect_db() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT created_at, payload, cancelled_at
              FROM inference_review_queue
             WHERE request_id = %s AND user_id = %s
               FOR UPDATE
            """,
            (request_id, user_id),
        )
        pending = cur.fetchone()
        if pending is None:
            # A retry can arrive after the final row has already been committed.
            cur.execute(
                """
                SELECT review_status, reviewed_label, reviewed_at, reviewed_defect_type, reviewed_boxes
                  FROM inference_requests
                 WHERE id = %s AND user_id = %s
                """,
                (request_id, user_id),
            )
            return cur.fetchone()

        if pending["cancelled_at"] is not None or label == "cancelled":
            cancelled_at = pending["cancelled_at"]
            if cancelled_at is None:
                cur.execute(
                    """
                    UPDATE inference_review_queue
                       SET cancelled_at = CURRENT_TIMESTAMP, payload = NULL
                     WHERE request_id = %s AND user_id = %s
                     RETURNING cancelled_at
                    """,
                    (request_id, user_id),
                )
                cancelled_at = cur.fetchone()["cancelled_at"]
            return {
                "review_status": "cancelled", "reviewed_label": None, "reviewed_at": cancelled_at,
                "reviewed_defect_type": None, "reviewed_boxes": [],
            }

        insert_inference(conn, pending["payload"], pending["created_at"])
        cur.execute(
            """
            UPDATE inference_requests
               SET review_status = 'finalized', reviewed_label = %s,
                   reviewed_at = CURRENT_TIMESTAMP, reviewed_defect_type = %s, reviewed_boxes = %s
             WHERE id = %s AND user_id = %s
             RETURNING review_status, reviewed_label, reviewed_at, reviewed_defect_type, reviewed_boxes
            """,
            (label, defect_type, Jsonb(boxes), request_id, user_id),
        )
        result = cur.fetchone()
        cur.execute(
            "DELETE FROM inference_review_queue WHERE request_id = %s AND user_id = %s",
            (request_id, user_id),
        )
    save_total.labels("ok").inc()
    return result


def list_history_workers(conn):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id AS worker_id, split_part(email, '@', 1) AS worker_account
            FROM users
            WHERE role = 'worker'
            ORDER BY lower(split_part(email, '@', 1)), id
            """
        )
        return cur.fetchall()

def list_inference_requests(conn, user_id, role, *, filename=None, prediction=None,
                            start=None, end=None, offset=0, limit=50, worker_id=None):
    conditions = """
            WHERE r.review_status IN ('not_required', 'finalized')
              AND (%(role)s = 'admin' OR r.user_id = %(user_id)s)
              AND (%(worker_id)s::bigint IS NULL OR r.user_id = %(worker_id)s::bigint)
              AND (%(filename)s::text IS NULL OR
                   strpos(lower(r.filename), lower(%(filename)s::text)) > 0)
              AND (%(prediction)s::text IS NULL OR COALESCE(r.reviewed_label, r.prediction) = %(prediction)s)
              AND (%(start)s::timestamptz IS NULL OR r.created_at >= %(start)s)
              AND (%(end)s::timestamptz IS NULL OR r.created_at < %(end)s)
    """
    params = dict(
        role=role, user_id=user_id, worker_id=worker_id, filename=filename,
        prediction=prediction, start=start, end=end, offset=offset, limit=limit,
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                r.id,
                r.filename,
                r.model_choice,
                COALESCE(r.reviewed_label, r.prediction) AS prediction,
                r.prediction AS model_prediction,
                r.created_at,
                r.user_id AS worker_id,
                split_part(u.email, '@', 1) AS worker_account,
                r.total_latency_ms,
                r.reviewed_label,
                r.reviewed_at,
                r.review_status,
                r.reviewed_defect_type,
                r.reviewed_boxes,
                d.model_name AS detector_name,
                d.model_version AS detector_version,
                d.anomaly_score,
                d.is_defect,
                d.latency_ms AS detector_latency_ms,
                c.model_name AS classifier_name,
                c.model_version AS classifier_version,
                CASE WHEN r.reviewed_label = 'normal' THEN NULL
                     ELSE COALESCE(r.reviewed_defect_type, c.defect_type)
                END AS defect_type,
                c.defect_type AS model_defect_type,
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


