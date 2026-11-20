-- Apply to the inference database before deploying the review API.
BEGIN;

ALTER TABLE inference_requests
    ADD COLUMN IF NOT EXISTS review_status text NOT NULL DEFAULT 'not_required'
        CHECK (review_status IN ('not_required', 'pending', 'finalized', 'cancelled')),
    ADD COLUMN IF NOT EXISTS reviewed_defect_type text
        CHECK (reviewed_defect_type IN ('brown', 'black', 'blue', 'surface')),
    ADD COLUMN IF NOT EXISTS reviewed_boxes jsonb NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(reviewed_boxes) = 'array');

ALTER TABLE inference_requests
    DROP CONSTRAINT IF EXISTS inference_final_review_required_fields,
    ADD CONSTRAINT inference_final_review_required_fields CHECK (
        review_status <> 'finalized' OR (
            reviewed_label IS NOT NULL AND reviewed_at IS NOT NULL AND (
                (reviewed_label = 'normal' AND reviewed_defect_type IS NULL
                    AND jsonb_array_length(reviewed_boxes) = 0)
                OR (reviewed_label = 'defect' AND reviewed_defect_type IS NOT NULL)
            )
        )
    );

CREATE TABLE IF NOT EXISTS inference_review_queue (
    request_id uuid PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    payload jsonb,
    cancelled_at timestamptz,
    CHECK (
        (cancelled_at IS NULL AND payload IS NOT NULL AND jsonb_typeof(payload) = 'object')
        OR (cancelled_at IS NOT NULL AND payload IS NULL)
    )
);

COMMIT;

