
-- ZeroDefect Inference Database Schema
-- Version: 2.0
-- Updated: 2026-11-16
-- Database: inference
-- Purpose: Store inference requests, detector results,
--          and classifier results.
-- Note: Run only when creating the schema in an empty database.

-- 1. Inference requests
CREATE TABLE inference_requests (
    id               uuid PRIMARY KEY,
    created_at       timestamptz NOT NULL DEFAULT now(),
    filename         text NOT NULL,
    model_choice     text NOT NULL CHECK (model_choice IN ('champion', 'challenger')),
    prediction       text NOT NULL CHECK (prediction IN ('normal', 'defect')),
    total_latency_ms integer,
    reviewed_label   text,
    reviewed_at      timestamptz
);

-- 2. Detector results
CREATE TABLE detector_results (
    request_id    uuid PRIMARY KEY REFERENCES inference_requests(id) ON DELETE CASCADE,
    model_name    text NOT NULL,
    model_version integer NOT NULL,
    anomaly_score double precision NOT NULL,
    is_defect     boolean NOT NULL,
    latency_ms    integer
);

-- 3. Classifier results
CREATE TABLE classifier_results (
    request_id    uuid PRIMARY KEY REFERENCES inference_requests(id) ON DELETE CASCADE,
    model_name    text NOT NULL,
    model_version integer NOT NULL,
    defect_type   text CHECK (defect_type IN ('brown', 'black', 'blue', 'surface')),
    confidence    double precision CHECK (confidence BETWEEN 0 AND 1),
    regions       jsonb NOT NULL DEFAULT '[]'::jsonb,
    latency_ms    integer
);

-- 4. Indexes
CREATE INDEX idx_inference_created_at
    ON inference_requests (created_at);

CREATE INDEX idx_inference_prediction
    ON inference_requests (prediction);

CREATE INDEX idx_detector_model_version
    ON detector_results (model_version);

CREATE INDEX idx_classifier_defect_type
    ON classifier_results (defect_type);

CREATE INDEX idx_classifier_model_version
    ON classifier_results (model_version);

