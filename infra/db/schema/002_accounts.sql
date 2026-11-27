BEGIN;

CREATE TABLE users (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email         text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    role          text NOT NULL CHECK (role IN ('admin', 'worker')),
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE invitations (
    token_hash text PRIMARY KEY,
    email      text NOT NULL,
    created_by bigint NOT NULL REFERENCES users(id),
    expires_at timestamptz NOT NULL,
    used_at    timestamptz,
    revoked_at timestamptz
);

CREATE TABLE sessions (
    token_hash text PRIMARY KEY,
    user_id    bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL
);

ALTER TABLE inference_requests
    ADD COLUMN user_id bigint REFERENCES users(id);

CREATE INDEX idx_inference_requests_user_created
    ON inference_requests (user_id, created_at DESC);

COMMIT;
