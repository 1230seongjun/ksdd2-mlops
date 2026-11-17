
-- Grafana 전용 PostgreSQL 읽기 계정
-- 대상 DB: inference
-- 비밀번호는 SQL 파일에 저장하지 않고 별도로 설정한다.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = 'grafana_reader'
    ) THEN
        CREATE ROLE grafana_reader
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOREPLICATION;
    END IF;
END
$$;

-- inference DB 접속 권한
GRANT CONNECT ON DATABASE inference TO grafana_reader;

-- public 스키마 접근 권한
GRANT USAGE ON SCHEMA public TO grafana_reader;

-- 기존에 부여한 전체 테이블 조회 권한 제거
REVOKE SELECT ON ALL TABLES IN SCHEMA public
FROM grafana_reader;

-- 추론 관련 테이블 3개만 조회 허용
GRANT SELECT ON TABLE
    public.inference_requests,
    public.detector_results,
    public.classifier_results
TO grafana_reader;

-- 향후 생성되는 테이블에 대한 자동 조회 권한 제거
ALTER DEFAULT PRIVILEGES FOR ROLE mlflow IN SCHEMA public
    REVOKE SELECT ON TABLES FROM grafana_reader;
