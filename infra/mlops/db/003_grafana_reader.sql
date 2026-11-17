
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

-- 기존 테이블 조회 권한
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;

-- 향후 mlflow 계정이 만드는 테이블에도 조회 권한 부여
ALTER DEFAULT PRIVILEGES FOR ROLE mlflow IN SCHEMA public
    GRANT SELECT ON TABLES TO grafana_reader;

