#!/bin/bash
# db 계층이 빈 데이터 폴더로 처음 뜰 때 한 번만 실행된다 (postgres 공식 이미지의 초기화 단계).
# - POSTGRES_DB(mlflow)는 공식 이미지가 만들고, 여기서는 검사 이력용 inference DB 를 만든다.
# - 스키마는 001 → 002 → 004 → 003 순서로 적용한다. 003(Grafana 읽기 계정)은 앞의 테이블에 권한을 주므로 마지막에 둔다.
# - 003 은 비밀번호를 SQL 파일에 두지 않으므로(운영 방식 그대로) 여기서 환경변수 값으로 정한다.
set -euo pipefail

INFERENCE_DB="${INFERENCE_DB_NAME:-inference}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE \"$INFERENCE_DB\""

for file in 001_inference_schema.sql 002_accounts.sql 004_inference_review.sql 003_grafana_reader.sql; do
    echo "inference DB 에 $file 적용"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$INFERENCE_DB" -f "/schema/$file"
done

# psql 변수(:'pw')는 -c 에서는 치환되지 않아 표준 입력으로 넘긴다 (따옴표 처리를 psql 에 맡겨 비밀번호를 안전하게 넣기 위해)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$INFERENCE_DB" \
    -v pw="${GRAFANA_READER_PASSWORD:?GRAFANA_READER_PASSWORD 가 필요합니다}" <<'SQL'
ALTER ROLE grafana_reader PASSWORD :'pw';
SQL
