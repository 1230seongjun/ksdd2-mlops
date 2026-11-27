#!/bin/bash
# postgres 컨테이너가 빈 데이터 폴더로 처음 뜰 때 한 번만 실행된다 (compose.mlops.yaml 이 연결함)
# 검사 이력용 inference DB 를 만들고 스키마를 순서대로 적용한다.
# 003_grafana_reader.sql 은 계정 비밀번호를 따로 정해야 해서 여기서는 적용하지 않는다.
set -euo pipefail

INFERENCE_DB="${INFERENCE_DB_NAME:-inference}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE \"$INFERENCE_DB\""

for file in 001_inference_schema.sql 002_accounts.sql 004_inference_review.sql; do
    echo "inference DB 에 $file 적용"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$INFERENCE_DB" -f "/schema/$file"
done
