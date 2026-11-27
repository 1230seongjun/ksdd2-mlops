#!/bin/bash
# postgres 가 빈 데이터 폴더로 처음 뜰 때 한 번만 실행됩니다 (운영 DB 에는 쓰지 않음).
# mlflow DB 는 POSTGRES_DB 로 이미 만들어지고, 여기서는 검사 이력용 inference DB 를 만듭니다.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c 'CREATE DATABASE inference'

# 003(Grafana 읽기 계정)은 테이블이 있어야 권한을 줄 수 있어 마지막에 적용
for file in 001_inference_schema.sql 002_accounts.sql 004_inference_review.sql 003_grafana_reader.sql; do
    echo "inference DB 에 $file 적용"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname inference -f "/schema/$file"
done

# 003 은 비밀번호를 파일에 두지 않으므로 여기서 설정
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname inference \
    -v pw="$GRAFANA_READER_PASSWORD" <<'SQL'
ALTER ROLE grafana_reader PASSWORD :'pw';
SQL
