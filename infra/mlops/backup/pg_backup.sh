#!/usr/bin/env bash
# PostgreSQL 일일 백업 (ADR-BZ-24)
# 설치: cp 이 파일 ~/bin/ && chmod 700 ~/bin/pg_backup.sh
# cron (mlops 서버, KST): 0 3 * * * $HOME/bin/pg_backup.sh >> $HOME/backups/postgres/backup.log 2>&1
# globals(비밀번호 제외) + DB별 pg_dump -Fc → 덤프 검증 → Object Storage backups 업로드 → 크기 확인 → 보관기간 정리
set -euo pipefail
umask 077

PG=ksdd2-mlops-infra-postgres-1
MLF=ksdd2-mlops-infra-mlflow-1
DB_USER=mlflow
DBS=(mlflow inference)
BUCKET=backups
ROOT="$HOME/backups/postgres"
LOCAL_KEEP_DAYS=7
REMOTE_KEEP_DAYS=30

log(){ echo "[$(date '+%F %T')] $*"; }
trap 'log "백업 실패 (line $LINENO)"' ERR

mkdir -p "$ROOT"
exec 9>"$ROOT/.lock"
flock -n 9 || { log "이미 실행 중이라 종료"; exit 1; }

N=$(date +%Y%m%d_%H%M)
B="$ROOT/$N"
mkdir -p "$B"
log "백업 시작: $N"

# 1) 역할·권한 (비밀번호 제외)
docker exec "$PG" pg_dumpall -U "$DB_USER" --globals-only --no-role-passwords > "$B/globals.sql"
[ -s "$B/globals.sql" ] || { log "globals.sql 비어 있음"; exit 1; }

# 2) DB별 덤프 + 목차 검증
FILES=(globals.sql)
for db in "${DBS[@]}"; do
  docker exec "$PG" pg_dump -U "$DB_USER" -Fc -d "$db" > "$B/$db.dump"
  n=$(docker exec -i "$PG" pg_restore -l < "$B/$db.dump" | grep -c 'TABLE DATA' || true)
  [ "$n" -gt 0 ] || { log "$db 덤프 검증 실패"; exit 1; }
  log "$db: 테이블 $n개, $(du -h "$B/$db.dump" | cut -f1)"
  FILES+=("$db.dump")
done
(cd "$B" && sha256sum "${FILES[@]}" > SHA256SUMS)
FILES+=(SHA256SUMS)

# 3) 업로드 (MLflow 컨테이너의 S3 접속 정보 사용, 표준 입력으로 전달)
PY_UP='
import boto3, os, sys
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
s3.upload_fileobj(sys.stdin.buffer, sys.argv[1], sys.argv[2])
'
for f in "${FILES[@]}"; do
  docker exec -i "$MLF" python -c "$PY_UP" "$BUCKET" "postgres/$N/$f" < "$B/$f"
done

# 4) 업로드 크기 확인
PY_LS='
import boto3, os, sys
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
for o in s3.list_objects_v2(Bucket=sys.argv[1], Prefix=sys.argv[2]).get("Contents", []):
    print(o["Size"], o["Key"])
'
remote=$(docker exec "$MLF" python -c "$PY_LS" "$BUCKET" "postgres/$N/")
for f in "${FILES[@]}"; do
  l=$(stat -c %s "$B/$f")
  r=$(echo "$remote" | awk -v k="postgres/$N/$f" '$2==k{print $1}')
  [ "$l" = "${r:-}" ] || { log "업로드 크기 불일치: $f (로컬 $l / 버킷 ${r:-없음})"; exit 1; }
done
log "업로드 확인 완료: $BUCKET/postgres/$N/ (${#FILES[@]}개)"

# 5) 보관기간 정리: 버킷 30일, 로컬 7일 (postgres/ 경로만 대상)
PY_CLEAN='
import boto3, os, sys
from datetime import datetime, timedelta, timezone
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
cut = datetime.now(timezone.utc) - timedelta(days=int(sys.argv[2]))
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=sys.argv[1], Prefix="postgres/"):
    for o in page.get("Contents", []):
        if o["LastModified"] < cut:
            s3.delete_object(Bucket=sys.argv[1], Key=o["Key"]); n += 1
print(n)
'
d=$(docker exec "$MLF" python -c "$PY_CLEAN" "$BUCKET" "$REMOTE_KEEP_DAYS")
find "$ROOT" -mindepth 1 -maxdepth 1 -type d -name '20*' -mtime +"$LOCAL_KEEP_DAYS" -exec rm -rf {} +
log "정리: 버킷에서 ${d}개 삭제(${REMOTE_KEEP_DAYS}일 초과), 로컬 ${LOCAL_KEEP_DAYS}일 초과 폴더 삭제"
log "백업 완료"
