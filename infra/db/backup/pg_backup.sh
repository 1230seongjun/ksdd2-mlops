#!/usr/bin/env bash
# PostgreSQL 백업 (DB 티어). 로컬 폴더에만 저장한다
# 실행: scripts/local.sh backup   (또는 직접 실행. cron 예: 0 3 * * * <레포>/infra/db/backup/pg_backup.sh >> $HOME/backups/postgres/backup.log 2>&1)
# globals(비밀번호 제외) + DB별 pg_dump -Fc → 덤프 검증 → 체크섬 → 보관기간 정리
set -euo pipefail
umask 077

ENGINE="${ENGINE:-$(command -v docker >/dev/null 2>&1 && echo docker || echo podman)}"
PG="${PG_CONTAINER:-ksdd2-db-postgres-1}"
DB_USER="${DB_USER:-mlflow}"
DBS=(mlflow inference)
ROOT="${BACKUP_DIR:-$HOME/backups/postgres}"
KEEP_DAYS="${KEEP_DAYS:-7}"

log(){ echo "[$(date '+%F %T')] $*"; }
trap 'log "백업 실패 (line $LINENO)"' ERR

mkdir -p "$ROOT"
exec 9>"$ROOT/.lock"
flock -n 9 || { log "이미 실행 중이라 종료"; exit 1; }

N=$(date +%Y%m%d_%H%M)
B="$ROOT/$N"
mkdir -p "$B"
log "백업 시작: $N ($ENGINE, $PG)"

# 1) 역할·권한 (비밀번호 제외)
"$ENGINE" exec "$PG" pg_dumpall -U "$DB_USER" --globals-only --no-role-passwords > "$B/globals.sql"
[ -s "$B/globals.sql" ] || { log "globals.sql 비어 있음"; exit 1; }

# 2) DB별 덤프 + 목차 검증
FILES=(globals.sql)
for db in "${DBS[@]}"; do
  "$ENGINE" exec "$PG" pg_dump -U "$DB_USER" -Fc -d "$db" > "$B/$db.dump"
  n=$("$ENGINE" exec -i "$PG" pg_restore -l < "$B/$db.dump" | grep -c 'TABLE DATA' || true)
  [ "$n" -gt 0 ] || { log "$db 덤프 검증 실패"; exit 1; }
  log "$db: 테이블 $n개, $(du -h "$B/$db.dump" | cut -f1)"
  FILES+=("$db.dump")
done
(cd "$B" && sha256sum "${FILES[@]}" > SHA256SUMS)

# 3) 보관기간 정리
find "$ROOT" -mindepth 1 -maxdepth 1 -type d -name '20*' -mtime +"$KEEP_DAYS" -exec rm -rf {} +
log "백업 완료: $B (${KEEP_DAYS}일 지난 백업은 삭제)"
