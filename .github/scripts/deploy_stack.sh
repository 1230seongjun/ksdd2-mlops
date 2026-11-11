#!/usr/bin/env bash
# 셀프 호스티드 러너에서 스택 하나를 배포한다 (deploy.yaml 이 호출)
#   사용법: bash .github/scripts/deploy_stack.sh <mlops|service>
#   순서: 배포 파일 백업 → 파일 동기화 → 이미지 태그 갱신 → pull → up → 헬스체크 → 실패 시 파일·컨테이너 복원
#         → 성공 시 정리 (우리 이미지는 이번·직전 태그만 남김, 배포 백업은 최근 KEEP_BACKUPS 개만 남김)
#   필요한 환경변수: GITHUB_SHA, NCR_REGISTRY, REGISTRY_PATH, IMAGES(이번에 빌드한 이미지 JSON)
set -euo pipefail

STACK="${1:?mlops 또는 service 를 지정하세요}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/bulryangzero}"
KEEP_BACKUPS="${KEEP_BACKUPS:-5}"   # 남길 배포 백업 폴더 수
SHORT_SHA="${GITHUB_SHA::7}"

case "$STACK" in
  mlops)
    SRC="infra/mlops"; FILE="compose.mlops.yaml"; PROJECT="ksdd2-mlops-infra"
    # MLflow + 실험 대시보드(Streamlit) 둘 다 통과해야 성공
    HEALTH_URLS="http://127.0.0.1:5000/health http://127.0.0.1:8502/_stcore/health"; HEALTH_TRIES=40
    NCR_SERVICES="mlflow experiments-ui"; TAG_VARS="MLFLOW_TAG EXPERIMENTS_UI_TAG" ;;
  service)
    # API 시작 시 MLflow 모델 로딩을 기다리며 /api/ready를 최대 60회 확인
    # 실패한 요청 사이에 3초 대기하며, 총 소요 시간에는 HTTP 요청 시간도 포함됨
    SRC="infra/service"; FILE="compose.service.yaml"; PROJECT="ksdd2-mlops"
    HEALTH_URLS="http://127.0.0.1:8080/api/ready"; HEALTH_TRIES=60
    NCR_SERVICES="api"; TAG_VARS="API_TAG" ;;
  *) echo "알 수 없는 스택: $STACK"; exit 2 ;;
esac
DIR="$DEPLOY_ROOT/$STACK"

if [ ! -f "$DIR/.env" ]; then
  echo "::error::$DIR/.env 가 없습니다. 서버 초기 설정(.env 배치)을 먼저 하세요."
  exit 1
fi

# 동기화 대상 밖에 백업 저장
BACKUP_ROOT="$DEPLOY_ROOT/.deploy-backups/$STACK"
install -d -m 700 "$BACKUP_ROOT"
BACKUP_DIR=$(mktemp -d "$BACKUP_ROOT/deploy.XXXXXXXX")

# .env, Compose, 설정 파일, 웹 파일 백업
mkdir -m 700 "$BACKUP_DIR/files"
cp -a "$DIR/." "$BACKUP_DIR/files/"

echo "배포 전 파일 백업 완료: $BACKUP_DIR"

# 파일 동기화 또는 환경변수 수정 실패 시 파일 복원
restore_files_on_exit() {
  local status=$?
  trap - EXIT

  if [ "$status" -ne 0 ]; then
    echo "::error::배포 준비 실패 → 이전 파일 복원"
    if ! rsync -a --delete "$BACKUP_DIR/files/" "$DIR/"; then
      echo "::error::파일 복원 실패. 백업 위치: $BACKUP_DIR"
    fi
  fi

  exit "$status"
}

trap restore_files_on_exit EXIT

# 1) 배포 파일 동기화 (.env, .env.prev, web/ 은 보존)
rsync -a --delete --exclude ".env" --exclude ".env.prev" --exclude "web/" "$SRC/" "$DIR/"
if [ "$STACK" = "service" ]; then
  rsync -a --delete serving/web/ "$DIR/web/"
fi

cd "$DIR"

# 2) 이미지 태그 갱신
set_env() {
  if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}
set_env NCR_REGISTRY "$NCR_REGISTRY"
set_env REGISTRY_PATH "$REGISTRY_PATH"
for img in $(python3 -c 'import json,os; print(" ".join(json.loads(os.environ.get("IMAGES") or "[]")))'); do
  case "$STACK:$img" in
    mlops:mlflow)         set_env MLFLOW_TAG "$SHORT_SHA" ;;
    mlops:experiments-ui) set_env EXPERIMENTS_UI_TAG "$SHORT_SHA" ;;
    service:api)          set_env API_TAG "$SHORT_SHA" ;;
  esac
done

for v in $TAG_VARS; do
  if ! grep -qE "^$v=.+" .env; then
    echo "::error::$v 가 없습니다. 이미지를 먼저 빌드하고 태그를 설정하세요."
    exit 1
  fi
done

echo "--- $STACK 배포 태그 ---"; grep -E "^($(echo "$TAG_VARS" | tr ' ' '|'))=" .env

# 3) 배포
compose() { docker compose -f "$FILE" -p "$PROJECT" --env-file .env "$@"; }

deploy() {
  compose pull $NCR_SERVICES || return 1          # 우리 이미지만 받음 (postgres·nginx 등은 있으면 그대로 사용)
  # --remove-orphans 는 쓰지 않음: compose 에 없는 컨테이너(현재 수동으로 띄운 ui)를 지우지 않기 위해
  compose up -d || return 1
  if [ "$STACK" = "service" ]; then  # 설정 파일 변경 반영 + 새 API 컨테이너 주소를 Nginx가 다시 찾게 함
    compose exec -T nginx nginx -s reload || true
    compose kill -s HUP prometheus || true
  fi
  compose ps
}

health() {
  for url in $HEALTH_URLS; do
    ok=false
    for i in $(seq 1 "$HEALTH_TRIES"); do
      # 연결 대기는 최대 2초, 요청 전체는 최대 5초로 제한
      if curl -fsS --connect-timeout 2 --max-time 5 "$url"; then echo; echo "헬스체크 통과: $url"; ok=true; break; fi
      echo "대기 중... $url ($i/$HEALTH_TRIES)"; sleep 3
    done
    if [ "$ok" != true ]; then echo "헬스체크 실패: $url"; return 1; fi
  done
}

# 4) 정리 (배포 성공 후에만 실행)
#   - 우리 이미지(NCR_SERVICES)는 이번 태그와 직전 태그만 남긴다. 직전 태그는 실패 시 복원(--pull never)에 필요
#   - 사용 중인 이미지는 docker 가 지우지 않으므로 그대로 둔다
#   - 태그 없는 이미지(dangling)와 오래된 배포 백업 폴더도 정리한다
tag_of() { grep -E "^$1=" "$2" 2>/dev/null | tail -n 1 | cut -d= -f2- || true; }

cleanup() {
  local name var keep repo ref tag
  for name in $NCR_SERVICES; do
    case "$name" in
      mlflow)         var=MLFLOW_TAG ;;
      experiments-ui) var=EXPERIMENTS_UI_TAG ;;
      api)            var=API_TAG ;;
      *)              continue ;;
    esac
    keep="$(tag_of "$var" .env) $(tag_of "$var" "$BACKUP_DIR/files/.env") $(tag_of "$var" .env.prev)"
    repo="$NCR_REGISTRY/$REGISTRY_PATH/$name"
    echo "--- $name 이미지 정리 (남김: $(echo $keep | tr ' ' '\n' | sort -u | tr '\n' ' ')) ---"
    docker images --format '{{.Repository}}:{{.Tag}}' "$repo" | while read -r ref; do
      tag="${ref##*:}"
      case " $keep " in *" $tag "*) continue ;; esac
      if docker rmi "$ref" >/dev/null 2>&1; then echo "삭제: $ref"; else echo "사용 중이라 남김: $ref"; fi
    done
  done
  docker image prune -f >/dev/null || true
  ls -1dt "$BACKUP_ROOT"/deploy.* 2>/dev/null | tail -n +$((KEEP_BACKUPS + 1)) | xargs -r rm -rf
  echo "배포 백업은 최근 $KEEP_BACKUPS 개만 남김: $BACKUP_ROOT"
}

# 이제부터는 아래에서 파일과 컨테이너를 함께 복구한다.
trap - EXIT

if deploy && health; then
  echo "배포 성공: $STACK"
  cleanup || echo "::warning::이미지·백업 정리 중 오류가 있었지만 배포는 성공했습니다"
  exit 0
fi

echo "::error::$STACK 배포 실패 → 이전 배포 복원"
compose logs --tail 50 || true

# 먼저 이전 Compose, 환경변수, 설정, 웹 파일 복원
if ! rsync -a --delete "$BACKUP_DIR/files/" "$DIR/"; then
  echo "::error::파일 복원 실패. 백업 위치: $BACKUP_DIR"
  exit 1
fi

# 복원한 설정과 로컬의 이전 이미지로 컨테이너 재생성
if ! compose up -d --force-recreate --pull never; then
  echo "::error::컨테이너 복원 실패. 수동 복구가 필요합니다."
  exit 1
fi

if health; then
  echo "::warning::이전 배포 복원 후 헬스체크 통과"
else
  echo "::error::복원 후에도 헬스체크 실패"
  compose logs --tail 50 || true
fi

# 복구 성공 여부와 별개로 이번 배포는 실패 처리
exit 1
