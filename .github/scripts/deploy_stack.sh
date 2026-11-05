#!/usr/bin/env bash
# 셀프 호스티드 러너에서 스택 하나를 배포한다 (deploy.yaml 이 호출)
#   사용법: bash .github/scripts/deploy_stack.sh <mlops|service>
#   필요한 환경변수: GITHUB_SHA, NCR_REGISTRY, REGISTRY_PATH, IMAGES(이번에 빌드한 이미지 JSON)
# 순서: 파일 동기화 → .env 태그 갱신(.env.prev 백업) → 우리 이미지 pull → up → 헬스체크 → 실패하면 롤백
set -euo pipefail

STACK="${1:?mlops 또는 service 를 지정하세요}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/bulryangzero}"
SHORT_SHA="${GITHUB_SHA::7}"

case "$STACK" in
  mlops)
    SRC="infra/mlops"; FILE="compose.mlops.yaml"; PROJECT="ksdd2-mlops-infra"
    HEALTH_URL="http://127.0.0.1:5000/health"; HEALTH_TRIES=40
    NCR_SERVICES="mlflow"; TAG_VARS="MLFLOW_TAG" ;;
  service)
    # API는 시작할 때 MLflow에서 모델을 받아 올리느라 약 2분 걸림 → 최대 3분(3초 × 60) 대기
    SRC="infra/service"; FILE="compose.service.yaml"; PROJECT="ksdd2-mlops"
    HEALTH_URL="http://127.0.0.1:8080/api/health"; HEALTH_TRIES=60
    NCR_SERVICES="api"; TAG_VARS="API_TAG" ;;
  *) echo "알 수 없는 스택: $STACK"; exit 2 ;;
esac
DIR="$DEPLOY_ROOT/$STACK"
WORKSPACE="$PWD"

if [ ! -f "$DIR/.env" ]; then
  echo "::error::$DIR/.env 가 없습니다. 서버 초기 설정(.env 배치)을 먼저 하세요."
  exit 1
fi

# 1) 배포 파일 동기화 (.env, .env.prev, web/ 은 보존)
rsync -a --delete --exclude ".env" --exclude ".env.prev" --exclude "web/" "$SRC/" "$DIR/"
if [ "$STACK" = "service" ]; then
  rsync -a --delete serving/web/ "$DIR/web/"
fi

cd "$DIR"

# 2) 이미지 태그 갱신
cp .env .env.prev
set_env() {
  if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}
set_env NCR_REGISTRY "$NCR_REGISTRY"
set_env REGISTRY_PATH "$REGISTRY_PATH"
for img in $(python3 -c 'import json,os; print(" ".join(json.loads(os.environ.get("IMAGES") or "[]")))'); do
  case "$STACK:$img" in
    mlops:mlflow) set_env MLFLOW_TAG "$SHORT_SHA" ;;
    service:api)  set_env API_TAG "$SHORT_SHA" ;;
  esac
done

for v in $TAG_VARS; do
  if ! grep -qE "^$v=.+" .env; then
    echo "::warning::$v 가 아직 없어 $STACK 배포를 건너뜁니다. Actions → Build and deploy → Run workflow 로 이미지를 먼저 빌드하세요."
    exit 0
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
  for i in $(seq 1 "$HEALTH_TRIES"); do
    if curl -fsS "$HEALTH_URL"; then echo; echo "헬스체크 통과: $HEALTH_URL"; return 0; fi
    echo "대기 중... ($i/$HEALTH_TRIES)"; sleep 3
  done
  echo "헬스체크 실패: $HEALTH_URL"; return 1
}

if deploy && health; then
  exit 0
fi

# 4) 롤백
echo "::error::$STACK 배포 실패 → 이전 태그로 롤백합니다."
compose logs --tail 50 || true
cp .env.prev .env
compose up -d || true
cd "$WORKSPACE"
exit 1
