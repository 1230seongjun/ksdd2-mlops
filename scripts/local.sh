#!/usr/bin/env bash
# 한 대에서 3티어(DB / 애플리케이션 / 웹)로 전체 스택을 띄운다. Podman(podman-compose), Docker 둘 다 지원.
#
#   DB 티어   infra/db    PostgreSQL                                  (외부 포트 없음)
#   앱 티어   infra/app   추론 API, 학습 서버, MLflow, Prometheus, Grafana
#   웹 티어   infra/web   nginx: 검사 화면, /api·/grafana 전달        (https://localhost:8443)
#   네트워크  웹 ↔ 앱: ksdd2-web-app   /   앱 ↔ DB: ksdd2-app-db   (웹은 DB 에 닿지 않음)
#
# 처음 한 번
#   scripts/local.sh up                 .env·인증서를 만들고 DB → 앱 → 웹 순서로 빌드·실행
#   scripts/local.sh admin <이메일>      관리자 계정 만들기 (비밀번호는 입력받음)
#   scripts/local.sh fetch-data         model/.env 의 버킷 키로 팀 데이터셋을 data/ 에 내려받기
#   → 검사 화면(관리자)에서 재학습 실행, "모델 등록" 체크 → scripts/local.sh restart-api
#
# 그 밖에
#   scripts/local.sh init               .env 와 자체 서명 인증서만 만든다 (이미 있으면 건드리지 않음)
#   scripts/local.sh status             티어별 컨테이너와 모델 준비 상태
#   scripts/local.sh logs <서비스>       로그 보기 (예: logs trainer, logs nginx, logs postgres)
#   scripts/local.sh restart-api        새로 등록한 모델을 API 가 다시 불러오게 한다
#   scripts/local.sh backup             DB 백업 (~/backups/postgres)
#   scripts/local.sh down               전부 내린다 (데이터는 남김)
#   scripts/local.sh reset              전부 내리고 DB·모델 파일 볼륨까지 지운다
#
# Compose 실행기는 자동으로 고른다 (podman-compose → docker compose). 직접 정하려면 COMPOSE="docker compose" scripts/local.sh up
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_DIR="$ROOT/infra/db"
APP_DIR="$ROOT/infra/app"
WEB_DIR="$ROOT/infra/web"
MODEL_DIR="$ROOT/model"

# Compose 실행기: Podman 이 있으면 podman-compose, 없으면 docker compose
# podman-compose 는 서비스를 pod 하나로 묶지 않게(--in-pod=false) 해서 티어별 네트워크를 쓸 수 있게 한다
if [ -n "${COMPOSE:-}" ]; then
    read -r -a COMPOSE_CMD <<< "$COMPOSE"
elif command -v podman >/dev/null 2>&1 && command -v podman-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(podman-compose --in-pod=false)
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
else
    echo "podman-compose 나 docker compose 를 찾지 못했습니다" >&2
    exit 1
fi

# 티어 폴더로 들어가서 실행한다 (그 폴더의 .env 를 읽고, 프로젝트 이름을 티어별로 고정)
tier() { local name="$1" dir="$2"; shift 2; (cd "$dir" && "${COMPOSE_CMD[@]}" -p "ksdd2-$name" -f compose.yaml "$@"); }
db()  { tier db  "$DB_DIR"  "$@"; }
app() { tier app "$APP_DIR" "$@"; }
web() { tier web "$WEB_DIR" "$@"; }

env_value() { grep "^$2=" "$1" | cut -d= -f2-; }

init() {
    # 비밀번호·토큰은 영문·숫자만 나와서 DB 접속 주소에 그대로 넣을 수 있다
    if [ -f "$DB_DIR/.env" ]; then
        echo "있음: infra/db/.env"
    else
        sed "s/CHANGE_ME/$(openssl rand -hex 16)/g" "$DB_DIR/.env.example" > "$DB_DIR/.env"
        echo "만듦: infra/db/.env"
    fi

    if [ -f "$APP_DIR/.env" ]; then
        echo "있음: infra/app/.env"
    else
        sed -e "s/CHANGE_ME/$(env_value "$DB_DIR/.env" POSTGRES_PASSWORD)/g" \
            -e "s/CHANGE_TOKEN/$(openssl rand -hex 24)/g" \
            "$APP_DIR/.env.example" > "$APP_DIR/.env"
        echo "만듦: infra/app/.env"
    fi

    if [ -f "$MODEL_DIR/.env" ]; then
        echo "있음: model/.env"
    else
        cp "$MODEL_DIR/.env.example" "$MODEL_DIR/.env"
        echo "만듦: model/.env (데이터를 받으려면 버킷 키를 채우세요)"
    fi

    mkdir -p "$ROOT/data"

    if [ -f "$WEB_DIR/tls/server.crt" ]; then
        echo "있음: infra/web/tls/"
    else
        mkdir -p "$WEB_DIR/tls"
        openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=localhost" \
            -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
            -keyout "$WEB_DIR/tls/server.key" -out "$WEB_DIR/tls/server.crt" 2>/dev/null
        chmod 644 "$WEB_DIR/tls/server.key"   # nginx 컨테이너가 읽을 수 있게 (테스트용 인증서)
        echo "만듦: infra/web/tls/ (자체 서명 인증서)"
    fi
}

wait_for() {
    local name="$1" url="$2" tries="${3:-60}"
    for _ in $(seq "$tries"); do
        if curl -fsS -k -o /dev/null "$url"; then
            echo "준비됨: $name"
            return 0
        fi
        sleep 2
    done
    echo "시간 초과: $name ($url)" >&2
    return 1
}

wait_db() {
    # DB 가 접속을 받을 때까지 기다린다 (compose 실행기마다 --wait 지원이 달라서 직접 확인)
    for _ in $(seq 60); do
        if db exec -T postgres sh -c 'pg_isready -q -U "$POSTGRES_USER" -d mlflow' >/dev/null 2>&1; then
            echo "준비됨: DB"
            return 0
        fi
        sleep 2
    done
    echo "시간 초과: DB" >&2
    return 1
}

up() {
    init
    db up -d
    wait_db
    app up -d --build
    wait_for "MLflow" "http://127.0.0.1:5000/health"
    web up -d
    wait_for "웹 → API" "http://127.0.0.1:8080/api/models" 90 || true
    cat <<EOF

접속 주소 (이 컴퓨터 안에서만 열려 있음. 다른 PC 에서는 SSH 터널: ssh -L 8443:localhost:8443 -L 5000:localhost:5000 ...)
  검사 화면  https://localhost:8443   (자체 서명 인증서라 브라우저 경고가 뜸)
  Grafana    https://localhost:8443/grafana/   (검사 화면에 로그인한 뒤)
  MLflow     http://127.0.0.1:5000             (개발용)

처음이면: scripts/local.sh admin <이메일>  →  scripts/local.sh fetch-data  →  검사 화면에서 재학습
EOF
}

admin() {
    local email="${1:?사용법: scripts/local.sh admin <이메일>}"
    local password
    read -r -s -p "관리자 비밀번호: " password
    echo
    ADMIN_EMAIL="$email" ADMIN_PASSWORD="$password" app exec -T -e ADMIN_EMAIL -e ADMIN_PASSWORD api python -c '
import os
import db
from auth import hash_password
with db.connect_db() as conn:
    conn.execute(
        "INSERT INTO users (email, password_hash, role) VALUES (%s, %s, %s) "
        "ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash, role = EXCLUDED.role",
        (os.environ["ADMIN_EMAIL"], hash_password(os.environ["ADMIN_PASSWORD"]), "admin"),
    )
print("관리자 계정 준비됨:", os.environ["ADMIN_EMAIL"])
'
}

fetch_data() {
    init >/dev/null
    if ! grep -q '^OBJECT_STORAGE_ACCESS_KEY=.' "$MODEL_DIR/.env"; then
        echo "model/.env 에 OBJECT_STORAGE_ACCESS_KEY, OBJECT_STORAGE_SECRET_KEY 를 먼저 채우세요" >&2
        exit 1
    fi
    # 학습 서버 이미지 안에서 받는다 (이 컴퓨터에 파이썬 라이브러리를 깔 필요 없음). 이미 받은 파일은 건너뜀
    app run --rm --no-deps trainer python fetch_dataset.py anomaly /data/anomaly
    app run --rm --no-deps trainer python fetch_dataset.py classifier /data/classifier
    echo "데이터 준비됨: data/ (DATASET_DIR 을 바꿨다면 그 폴더)"
}

logs() {
    local name="${1:?사용법: scripts/local.sh logs <서비스>}"
    case "$name" in
        postgres) db logs -f "$name" ;;
        nginx)    web logs -f "$name" ;;
        *)        app logs -f "$name" ;;
    esac
}

status() {
    echo "[DB 티어]";  db ps
    echo "[앱 티어]";  app ps
    echo "[웹 티어]";  web ps
    echo
    curl -fsS http://127.0.0.1:8080/api/models && echo || echo "API 응답 없음"
}

case "${1:-}" in
    init)        init ;;
    up)          up ;;
    status)      status ;;
    admin)       shift; admin "$@" ;;
    fetch-data)  fetch_data ;;
    logs)        shift; logs "$@" ;;
    restart-api) app restart api ;;
    backup)      "$DB_DIR/backup/pg_backup.sh" ;;
    down)        web down; app down; db down ;;
    reset)       web down -v; app down -v; db down -v ;;
    *)           sed -n '2,25p' "$0"; exit 1 ;;
esac
