#!/usr/bin/env bash
# 불량제로 로컬 실행: db → app → web 순서로 Pod 3개를 띄운다 (Podman 기본 기능만 사용).
# 먼저 이미지 3개를 빌드한다 (README 참고):
#   podman build -f Containerfile.db  -t localhost/ksdd2/db:v1  .
#   podman build -f Containerfile.app -t localhost/ksdd2/app:v1 .
#   podman build -f Containerfile.web -t localhost/ksdd2/web:v1 .
#
# 이 스크립트가 하는 일은 아래 명령을 순서대로 실행하는 것뿐이다. 직접 한 줄씩 실행해도 결과가 같다.
set -euo pipefail
cd "$(dirname "$0")"

for tier in db app web; do
    if ! podman image exists "localhost/ksdd2/$tier:v1"; then
        echo "이미지가 없습니다: localhost/ksdd2/$tier:v1 → podman build -f Containerfile.$tier -t localhost/ksdd2/$tier:v1 ." >&2
        exit 1
    fi
done

# 계층 사이 네트워크 두 개: web ↔ app, app ↔ db (web 은 DB 에 닿지 않는다)
# 대역을 직접 정하는 이유: Grafana 가 web 계층 nginx 의 고정 주소(192.168.220.10)에서 온 로그인 헤더만 믿기 때문
podman network exists ksdd2-app-db  || podman network create --subnet 192.168.210.0/24 ksdd2-app-db
podman network exists ksdd2-web-app || podman network create --subnet 192.168.220.0/24 ksdd2-web-app

podman kube play --replace --configmap demo-config.yaml --network ksdd2-app-db db.yaml
podman kube play --replace --configmap demo-config.yaml --network ksdd2-app-db --network ksdd2-web-app app.yaml
podman kube play --replace --network ksdd2-web-app:ip=192.168.220.10 web.yaml

cat <<'MSG'

실행했습니다. 처음에는 운영 모델 등록과 DB 초기화로 1~2분 걸립니다.
  검사 화면     http://localhost:8080        (데모 관리자 admin@example.com / localdemo-admin)
  Grafana       http://localhost:8080/grafana/   (검사 화면에 로그인한 뒤)
  MLflow        http://localhost:5000
  실험 대시보드  http://localhost:8502
상태: podman pod ps    준비 확인: curl -s http://localhost:8080/api/ready    로그: podman logs ksdd2-app-api
MSG
