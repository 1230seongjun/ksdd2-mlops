#!/usr/bin/env bash
# 불량제로 로컬 실행 종료: web → app → db 순서로 Pod 를 내린다.
#   deploy/down.sh            데이터(DB · 모델 · 검사 이미지 볼륨)는 남긴다
#   deploy/down.sh --volumes  볼륨까지 지워 처음 상태로 되돌린다
set -euo pipefail
cd "$(dirname "$0")"

force=()
if [ "${1:-}" = "--volumes" ]; then
    force=(--force)
fi

for file in web.yaml app.yaml db.yaml; do
    podman kube down "${force[@]}" "$file" || true   # 이미 내려가 있어도 나머지는 계속 내린다
done
