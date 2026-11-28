#!/bin/sh
# app 계층의 추론 API 시작 스크립트.
# 운영에서는 .env 의 MODEL_URI 로 모델을 고정했지만, 로컬에서는 처음 실행할 때 운영 모델을 등록하고
# 재학습하면 새 버전이 생기므로, 시작할 때마다 bootstrap.py 가 기본 모델을 정해 넘긴다.
# (재학습 후 podman restart ksdd2-app-api 를 하면 가장 최근 버전으로 다시 뜬다)
set -eu
cd /app/api
# 결과를 먼저 변수에 받아야 bootstrap.py 가 실패했을 때 set -e 로 멈춘다 (eval "$(...)" 는 실패를 놓침)
# PYTHONPATH: bootstrap.py 가 추론 API 의 db.py · auth.py 를 그대로 써서 관리자 계정을 만든다
exports=$(PYTHONPATH=/app/api python /app/local/bootstrap.py)
eval "$exports"
exec uvicorn main:app --host 0.0.0.0 --port 8000
