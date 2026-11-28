#!/usr/bin/env bash
# 사용: ./run_training.sh <anomaly|classifier> [train.py 인자...]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  anomaly)    MODEL_DIR="이상치 모델" ;;
  classifier) MODEL_DIR="분류 모델" ;;
  *) echo "사용법: $0 <anomaly|classifier> [train.py 인자...]" >&2; exit 2 ;;
esac
MODEL_KEY="$1"
shift

PYTHON="${PYTHON:-$HOME/venvs/defect/bin/python}"
mkdir -p "$ROOT/runs/logs"
LOG="${LOG_FILE:-$ROOT/runs/logs/$(date +%Y%m%d_%H%M%S)_${MODEL_KEY}.log}"

# GPU 한 장이라 동시 학습 불가
exec 9>"$ROOT/runs/train.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] 다른 학습이 실행 중이라 시작하지 않았습니다." | tee -a "$LOG" >&2
  echo 3 > "$LOG.exit"
  exit 3
fi

# 중단 신호는 파이썬만 종료시키고, 종료코드 기록까지 이 스크립트는 살아 있어야 한다
trap ':' TERM INT

cd "$ROOT/$MODEL_DIR" || exit 2

{
  echo "[$(date '+%F %T')] 시작: model=$MODEL_KEY args=$*"
  echo "[$(date '+%F %T')] python=$PYTHON"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
} >> "$LOG"

# 학습 데이터를 준비하는 명령이 있으면 먼저 실행한다 (로컬 실행에서만 설정: 기본 데이터 + 작업자가 확정한 검사 이미지 합치기)
if [ -n "${PREPARE_DATASET_CMD:-}" ]; then
  if ! $PREPARE_DATASET_CMD "$MODEL_KEY" >> "$LOG" 2>&1; then
    echo "[$(date '+%F %T')] 학습 데이터 준비 실패" >> "$LOG"
    echo 1 > "$LOG.exit"
    exit 1
  fi
fi

PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PYTHON" train.py "$@" >> "$LOG" 2>&1 &
child=$!
wait "$child"; code=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; code=$?; done

echo "[$(date '+%F %T')] 종료: code=$code" >> "$LOG"
echo "$code" > "$LOG.exit"
exit "$code"
