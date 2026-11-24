#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-/home/ubuntu/miniforge3/envs/defect/bin/python}"
EXPERIMENT="scheduled_retrain"
STAMP="$(date +%Y%m%d)"

cd "$ROOT"
bash run_training.sh anomaly --experiment "$EXPERIMENT" --run-name "anomaly_$STAMP" --no-register
bash run_training.sh classifier --fold 0 --experiment "$EXPERIMENT" --run-name "classifier_$STAMP" --no-register
