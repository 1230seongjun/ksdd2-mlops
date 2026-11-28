# app 계층: 추론 API · MLflow · 학습 서버 · 실험 대시보드를 하나의 이미지로 만든다.
# 빌드: podman build -f Containerfile.app -t localhost/ksdd2/app:v1 .
#
# 왜 하나의 이미지인가: 학습한 모델을 추론 API 가 그대로 불러오려면 학습과 서빙의 torch · mlflow 버전이 같아야 한다.
# 서비스마다 이미지를 나누면 빌드가 여러 번 필요하고 버전이 어긋날 수 있어, 한 이미지에 넣고 실행 명령만 다르게 한다(deploy/app.yaml).
FROM docker.io/library/python:3.11-slim

# 운영 추론 API(serving/api/Dockerfile)와 같은 CPU 버전의 PyTorch
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 각 서비스의 의존성 목록을 운영과 같은 파일에서 그대로 설치한다 (목록을 새로 만들지 않아 운영과 어긋나지 않게)
COPY serving/api/requirements.txt /tmp/requirements/api.txt
COPY model/requirements.txt /tmp/requirements/model.txt
COPY infra/mlops/mlflow/requirements.txt /tmp/requirements/mlflow.txt
COPY infra/mlops/experiments-ui/requirements.txt /tmp/requirements/experiments-ui.txt
RUN pip install --no-cache-dir \
      -r /tmp/requirements/api.txt -r /tmp/requirements/model.txt \
      -r /tmp/requirements/mlflow.txt -r /tmp/requirements/experiments-ui.txt \
 && rm -rf /tmp/requirements

# 이상 탐지 백본(Wide ResNet50-2, ImageNet V1) 가중치를 미리 받아 둔다 → 재학습할 때 인터넷이 없어도 시작
RUN python -c "from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights as W; wide_resnet50_2(weights=W.IMAGENET1K_V1)"

# 부분 데이터셋과 운영 모델을 GitHub Release 에서 받아 sha256 을 확인한다 (목록: deploy/app/assets.json)
# 코드보다 먼저 받아서, 코드만 고쳤을 때는 이 단계를 다시 하지 않게 한다
COPY deploy/app/assets.json deploy/app/fetch_assets.py /tmp/assets/
RUN python /tmp/assets/fetch_assets.py /tmp/assets/assets.json /opt/ksdd2 && rm -rf /tmp/assets

# 서비스 코드: 운영과 같은 파일을 그대로 쓴다
COPY serving/api/ /app/api/
COPY model/ /app/model/
COPY infra/mlops/experiments-ui/ /app/experiments-ui/
COPY deploy/app/ /app/local/

# 로컬(CPU)에서 바꾸는 학습 설정 — 원본 config.yaml 은 저장소에 그대로 두고 이미지 안에서만 바꾼다
#   FINE_TUNE_STAGE 0      : 백본 파인튜닝은 CPU 로 너무 오래 걸려 끈다 (파인튜닝 전 PatchCore 로 학습)
#   CORESET_POOL_SIZE 100000: coreset 후보 패치 30만 개(약 1.8GB 메모리)를 노트북에 맞게 줄인다
RUN sed -i \
      -e 's#^FINE_TUNE_STAGE:.*#FINE_TUNE_STAGE: 0#' \
      -e 's#^CORESET_POOL_SIZE:.*#CORESET_POOL_SIZE: 100000#' \
      "/app/model/이상치 모델/config.yaml" \
 && grep -E '^(FINE_TUNE_STAGE|CORESET_POOL_SIZE):' "/app/model/이상치 모델/config.yaml"

# Grafana 대시보드: 운영 대시보드 JSON 의 데이터 소스 자리표시자를 로컬 데이터 소스 uid 로 바꿔 넣는다
# (운영에서는 Grafana 화면에서 가져오기 할 때 골랐던 값. 설정: deploy/app/grafana/datasources/datasources.yaml)
COPY infra/service/monitoring/grafana/dashboards/ /app/local/grafana/dashboards/files/
RUN sed -i -e 's/\${DS_PROMETHEUS}/prometheus/g' -e 's/\${DS_INFERENCE}/inference/g' \
      /app/local/grafana/dashboards/files/*.json

# 학습 서버(model/gpu_training_api.py)가 학습을 실행할 파이썬 (기본값은 운영 GPU 서버의 경로라 바꿔 준다)
ENV GPU_TRAINING_PYTHON=/usr/local/bin/python \
    PYTHONIOENCODING=utf-8

WORKDIR /app
CMD ["/app/local/start-api.sh"]
