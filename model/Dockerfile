# 학습 서버 이미지: 학습에 필요한 라이브러리만 넣고, 코드는 compose 가 model/ 폴더를 그대로 연결한다
# (코드를 고쳐도 다시 빌드할 필요 없음)
FROM python:3.11-slim

# 서빙 이미지(serving/api/Dockerfile)와 같은 버전의 CPU 용 PyTorch
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt fastapi uvicorn

WORKDIR /app/model

# 학습 서버(gpu_training_api.py)가 학습을 실행할 때 쓰는 파이썬
ENV GPU_TRAINING_PYTHON=/usr/local/bin/python

EXPOSE 8100
CMD ["uvicorn", "gpu_training_api:app", "--host", "0.0.0.0", "--port", "8100"]
