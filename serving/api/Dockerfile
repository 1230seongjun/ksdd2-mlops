# v2 학습 환경(Python 3.11)과 MLflow 서버(3.6.0)에 맞춤
FROM python:3.11-slim

# 작업 디렉터리 설정
WORKDIR /app

# PyTorch CPU 버전 설치
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 라이브러리 목록 복사
COPY requirements.txt .


# FastAPI, MLflow 등 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 코드와 모델 추론 코드 복사
COPY main.py model_service.py ./

# FastAPI 포트
EXPOSE 8000

# API 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

