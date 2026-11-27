# 모델 학습

두 단계 검사 모델의 학습 코드입니다. 학습 결과는 MLflow에 기록되고, `SAVE_MODEL: true`이면 Model Registry에 등록됩니다.

| 경로 | 역할 |
|---|---|
| `이상치 모델/` | 1단계 이상 탐지 (PatchCore, Wide ResNet50-2). 정상 이미지만으로 학습 |
| `분류 모델/` | 2단계 결함 유형 분류 (U-Net, ResNet34 인코더). 5-fold 교차검증 |
| `common.yaml` | 두 모델 공통 설정 (Object Storage 엔드포인트 · 버킷, 학습 장치, MLflow 주소) |
| `gpu_training_api.py` | GPU 서버에서 띄우는 재학습 API (FastAPI). 서비스 화면의 재학습 버튼이 호출 |
| `run_training.sh` | 재학습 API가 실행하는 학습 스크립트 (백그라운드 실행 · 로그 저장) |
| `주간 자동 학습/` | cron으로 주 1회 정기 재학습 (`crontab.txt`에 등록 예시) |

## 준비

1. 패키지 설치: `pip install -r requirements.txt`
2. 설정 파일: `cp .env.example .env` 후 값을 채웁니다. `model/.env` 하나를 두 모델 폴더가 함께 읽습니다.
3. 학습 데이터: KolektorSDD2는 라이선스상 저장소에 없습니다. 원본을 내려받아 S3 호환 버킷에 올리고 `common.yaml`의 `OBJECT_STORAGE_ENDPOINT` · `OBJECT_STORAGE_BUCKET`을 맞춥니다.
4. MLflow: 실행 중인 MLflow 서버 주소를 `MLFLOW_TRACKING_URI`에 넣습니다. 운영 당시에는 Private 서브넷의 MLflow에 SSH 터널(`127.0.0.1:5000`)로 접속했습니다.

## 학습

```bash
# 1단계 이상 탐지
cd "이상치 모델"
python train.py --config config.yaml

# 2단계 결함 유형 분류 (fold 0~4를 각각 학습한 뒤 요약)
cd "../분류 모델"
python train.py --config config.yaml --fold 0
python summary.py
```

평가 원칙(검증 데이터로만 선택, 테스트는 최종 모델 1회)과 실험 기록은 [루트 README의 모델 섹션](../README.md#-모델)에 있습니다.
