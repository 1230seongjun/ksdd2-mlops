
# MLOps Server

KSDD2 표면 결함 이상 탐지 프로젝트의 MLOps 서버 구성입니다.

## 담당 기술
- Docker Compose
- MLflow Tracking / Model Registry
- PostgreSQL
- SeaweedFS (S3 호환 아티팩트 저장소)

## 디렉터리 구조
- mlflow/: MLflow 서버 설정 및 Dockerfile
- postgres/: PostgreSQL 설정
- seaweedfs/: 모델 아티팩트 저장소 설정
- scripts/: 백업 및 운영 스크립트

## 구축 순서
1. SeaweedFS 구축
2. PostgreSQL 구축
3. MLflow Tracking Server 구축
4. MLflow와 SeaweedFS 연동
5. GPU 학습 서버에서 실험 기록 테스트
6. 서비스 서버에서 모델 다운로드 테스트

## 보안
- 비밀번호와 접근 키는 환경변수로 관리합니다.
- 실제 .env 파일은 GitHub에 업로드하지 않습니다.
- 데이터와 모델 파일은 영구 볼륨에 저장합니다.
