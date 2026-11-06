# MLOps Server

KSDD2 표면 결함 이상 탐지 프로젝트의 MLOps 인프라 구성 및 운영 문서입니다.

본 서버는 AI 모델의 학습 실험 기록, 모델 버전 관리, 아티팩트 저장 및 CI/CD 기반 자동 배포를 지원합니다.

## 1. 담당 기술

- **Docker Compose**: MLflow 및 PostgreSQL 컨테이너 관리
- **MLflow Tracking Server**: 모델 학습 실험, 파라미터, 성능 지표 기록
- **MLflow Model Registry**: 학습 모델 등록 및 버전 관리
- **PostgreSQL 16**: MLflow 실험 메타데이터 및 모델 등록 정보 저장
- **NHN Cloud Object Storage**: 모델 파일 및 실험 아티팩트 저장
- **GitHub Actions**: CI/CD 파이프라인 실행
- **NHN Container Registry(NCR)**: Docker 이미지 저장 및 버전 관리

## 2. 인프라 아키텍처

### 모델 학습 및 실험 관리

1. 모델팀이 KSDD2 데이터로 AI 모델을 학습합니다.
2. 학습 코드가 MLflow Tracking Server에 실험 정보를 기록합니다.
3. MLflow는 실험 메타데이터를 PostgreSQL에 저장합니다.
4. 모델 파일 및 실험 아티팩트는 NHN Object Storage에 저장합니다.
5. 필요한 모델을 MLflow Model Registry에 등록하여 버전을 관리합니다.

### CI/CD 자동 배포

1. 개발자가 GitHub에 변경된 코드를 Push합니다.
2. GitHub Actions가 설정된 조건에 따라 워크플로를 실행합니다.
3. Docker 이미지를 빌드하고 NHN Container Registry에 Push합니다.
4. 배포 스크립트가 NHN Cloud 서버에서 새 이미지를 사용하도록 배포합니다.
5. 배포 이후 서비스 동작 여부를 확인합니다.

※ 자동 배포는 팀 통합 테스트에서 성공적으로 검증하였습니다.

## 3. 주요 디렉터리 구조

```text
infra/mlops/
├── experiments-ui/       # MLflow 실험 대시보드
├── mlflow/
│   ├── Dockerfile        # MLflow 컨테이너 이미지 설정
│   └── requirements.txt  # MLflow 의존성
├── .env.example          # 환경변수 예시
├── compose.mlops.yaml    # MLflow 및 PostgreSQL 구성
└── README.md             # MLOps 인프라 설명서
```

CI/CD 관련 워크플로와 배포 스크립트는 프로젝트 루트의 `.github/` 디렉터리에서 관리합니다.

## 4. 구축 및 검증 결과

### MLflow Tracking Server

- Docker Compose 기반 MLflow 서버 구성
- 학습 실험의 파라미터 및 성능 지표 기록 확인
- MLflow UI에서 실험 결과 조회 확인
- Model Registry 등록 및 버전 관리 기능 테스트

### PostgreSQL

- PostgreSQL 16 컨테이너 구성
- MLflow Backend Store 연결
- 실험 및 모델 관련 메타데이터 저장
- Docker Volume 기반 데이터 영속성 구성

### NHN Object Storage

- S3 호환 Object Storage 버킷 구성
- MLflow 아티팩트 저장소 연동
- 실험 아티팩트 저장 및 조회 테스트

### CI/CD

- GitHub Actions 워크플로 연동
- NCR 기반 Docker 이미지 관리
- NHN Cloud 서버 자동 배포 성공 확인

## 5. 운영 및 보안

- 실제 접속 정보와 인증 키는 서버의 `.env` 파일에서 관리합니다.
- `.env` 파일은 GitHub에 업로드하지 않습니다.
- GitHub Actions에 필요한 민감 정보는 GitHub Secrets를 사용합니다.
- MLflow 관리 화면은 SSH 포트포워딩을 통해 접근합니다.
- PostgreSQL 데이터는 Docker Volume에 저장하여 컨테이너 재생성 시에도 보존하도록 구성합니다.
- 운영 환경 변경 시 서비스 중단 여부를 확인한 뒤 배포합니다.

## 6. 향후 개선 사항

- CPU 기반 모델 추론 성능 및 응답 시간 최적화
- CI/CD 빌드 및 배포 소요 시간 분석
- 모델 버전 전환 및 롤백 절차 검증
- 모델 성능 모니터링과 재학습 트리거 연동 검토
- 모델 학습부터 서비스 배포까지 라이프사이클 완성도 향상
