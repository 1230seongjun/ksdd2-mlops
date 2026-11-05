# DefectCatch 서비스 구성

제조품 표면 이미지를 업로드하고 검사 결과를 확인하는 서비스입니다.

현재는 **더미 API**를 이용하여 검사 화면과 서버의 연결을 테스트하고 있습니다. 실제 결함 판정 모델은 아직 추론 API에 연결되지 않았습니다.

서비스는 Docker Compose로 구성하며, 운영 배포는 GitHub Actions 기반 CI/CD 파이프라인을 통해 수행하도록 전환 중입니다.

## 1. 담당 범위

| 구성 | 기술 | 역할 |
|---|---|---|
| 검사 화면 | HTML·CSS·JavaScript | 이미지 업로드, 결과 및 진행 로그 표시 |
| 웹 서버 | Nginx | HTML 제공 및 API 요청 전달 |
| 추론 API | FastAPI | 이미지 요청 수신 및 검사 결과 반환 |
| 지표 수집 | Prometheus | API 요청 수와 처리 시간 수집 |
| 시각화 | Grafana | 수집한 지표를 대시보드로 표시 |
| 모델 관리 연결 | MLflow | 모델 등록 및 로드 확인 |
| 컨테이너 실행 | Docker Compose | 서비스 실행 및 네트워크 구성 |
| 이미지 저장소 | NHN Container Registry | 빌드된 Docker 이미지 저장 |
| CI/CD | GitHub Actions | 이미지 빌드, 배포 및 헬스체크 자동화 |

## 2. 서비스 아키텍처

### 이미지 검사 흐름

```text
사용자
  │ 이미지 업로드
  ▼
브라우저 (HTML·JavaScript)
  │
  ▼
Nginx (:8080)
  │ /api 요청 전달
  ▼
FastAPI
  │
  ├── 현재: 더미 응답
  │
  └── 향후: MLflow 등록 모델 로드 및 추론
  │
  ▼
검사 결과 반환
  │
  ▼
브라우저에 결과 표시
```

현재 FastAPI는 실제 모델 대신 다음과 같은 더미 응답을 반환합니다.

```json
{
  "filename": "example.png",
  "model_name": "dummy",
  "prediction": "not_evaluated",
  "anomaly_score": null
}
```

### 모니터링 흐름

```text
FastAPI /metrics
       │
       ▼
Prometheus
       │
       ▼
Grafana
       │
       ▼
HTML iframe
```

- Prometheus는 FastAPI 지표를 15초마다 수집합니다.
- Grafana에서 검사 요청 수와 평균 요청 처리 시간을 표시합니다.
- 현재 처리 시간은 더미 API 요청을 기준으로 합니다.
- HTML의 검사 진행 로그는 브라우저에서만 표시되며, 새로고침하면 초기화됩니다.

## 3. 프로젝트 파일 구조

프로젝트 최상위 디렉터리 기준입니다.

```text
ksdd2-mlops/
│
├── .github/
│   ├── workflows/
│   │   └── deploy.yaml
│   └── scripts/
│       └── deploy_stack.sh
│
├── infra/
│   └── service/
│       ├── compose.service.yaml
│       ├── README.md
│       │
│       ├── nginx/
│       │   └── default.conf
│       │
│       └── prometheus/
│           └── prometheus.yml
│
└── serving/
    ├── api/
    │   ├── main.py
    │   ├── requirements.txt
    │   └── Dockerfile
    │
    └── web/
        └── index.html
```

`deploy.yaml`과 `deploy_stack.sh`는 CI/CD 담당자가 관리하는 배포 파일이며, 통합 과정에서 반영합니다.

### 주요 파일 역할

| 파일 | 역할 |
|---|---|
| `compose.service.yaml` | 서비스 컨테이너, 볼륨, 네트워크 설정 |
| `nginx/default.conf` | 웹 서버 및 API 프록시 설정 |
| `prometheus/prometheus.yml` | Prometheus 지표 수집 설정 |
| `serving/api/main.py` | FastAPI 및 모니터링 지표 코드 |
| `serving/api/Dockerfile` | API Docker 이미지 빌드 |
| `serving/web/index.html` | 검사 및 모니터링 화면 |
| `deploy.yaml` | GitHub Actions 빌드·배포 워크플로 |
| `deploy_stack.sh` | 서버 배포, 헬스체크 및 롤백 처리 |

## 4. Docker Compose 구성

서비스는 다음 4개 컨테이너로 구성됩니다.

| 서비스 | 이미지 | 포트 |
|---|---|---|
| FastAPI | NCR의 API 이미지 | 내부 네트워크 |
| Nginx | `nginx:stable-alpine` | 8080 |
| Prometheus | `prom/prometheus:latest` | 127.0.0.1:9090 |
| Grafana | `grafana/grafana:latest` | 127.0.0.1:3000 |

### Compose 프로젝트 이름

```yaml
name: ksdd2-mlops
```

기존 컨테이너와 Docker 볼륨을 유지하기 위해 프로젝트 이름을 고정합니다.

### API 이미지

API 이미지는 배포 서버에서 직접 빌드하지 않고 NCR에서 가져옵니다.

```yaml
image: ${NCR_REGISTRY}/${REGISTRY_PATH}/api:${API_TAG:?API_TAG is not set in .env}
```

환경변수 역할:

- `NCR_REGISTRY`: NCR 레지스트리 주소
- `REGISTRY_PATH`: 이미지 저장 경로
- `API_TAG`: 이미지 버전 식별용 Git 커밋 앞 7자리

### MLflow 네트워크

FastAPI는 MLOps 스택이 생성하는 외부 Docker 네트워크에 연결됩니다.

```yaml
networks:
  mlops-net:
    external: true
    name: ksdd2-mlops-infra_mlops-net
```

MLflow 서비스의 내부 주소:

```text
http://mlflow:5000
```

서비스 스택을 배포하기 전에 MLOps 네트워크가 존재해야 합니다.

## 5. CI/CD 배포 방법

운영 환경의 배포는 GitHub Actions를 통해 수행합니다.

### 5.1 자동 배포

`main` 브랜치에 PR이 병합되면 변경된 파일에 따라 필요한 이미지 빌드 및 배포를 수행합니다.

```text
GitHub PR 병합
       │
       ▼
변경 파일 확인
       │
       ▼
필요한 Docker 이미지 빌드
       │
       ▼
NCR에 이미지 Push
       │
       ▼
배포 서버에서 이미지 Pull
       │
       ▼
Docker Compose 컨테이너 배포
       │
       ▼
헬스체크
       │
       ├── 성공 → 배포 완료
       │
       └── 실패 → 이전 이미지 태그로 롤백
```

이미지 빌드가 실패하면 신규 이미지 배포를 진행하지 않습니다.

이미지를 정상적으로 배포한 후 헬스체크가 실패하면 이전 이미지 태그를 사용하여 롤백합니다.

### 5.2 변경 파일별 배포

| 변경 경로 | 이미지 빌드 | 배포 대상 |
|---|---|---|
| `serving/api/**` | API 빌드 | 서비스 스택 |
| `serving/web/**` | 없음 | 서비스 스택 |
| `infra/service/**` | 없음 | 서비스 스택 |
| `infra/mlops/mlflow/**` | MLflow 빌드 | MLOps 스택 |
| `infra/mlops/compose.mlops.yaml` | 없음 | MLOps 스택 |

웹 파일 및 Nginx·Prometheus 설정 변경은 이미지 재빌드 없이 파일 복사와 필요한 설정 재적용을 통해 배포합니다.

### 5.3 수동 배포

GitHub Actions에서 배포 워크플로를 직접 실행할 수 있습니다.

1. GitHub 저장소의 **Actions** 탭에 접속합니다.
2. **Build and deploy** 워크플로를 선택합니다.
3. **Run workflow**를 실행합니다.
4. 필요한 `target`을 선택합니다.

| target | 실행 내용 |
|---|---|
| `all` | API·MLflow 빌드 및 두 스택 배포 |
| `api` | API 이미지 빌드 및 서비스 배포 |
| `mlflow` | MLflow 이미지 빌드 및 MLOps 배포 |
| `deploy-only` | 이미지 빌드 없이 재배포 |

### 5.4 운영 배포 주의사항

운영 서버에서 직접 다음 명령어로 배포하지 않습니다.

```bash
docker compose up -d --build
```

프로젝트 이름이 같으면 기존 CI/CD 관리 컨테이너에 영향을 줄 수 있습니다.

운영 배포는 `main` 병합 또는 GitHub Actions의 `Run workflow`를 통해 수행합니다.

## 6. Compose 설정 검증

프로젝트 최상위 디렉터리에서 실행합니다.

```bash
cd ~/ksdd2-mlops
```

Compose 설정 검증:

```bash
NCR_REGISTRY=x REGISTRY_PATH=y API_TAG=abc1234 \
docker compose -f infra/service/compose.service.yaml config -q
```

오류 없이 종료되면 Compose 설정 검증을 통과한 것입니다.

이 검증은 실제 이미지 다운로드, 컨테이너 실행, 서비스 정상 작동까지 보장하지 않습니다.

## 7. 배포 서버 구조

배포 파일은 서버의 다음 디렉터리에서 관리합니다.

```text
/opt/bulryangzero/
│
├── mlops/
│   ├── compose.mlops.yaml
│   └── .env
│
└── service/
    ├── compose.service.yaml
    ├── .env
    │
    ├── nginx/
    │   └── default.conf
    │
    ├── prometheus/
    │   └── prometheus.yml
    │
    └── web/
        └── index.html
```

Git 저장소의 `serving/web/` 파일은 배포 과정에서 서버의 `service/web/`으로 복사합니다.

### 환경변수 관리

서비스 환경변수 파일:

```text
/opt/bulryangzero/service/.env
```

배포 워크플로는 이미지 태그와 레지스트리 경로 정보를 관리합니다.

인증 키, 비밀번호 등 민감한 정보는 Git 저장소에 커밋하지 않습니다.

## 8. 배포 후 상태 확인

### 8.1 컨테이너 상태

배포 서버에서 실행합니다.

```bash
docker ps
```

컨테이너의 상태와 사용 중인 이미지 태그를 확인합니다.

### 8.2 FastAPI 헬스체크

```bash
curl -i http://127.0.0.1:8080/api/health
```

기존 API의 정상 응답 예시:

```json
{
  "status": "ok"
}
```

배포 스크립트는 해당 주소를 사용하여 서비스 상태를 확인합니다.

실제 모델이 아직 연결되지 않아도 `/health` 엔드포인트는 정상 응답을 유지해야 합니다.

### 8.3 FastAPI 지표

```bash
curl -s http://127.0.0.1:8080/api/metrics | grep defectcatch
```

FastAPI에서 노출하는 Prometheus 지표를 확인합니다.

### 8.4 Prometheus 수집 대상

```bash
curl -s http://127.0.0.1:9090/api/v1/targets
```

`defectcatch-api` 대상의 `health`가 `up`이면 Prometheus가 해당 대상에서 지표를 정상적으로 수집 중인 상태입니다.

### 8.5 Grafana

```bash
curl -i http://127.0.0.1:3000/api/health
```

Grafana API의 응답 상태를 확인합니다.

## 9. 브라우저 접속

서비스 웹 화면은 서버의 8080 포트를 통해 제공됩니다.

```text
http://<서버 플로팅 IP>:8080
```

현재 테스트 환경의 Grafana는 서버의 `127.0.0.1:3000`에 연결되어 있습니다.

HTML에 포함된 Grafana 패널도 브라우저를 실행하는 PC의 `localhost:3000`을 사용합니다.

Grafana 패널과 검사 화면을 함께 볼 때는 PC에서 8080·3000 포트에 대한 SSH 터널을 연결한 뒤 다음 주소로 접속합니다.

- 검사 화면: `http://localhost:8080`
- Grafana: `http://localhost:3000`

Grafana 패널을 표시하려면 iframe 허용 설정과 Grafana 로그인 상태가 필요합니다.

## 10. Grafana 대시보드 보관

현재 Grafana에서 만든 대시보드는 Docker 데이터 볼륨에 저장됩니다.

별도로 내보내지 않았다면 Git 저장소에는 포함되지 않습니다.

다른 환경에서 같은 대시보드를 사용하려면 대시보드 JSON과 데이터 소스 구성을 내보내거나 프로비저닝 설정을 추가해야 합니다.

현재 HTML의 iframe에는 대시보드 UID와 패널 ID가 들어 있어, 다른 Grafana 환경에서는 주소를 수정해야 할 수 있습니다.

### 볼륨 유지

Compose 프로젝트 이름을 변경하지 않아야 기존 Grafana 볼륨을 계속 사용할 수 있습니다.

```yaml
name: ksdd2-mlops
```

기존 볼륨:

```text
ksdd2-mlops_grafana-data
```

프로젝트 이름이 변경되면 새로운 이름의 볼륨이 생성되어 기존 대시보드가 보이지 않을 수 있습니다.

## 11. 배포 실패 및 롤백

배포 과정에서 신규 컨테이너의 헬스체크가 실패하면 배포 스크립트가 이전 이미지 태그로 복구를 시도합니다.

```text
새 이미지 배포
      │
      ▼
헬스체크
      │
      ├── 성공 → 배포 완료
      │
      └── 실패
           │
           ▼
      오류 로그 기록
           │
           ▼
      이전 .env 복원
           │
           ▼
      이전 이미지로 재배포
```

서비스 헬스체크 주소:

```text
http://127.0.0.1:8080/api/health
```

주요 확인 사항:

- API의 `/health` 엔드포인트 유지
- Nginx의 `/api/` 프록시 설정 유지
- NCR 이미지 및 태그 존재 여부
- 서비스 환경변수 설정
- Docker 네트워크 연결 상태

배포 로그는 GitHub Actions의 해당 실행 기록에서 확인합니다.

## 12. 남은 작업

### CI/CD 연결

- [x] Compose 파일을 `infra/service/`로 이동
- [x] API 이미지 설정을 NCR 방식으로 변경
- [x] Nginx 볼륨 경로 수정
- [x] Prometheus 볼륨 및 실행 옵션 수정
- [x] Compose 프로젝트 이름 고정
- [x] Compose 설정 검증
- [ ] README 변경 사항 커밋
- [ ] CI/CD 통합 PR 병합
- [ ] 첫 자동 배포 검증
- [ ] API 변경 후 자동 빌드·배포 확인
- [ ] 기존 Grafana 데이터 유지 확인

### 모델 연동

- [ ] 실제 모델 파일과 로드 코드 확인
- [ ] `memory_bank`와 `threshold` 저장 형식 확인
- [ ] 실제 모델을 MLflow에 기록·등록
- [ ] `models:/ksdd2-classifier@champion` 모델 로드 구현
- [ ] FastAPI의 더미 응답을 실제 추론으로 교체
- [ ] `/admin/reload` 모델 재로드 기능 구현
- [ ] 정상·결함 이미지로 추론 결과 확인
- [ ] 모델 버전 및 불량 판정 지표 추가
- [ ] Grafana 대시보드 공유·재생성 방법 정리
- [ ] 검사 이력 저장 항목과 저장 방식 결정

---

**현재 단계:** 서비스 Compose의 CI/CD 전환 설정을 수정하고 정적 검증을 수행한 상태입니다. 실제 GitHub Actions 통합 배포와 모델 추론 연동은 별도 검증이 필요합니다.
