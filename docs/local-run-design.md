# 로컬 실행 설계 (3계층)

이 문서는 저장소를 받은 사람이 노트북 한 대에서 불량제로를 실행할 수 있도록 바꾼 이유와 방법을 정리합니다.
코드나 설정을 바꿀 때는 이 문서에 적은 이유 중 하나에 해당해야 하며, 이유가 없는 변경은 하지 않습니다.

## 1. 왜 바꾸는가

운영 당시 저장소는 NHN Cloud 서버 두 대, NHN Container Registry(NCR), 자가 호스팅 GitHub 러너, 사설 IP,
private Object Storage를 전제로 짜여 있었습니다. 그래서 면접관이나 인사 담당자가 저장소를 받아도 실행할 수 없었습니다.

**목표:** 저장소를 받은 사람이 이미지 3개(db · app · web)를 빌드하고 실행 명령 한 줄로
운영 때와 같은 기능(로그인, 검사, 재검토, 검사 이력, 재학습, 모델 관리, 모니터링)을 직접 써 볼 수 있게 합니다.

```bash
podman build -f Containerfile.db  -t localhost/ksdd2/db:v1  .
podman build -f Containerfile.app -t localhost/ksdd2/app:v1 .
podman build -f Containerfile.web -t localhost/ksdd2/web:v1 .
deploy/up.sh
```

## 2. 외부 의존 항목을 어떻게 바꾸는가

| 항목 | 운영 당시 | 로컬 실행 | 이유 |
|---|---|---|---|
| 이미지 저장소 | NCR (사설) | 받은 사람이 직접 `podman build` | 레지스트리 계정 없이 소스만으로 재현하기 위해 |
| PostgreSQL | mlops 서버 (사설 IP) | db 계층 이미지 | 사설 IP에 닿을 수 없음 |
| MLflow 메타데이터 · 모델 레지스트리 | 운영 DB | db 계층의 `mlflow` DB | 위와 같음 |
| 모델 파일 | NHN Object Storage | GitHub Release에서 빌드할 때 받아 이미지에 포함 → 처음 실행할 때 로컬 MLflow에 등록 | NHN 계정은 부트캠프가 끝나면 사라져, NHN에 의존하면 지원 시점에 동작하지 않음 |
| 학습 데이터셋 | NHN Object Storage (800MB) | 약 50MB 부분 데이터셋을 GitHub Release에서 빌드할 때 받아 이미지에 포함 | 위와 같은 이유 + CPU로 몇 분 안에 학습할 수 있는 크기 |
| GPU 학습 서버 | 별도 서버 (외부 URL) | app 계층 안에서 CPU로 학습 | 재학습 버튼이 동작해야 함 |
| HTTPS · 공인 IP | 서버 인증서, 공인 IP 하드코딩 | `http://localhost:8080` | localhost는 브라우저가 안전한 주소로 취급해 로그인 쿠키가 그대로 동작함. 자체 서명 인증서는 경고만 늘림 |
| Slack · 카카오 알림 | 외부 토큰 | 사용하지 않음 (코드는 운영 기록으로 유지) | 토큰 없이는 동작할 수 없음 |
| CI/CD (NCR push, 러너, 서버 배포) | GitHub Actions + 자가 호스팅 러너 | 운영본은 태그 `v1.0-operated`와 문서로 보존, main CI는 3계층 빌드 검증 | 받은 사람 환경에서는 동작하지 않지만 운영 결과물이므로 지우지 않고 기록으로 남김 |

## 3. 3계층 구성

```
  브라우저 ── 127.0.0.1:8080 ──▶ [web 계층]  nginx: 검사 화면, /api · /grafana 전달
                                     │  ksdd2-web-app 네트워크
                                     ▼
                               [app 계층]  추론 API · MLflow · 학습 서버 · 실험 대시보드 · Prometheus · Grafana
                                     │  ksdd2-app-db 네트워크
                                     ▼
                               [db 계층]   PostgreSQL (mlflow DB, inference DB)
```

- 계층마다 Pod 하나로 실행합니다(`podman kube play`). 같은 계층 안의 컨테이너는 localhost로 통신합니다.
- 네트워크는 web ↔ app, app ↔ db 두 개로 나눕니다. web 계층은 DB에 닿을 수 없습니다. 운영 때 Public/Private 서브넷으로 나눴던 구조를 로컬에서도 같은 원칙으로 보여 주기 위해서입니다.
- 바깥에 여는 포트는 모두 `127.0.0.1`에만 엽니다: 검사 화면 8080, MLflow 5000, 실험 대시보드 8502.
- 학습 서버(8100)와 Prometheus(9090)는 app Pod 안에서만 엽니다. 학습 서버는 추론 API만 호출하면 되기 때문입니다.

### 이미지 3개

| 이미지 | 내용 | 이유 |
|---|---|---|
| `db` | PostgreSQL 16 + 스키마 SQL + 초기화 스크립트 | 처음 실행할 때 DB와 계정을 자동으로 만들기 위해 |
| `app` | Python 3.11 + CPU PyTorch + 추론 API · 학습 · MLflow · 실험 대시보드 코드, 부분 데이터셋, 모델 파일, 모니터링 설정 | 학습과 서빙이 같은 torch · mlflow 버전을 써야 학습한 모델을 그대로 불러올 수 있음. 서비스마다 이미지를 나누면 빌드가 6번이 되고 버전이 어긋날 수 있음 |
| `web` | nginx + 검사 화면 정적 파일 + nginx 설정 | 운영과 같은 경로(`/api`, `/grafana`)로 전달 |

Prometheus와 Grafana는 공식 이미지를 버전을 고정해 씁니다(`docker.io/prom/prometheus:v3.15.1`, `docker.io/grafana/grafana:13.3.0`, 운영 당시 실행 버전).
설정 파일은 app 이미지에 넣어 두고, Pod 시작 시 init 컨테이너가 공유 볼륨으로 복사합니다. 빌드 횟수를 3번으로 유지하면서 설정을 저장소에서 관리하기 위해서입니다.

## 4. 데이터셋과 모델 (GitHub Release)

- **부분 데이터셋:** 원본 KolektorSDD2를 184×597로 자르고 원본 GT 마스크로 정상(`normal/`) · 결함(`anomaly/`)만 나눈 약 50MB 묶음입니다. 원저작자가 공개한 정보만 쓰므로 CC BY-NC-SA 4.0 조건(비상업, 출처 표시, 같은 라이선스)으로 배포하고, 압축 파일 안에 라이선스와 출처를 함께 넣습니다.
- **공개하지 않는 것:** 팀이 원저자의 비공개 분류 기준을 따라 만든 결함 유형 라벨과 바운딩 박스는 공개 가능 여부가 불확실해 넣지 않습니다. 그래서 로컬에서 재학습할 수 있는 것은 이상 탐지 모델이고, 결함 분류 모델은 운영에서 학습한 모델을 그대로 씁니다.
- **모델:** 운영에서 쓰던 이상 탐지 모델과 결함 분류 모델을 MLflow 모델 폴더째 묶어 Release에 올립니다. 처음 실행할 때 로컬 MLflow에 **운영 때와 같은 버전 번호**로 등록합니다(예: v8). 로컬에서 재학습한 모델은 그다음 번호(v9)로 로컬에만 저장됩니다.
- **받는 방법과 검증:** 파일 주소와 sha256은 `deploy/app/assets.json` 한 곳에 적습니다. app 이미지를 빌드할 때 받아서 해시를 확인하므로, 누가 빌드해도 같은 데이터와 모델을 쓰고 빌드 후에는 인터넷 없이 실행됩니다.
- **Release 파일 만들기:** `deploy/app/pack_dataset.py`, `deploy/app/pack_model.py`로 만들고 해시를 출력합니다(팀이 한 번만 실행).

## 5. 검사 이미지로 재학습하기

운영 때는 검사 이미지를 저장하면 Object Storage 비용이 계속 늘어서 저장하지 않았습니다. 로컬에서는 검사 이미지를 저장하고, 작업자가 확정한 이미지만 재학습에 씁니다.

1. **저장:** 검사할 때 추론 API가 원본 이미지를 `<요청 ID>.png`로 app 계층 볼륨에 저장합니다(`INFERENCE_IMAGE_STORE=local`, 운영 기본값은 저장 안 함).
2. **라벨:** 작업자가 재분류한 판정은 이미 검사 이력 DB(`inference_requests.review_status`, `reviewed_label`)에 저장되므로 새 테이블을 만들지 않습니다. 요청 ID로 이미지와 이어집니다.
3. **어떤 이미지를 쓰는가:** 작업자가 확정한(`review_status = 'finalized'`) 이미지만 씁니다. 모델이 혼자 판정한 이미지까지 넣으면 모델이 자신의 오류를 다시 학습하기 때문입니다.
   - 작업자가 정상으로 확정 → 이상 탐지 학습의 `normal/` (모델이 잘못 불량으로 본 정상 이미지라 학습 가치가 큼)
   - 작업자가 결함으로 확정 → 이상 탐지 검증 · 테스트용 `anomaly/`
4. **합치기:** 재학습을 시작하면 학습 직전에 `deploy/app/build_dataset.py`가 기본 데이터셋과 확정 이미지를 합쳐 학습 폴더를 새로 만들고, "기본 몇 장 + 추가 몇 장"을 학습 로그에 남깁니다.

## 6. 팀원 `local-3tier` 브랜치에서 가져온 것과 가져오지 않은 것

**가져온 것**
- db · app · web 3계층 구분과 계층별 네트워크 분리(web은 DB에 닿지 않음)
- DB 초기화 방식: 빈 데이터 폴더로 처음 뜰 때 `inference` DB를 만들고 스키마를 순서대로 적용
- nginx resolver를 컨테이너 DNS 주소로 채우는 템플릿(Docker와 Podman 모두 동작)
- 학습 코드 `config.py`의 `DATASET_ROOT` 덮어쓰기: 팀 학습 코드를 거의 건드리지 않고 학습 폴더만 바꿀 수 있음

**가져오지 않았거나 고친 것**
- `podman-compose` 의존: 따로 설치해야 하고, 중간에 실패하면 web 계층이 뜨지 않았음 → Podman 기본 기능(`kube play`)만 사용
- 짧은 이미지 이름과 `latest` 태그(`prom/prometheus:latest`): 레지스트리 검색 설정이 없는 Podman에서 실패함(크로스 테스트에서 확인) → 전체 주소와 버전 고정
- `003_grafana_reader.sql` 미적용: Grafana의 검사 DB 패널이 비어 있었음 → 초기화 때 비밀번호까지 설정
- `MODEL_URI` 고정(`/1`): 처음 받은 사람이 학습부터 해야 검사할 수 있었음 → 운영 모델을 불러와 바로 검사
- 데이터셋을 버킷 키로 받는 방식(`fetch_dataset.py`): 키가 없는 사람은 쓸 수 없음 → Release 부분 데이터셋
- 실험 대시보드, 카카오 중계, CI/CD 일괄 삭제: 기능과 운영 기록이 사라짐 → 실험 대시보드는 app 계층에서 실행하고, 운영 전용 파일은 기록으로 유지

## 7. 운영 당시 파일

- `infra/`의 compose 파일과 모니터링 · 알림 코드는 운영 당시 구성 그대로 둡니다. 로컬 실행에서도 DB 스키마(`infra/mlops/db/*.sql`), Grafana 대시보드, 실험 대시보드 코드는 같은 파일을 그대로 씁니다.
- 운영 배포 워크플로(`deploy`, `ncr-check`, `runner-check`, `train`)와 `deploy_stack.sh`는 `docs/operations/`로 옮겨 기록으로 보존합니다. NHN 시크릿과 자가 호스팅 러너가 있어야 동작해서, `.github/workflows/`에 두면 공개 저장소의 PR마다 실패하거나 러너를 기다리며 멈추기 때문입니다. 운영 당시 상태 그대로의 저장소는 태그 `v1.0-operated`에서 볼 수 있습니다.
- main CI는 기존 테스트(`ci.yaml`)에 더해, 받은 사람과 같은 방법으로 이미지 3개를 빌드하고 실행해 확인하는 워크플로를 둡니다. README의 실행 방법이 실제로 동작하는지 PR마다 확인하기 위해서입니다.
