# 불량제로 추론 API

FastAPI 기반 검사 API입니다. 로그인한 사용자의 이미지를 검사하고 결과를 반환하며, 검사 이력을 PostgreSQL에 저장합니다. 서비스 배포와 HTTPS 구성은 [서비스 운영 가이드](../../infra/service/README.md)를 참고합니다.

## API 경로

아래 표는 FastAPI 내부 경로입니다. 브라우저에서 호출할 때는 Nginx를 통해 `/api`를 앞에 붙입니다. 예를 들어 FastAPI의 `/ready`는 외부에서 `/api/ready`로 호출합니다.

| 메서드·경로 | 기능 |
| --- | --- |
| `GET /health` | API 프로세스 응답 확인 |
| `GET /ready` | 추론 모델 준비 상태 확인. 준비되지 않으면 `503` |
| `GET /models` | 모델 선택지와 로딩 상태 조회 |
| `POST /predict` | 로그인 후 이미지 검사 요청 |
| `POST /auth/login` | 로그인 및 세션 쿠키 발급 |
| `GET /auth/me` | 현재 로그인한 사용자 ID와 역할 조회 |
| `POST /auth/logout` | 현재 세션 무효화 |
| `POST /auth/invitations` | 관리자가 작업자 초대 발급 |
| `POST /auth/register` | 초대 토큰으로 작업자 가입 |
| `GET /inference/history` | 역할에 따라 검사 이력 조회 |
| `GET /metrics` | Prometheus용 지표 노출 |

## 이미지 검사 규격

`/predict`는 `multipart/form-data` 요청에서 이미지 파일과 모델 선택값을 받습니다. 입력 이미지는 **가로 184 × 세로 597 픽셀**, 파일 크기는 **최대 5 MiB**입니다. API는 이미지 형식과 크기를 검증한 뒤 추론합니다.

응답에는 정상·불량 판정에 사용하는 `is_anomaly`, 이상 점수 `anomaly_score`, 기준값 `threshold`가 포함됩니다. 결함 유형과 영역은 분류 결과가 있는 경우 확인할 수 있습니다.

| 응답 | 확인할 상황 |
| --- | --- |
| `401` | 로그인 세션이 없거나 유효하지 않음 |
| `413` | API의 파일 크기 제한 초과 |
| `415` | 지원하지 않는 파일 형식 |
| `422` | 이미지 해상도 또는 요청 입력값 검증 실패 |
| `503` | 해당 모델이 준비되지 않음 |

현재 배포 Compose에서 `champion` 선택지는 탐지 모델 `models:/Zero-Defects/2`와 분류 모델 `models:/Zero-Defects-Classifier/1`을 사용합니다. 서비스의 선택지 이름이 `champion`이어도 실제로 로드하는 모델은 배포 설정에 지정된 URI를 따릅니다.

## 로그인·초대·권한

- 비밀번호는 Argon2 해시로 저장합니다.
- 로그인 후 세션 쿠키를 발급합니다. 쿠키에는 `HttpOnly`, `Secure`, `SameSite=Lax`가 설정되며 세션은 8시간 동안 유효합니다.
- 초대 링크는 관리자만 발급할 수 있습니다. 초대 토큰은 24시간 동안 유효하고 한 번 사용하면 재사용할 수 없습니다. 세션·초대 토큰은 해시로 저장합니다.
- 작업자는 자신의 검사 기록을, 관리자는 전체 검사 기록을 조회합니다. 서버는 로그인 세션으로 사용자를 식별하고 DB 조회에 역할 조건을 적용합니다.
- 검사 이력은 권한 범위 안에서 **최신 50건**을 반환합니다. 더 오래된 기록을 탐색하는 페이지 분할 기능은 아직 구현되지 않았습니다.

## 검사 데이터 저장

검사 결과는 PostgreSQL의 `inference` 데이터베이스에 저장합니다.

| 테이블 | 저장 내용 |
| --- | --- |
| `inference_requests` | 검사 요청 ID, 사용자 ID, 파일명, 선택 모델, 판정, 처리 시간 |
| `detector_results` | 탐지 모델·버전, 이상 점수, 탐지 결과, 처리 시간 |
| `classifier_results` | 분류 모델·버전, 결함 유형, 신뢰도, 결함 영역, 처리 시간 |
| `users` | 사용자 계정과 역할 |
| `invitations`, `sessions` | 초대와 로그인 세션 |

API는 추론 응답을 반환하고, DB 저장 설정이 활성화된 경우 백그라운드에서 검사 요청과 모델별 결과를 하나의 트랜잭션으로 기록합니다. 분류 결과가 있을 때만 `classifier_results`에 행을 추가합니다. 검사 이력이 누락되면 API 로그의 `Inference save failed`와 `defectcatch_inference_save_total` 지표를 확인합니다.

DB 연결은 `INFERENCE_DB_HOST`, `INFERENCE_DB_PORT`, `INFERENCE_DB_NAME`, `INFERENCE_DB_USER`, `INFERENCE_DB_PASSWORD`를 사용합니다. 운영 `inference` DB에는 계정·세션·초대 테이블과 검사 소유자 컬럼을 추가하는 `002_accounts.sql`이 적용돼 있습니다. 새 환경에서는 [DB 스키마와 마이그레이션](../../infra/mlops/db/)을 확인합니다.

## 코드 위치

| 파일 | 역할 |
| --- | --- |
| [`main.py`](main.py) | API 경로, 입력 검증, 모델 준비 상태, 검사 및 요청 지표 |
| [`model_service.py`](model_service.py) | 모델 로딩, 입력 변환, 추론 응답 처리 |
| [`auth.py`](auth.py) | 비밀번호 확인, 초대, 계정 등록, 세션 관리 |
| [`db.py`](db.py) | DB 연결, 검사 결과 저장, 사용자 역할별 이력 조회 |
| [`Dockerfile`](Dockerfile), [`requirements.txt`](requirements.txt) | API 이미지 빌드와 Python 의존성 설정 |

## 검증 기준

운영 환경에서 HTTPS 관리자 로그인, 세션을 통한 `/api/auth/me` 조회, 브라우저 검사, 관리자·작업자 검사 기록 저장을 확인했습니다. 로그인하지 않은 `/api/auth/me`는 `401`을 반환했습니다.

권한 변경을 검증할 때는 관리자와 작업자를 서로 다른 브라우저 세션에서 로그인합니다. 작업자는 자신의 기록만, 관리자는 두 계정의 기록을 모두 조회할 수 있는지 확인합니다. 저장 여부를 대조할 때는 MLOps 서버의 `inference_requests`에서 파일명·시각·`user_id`를 확인합니다.
