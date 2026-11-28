# 불량제로 추론 API

FastAPI가 로그인·모델 선택·이미지 추론·검사 이력·재검토를 처리합니다. Nginx 외부 경로는 `/api/`이며 아래 표는 그 접두어를 포함합니다. 실제 라우트는 Nginx가 `/api/`를 제거한 경로입니다.

## 파일과 연결

| 파일 | 역할 |
| --- | --- |
| [main.py](main.py) | HTTP 요청, 인증·모델 준비·추론·재검토·지표·GPU 요청 중계 |
| [model_service.py](model_service.py) | MLflow 모델 로드, 입력·출력 변환, 박스 처리 |
| [auth.py](auth.py) | 비밀번호·초대·세션 처리 |
| [db.py](db.py) | DB 연결, 모델 원본·검토 결과 저장, 역할별 이력 조회 |
| [서비스 Compose](../../infra/service/compose.service.yaml) | 실행 이미지와 연결 환경변수 |
| [DB 스키마](../../infra/mlops/db/) | 검사·계정·재검토 SQL. 커밋과 운영 적용은 별도 |

모델은 `MLFLOW_TRACKING_URI`, `MODEL_URI`, `CLASSIFIER_MODEL_URI`로 연결합니다. DB는 `INFERENCE_DB_HOST`, `INFERENCE_DB_PORT`, `INFERENCE_DB_NAME`, `INFERENCE_DB_USER`, `INFERENCE_DB_PASSWORD`를 사용합니다. 실제 값은 서버 환경 파일에서 관리합니다. DB 연결·스키마는 계정 인증에도 필요합니다.

## 인증과 API

로그인은 `Secure`, `HttpOnly`, `SameSite=Lax` 서비스 세션 쿠키를 발급합니다. 브라우저는 HTTPS로 사용합니다. 이력 권한은 세션의 사용자 ID·역할로 서버에서 검사합니다.

| 메서드·외부 경로 | 동작·조건 |
| --- | --- |
| `POST /api/auth/login` | 로그인·세션 쿠키 발급 |
| `GET /api/auth/me`, `POST /api/auth/logout` | 현재 계정 조회·세션 종료 |
| `POST /api/auth/invitations` | 관리자만 초대 발급 |
| `POST /api/auth/register` | 초대로 작업자 가입 |
| `GET /api/models/versions`, `GET /api/models` | 모델 버전·선택지·준비 상태 |
| `POST /api/models/{model_choice}/prepare` | 로그인 후 Registry 조합 사전 로드 |
| `POST /api/predict` | 로그인 후 이미지 검사 |
| `GET /api/inference/history` | 작업자 본인·관리자 전체 이력, 50건 단위 |
| `GET /api/inference/history/workers` | 관리자용 이력 조회 계정 목록 |
| `PUT /api/inference/{request_id}/review` | 본인 재검토 건 최종 확정·취소 |
| `GET /api/health`, `GET /api/ready` | 프로세스·기본 탐지 모델 상태 |
| `GET /api/metrics` | 요청 상태·지연·저장 결과 지표 |

이력은 파일명·판정·KST 날짜·검토 여부·점수 범위 등을 필터링하며 관리자는 작업자도 지정할 수 있습니다. `records`, `page`, `has_next`, `total`, `total_pages`를 반환합니다. 최종 작업자 판단이 있으면 이력의 표시 판정에 우선 적용하고 원래 모델 판정은 별도로 제공합니다.

## 모델 선택과 입력

`model_choice`와 `file`을 multipart form으로 전달합니다. API 기본 선택값 `champion`은 서비스 Compose에서 탐지 v2·분류 v1 URI로 고정돼 있습니다. 이는 MLflow alias의 자동 승격이나 최신 모델 자동 선택을 의미하지 않습니다. UI는 `registry-dN-cM` 조합을 사전 준비한 뒤 검사합니다. 실제 선택 버전은 응답의 모델 URI로 확인합니다.

- PNG/JPEG, 최대 5 MiB, 가로 184×세로 597px. 자동 리사이즈하지 않습니다.
- 탐지 모델이 정상·불량을 판정하고 불량일 때만 분류를 시도합니다.
- 분류 실패 시 탐지 판정은 유지하지만 재검토 분기에 들어갈 수 있습니다. 대기 저장까지 실패하면 최종 HTTP 응답은 503입니다.
- `id`, `prediction`, `model_name`, `anomaly_score`, `threshold`, `classifier_name`, `defect_type`, `confidence`, `boxes`, `review_required` 등을 반환합니다. `anomaly_score`는 보정된 불량 확률이 아닙니다.

## 재검토와 저장

API의 재검토 조건은 **불량이며 박스가 없거나 유형이 brown·black·blue·surface 중 하나가 아닌 경우**입니다. 신뢰도 0.88 조건은 폐기했습니다. UI의 크기 오류 처리는 추론 후 재검토와 다른 경로입니다.

| 상황 | 처리 |
| --- | --- |
| 일반 결과 | DB 저장 활성화 시 응답 후 요청·탐지·성공한 분류 결과를 한 트랜잭션으로 저장 |
| 재검토 대상 | `inference_review_queue`에 결과 메타데이터 저장 후 응답. DB 비활성화·저장 실패 시 503 |
| 최종 정상 | 원본 결과 저장, `reviewed_label=normal`, 유형 없음·빈 박스, 상태 finalized, 대기 행 삭제 |
| 최종 불량 | 유형 필수·박스 선택. 원본 결과와 별도 검토 컬럼 저장, 상태 finalized, 대기 행 삭제 |
| 취소 | 대기 payload를 NULL로 비우고 cancelled_at 기록. 최종 이력에는 넣지 않음 |

정상·불량 확정 요청은 예를 들어 다음 JSON을 사용합니다. 박스 좌표는 184×597 이미지 범위 안이어야 하며 w·h는 양의 정수입니다. 정상·취소에는 유형과 박스를 지정하지 않습니다.

```json
{"label":"defect","defect_type":"surface","boxes":[{"x":10,"y":20,"w":30,"h":40}]}
```

작업자 판단은 `review_status`, `reviewed_label`, `reviewed_at`, `reviewed_defect_type`, `reviewed_boxes`에 저장합니다. 모델 원래 판정·유형·영역을 덮어쓰지 않습니다. 본인 대기 건을 잠그고 원본 저장·확정 갱신·대기 행 삭제를 같은 트랜잭션에서 수행합니다. 같은 확정 요청 재전송은 저장된 결과를 비교하고, 다른 확정 값이나 취소 이후 확정은 409로 거절합니다.

메타데이터 저장은 이미지 파일 저장과 별개입니다. 서비스의 이미지·라벨 Object Storage 업로드 완료는 [확인 필요]입니다.

## 오류와 검증

| 응답 | 주요 조건 |
| --- | --- |
| 400 | 빈 파일·손상 이미지 |
| 401 / 403 | 로그인 필요 / 권한 부족 |
| 413 / 415 / 422 | 용량 초과 / 미지원 MIME / 해상도·선택값·검토 값 오류 |
| 409 | 모델 조합 미준비·충돌 또는 이미 처리한 검토와 다른 요청 |
| 500 | 탐지 추론 실패 |
| 503 | 모델 로드·조회·준비 문제 또는 재검토 대기 저장 실패 |

일반 결과의 200은 DB 저장 완료 보장이 아닙니다. `Inference save failed`와 `defectcatch_inference_save_total`을 보고 요청 ID로 실제 행을 대조합니다. `/ready`도 기본 탐지 모델 준비만 확인하므로 분류·DB·계정·선택 조합·판정 품질은 별도 검증합니다.

재검사는 새 `/predict` 요청과 새 UUID를 생성합니다. UI는 성공 시 새 결과로 교체하지만 이전 대기 건 정리 호출은 확인되지 않았습니다. 잔존 가능성은 코드상 추론이며 실제 요청 ID 대조가 필요한 후속 항목입니다. 이번 문서 갱신에서는 운영 DB 저장·최종 확정·취소를 다시 실행하지 않았습니다.

## Grafana와 GPU API 중계

Nginx 내부 인증 요청은 `/auth/grafana-proxy`에서 서비스 세션을 검사합니다. 관리자·작업자이면 204와 `X-DefectCatch-User: defectcatch-{user_id}`를 반환합니다. Nginx가 `X-WEBAUTH-USER`로 전달하며 외부 `/api/auth/grafana-proxy` 직접 호출은 Nginx에서 차단합니다. 서비스 역할 자체는 전달하지 않습니다. Grafana 권한·주소·검증 결과는 [서비스 문서](../../infra/service/README.md)를 참고합니다.

관리자만 `/api/training/jobs`의 목록·시작, `/api/training/jobs/{job_id}/log`, `/api/training/jobs/{job_id}/stop`을 사용할 수 있습니다. 서버의 `GPU_TRAINING_URL`, `GPU_TRAINING_TOKEN`으로 GPU API에 중계하며 토큰은 브라우저에 전달하지 않습니다. 학습·등록·모델 승격·배포는 이 중계 기능의 완료와 구분합니다.
