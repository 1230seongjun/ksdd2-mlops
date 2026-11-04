# DefectCatch 서비스 구성

제조품 표면 이미지를 업로드하고 검사 결과를 확인하는 서비스입니다.

현재는 **더미 API**로 화면과 서버의 연결을 테스트하고 있습니다.
실제 결함 판정 모델은 아직 연결되지 않았습니다.

## 담당 범위

| 구성 | 기술 | 역할 |
|---|---|---|
| 검사 화면 | HTML·CSS·JavaScript | 이미지 업로드, 결과 및 진행 로그 표시 |
| 웹 서버 | Nginx | HTML 제공 및 API 요청 전달 |
| 추론 API | FastAPI | 이미지 요청 수신 및 검사 결과 반환 |
| 지표 수집 | Prometheus | API 요청 수와 처리 시간 수집 |
| 시각화 | Grafana | 수집한 지표를 대시보드로 표시 |
| 모델 관리 연결 | MLflow | 테스트 모델 등록 및 로드 확인 |
| 컨테이너 실행 | Docker Compose | 서비스 실행 및 네트워크 구성 |

## 현재 서비스 흐름

### 이미지 검사

```text
브라우저 → Nginx → FastAPI → 더미 응답 → 브라우저
```

FastAPI는 현재 실제 모델 대신 다음과 같은 응답을 반환합니다.

```json
{
  "filename": "example.png",
  "model_name": "dummy",
  "prediction": "not_evaluated",
  "anomaly_score": null
}
```

### 모니터링

```text
FastAPI /metrics → Prometheus → Grafana → HTML iframe
```

- Prometheus는 FastAPI 지표를 15초마다 수집합니다.
- Grafana에서 검사 요청 수와 평균 요청 처리 시간을 표시합니다.
- 현재 처리 시간은 더미 API 요청 기준입니다.
- HTML의 검사 진행 로그는 브라우저에서만 표시되고 새로고침하면 초기화됩니다.

## 관련 파일

프로젝트 최상위 기준 경로입니다.

```text
compose.service.yml                         # 서비스 실행 설정
infra/service/nginx/default.conf            # Nginx 설정
infra/service/prometheus/prometheus.yml      # Prometheus 수집 설정
infra/service/README.md                     # 서비스 안내
serving/api/main.py                         # FastAPI 및 지표 코드
serving/api/requirements.txt                # API 패키지
serving/api/Dockerfile                      # API 이미지 빌드 설정
serving/web/index.html                      # 검사 및 모니터링 화면
```

## 실행 방법

프로젝트 최상위에서 실행합니다.

```bash
cd ~/ksdd2-mlops
```

Compose 설정을 확인합니다.

```bash
docker compose -f compose.service.yml config
```

서비스를 실행합니다.

```bash
docker compose -f compose.service.yml up -d --build
```

실행 상태를 확인합니다.

```bash
docker compose -f compose.service.yml ps
```

> 현재 API는 기존 MLflow 스택의 외부 Docker 네트워크
> `ksdd2-mlops-infra_mlops-net`에 연결됩니다.
> 다른 환경에서는 해당 네트워크를 사용할 수 있는지 확인해야 합니다.

## 상태 확인

### FastAPI

```bash
curl -i http://127.0.0.1:8080/api/health
```

정상 응답:

```json
{"status":"ok"}
```

### FastAPI 지표

```bash
curl -s http://127.0.0.1:8080/api/metrics | grep defectcatch
```

### Prometheus 수집 대상

```bash
curl -s http://127.0.0.1:9090/api/v1/targets
```

`defectcatch-api` 대상의 `health`가 `up`이면 수집 중입니다.

### Grafana

```bash
curl -i http://127.0.0.1:3000/api/health
```

## 브라우저 접속

현재 테스트 환경의 Grafana는 서버의 `127.0.0.1:3000`에 연결되어 있습니다.
HTML에 포함된 Grafana 패널도 브라우저를 실행하는 PC의 `localhost:3000`을 사용합니다.

Grafana 패널과 검사 화면을 함께 볼 때는 PC에서 8080·3000 포트에 대한
SSH 터널을 연결한 뒤 다음 주소로 접속합니다.

- 검사 화면: `http://localhost:8080`
- Grafana: `http://localhost:3000`

Grafana 패널을 표시하려면 iframe 허용 설정과 Grafana 로그인 상태가 필요합니다.

## Grafana 대시보드 보관

현재 Grafana에서 만든 대시보드는 Docker 데이터 볼륨에 저장됩니다.
별도로 내보내지 않았다면 Git 저장소에는 포함되지 않습니다.

다른 환경에서 같은 대시보드를 사용하려면 대시보드 JSON과 데이터 소스 구성을
내보내거나 프로비저닝 설정을 추가해야 합니다.

현재 HTML의 iframe에는 대시보드 UID와 패널 ID가 들어 있어,
다른 Grafana 환경에서는 주소를 수정해야 할 수 있습니다.

## 남은 작업

- [ ] 실제 모델 파일과 로드 코드 확인
- [ ] `memory_bank`와 `threshold` 저장 형식 확인
- [ ] 실제 모델을 MLflow에 기록·등록
- [ ] FastAPI의 더미 응답을 실제 추론으로 교체
- [ ] 정상·결함 이미지로 추론 결과 확인
- [ ] Grafana 대시보드 공유·재생성 방법 정리
- [ ] 검사 이력 저장 항목과 저장 방식 결정
