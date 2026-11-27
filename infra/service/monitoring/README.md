# 모니터링 · 알림

서비스 스택의 지표 수집, 대시보드, 알림 설정입니다. Prometheus와 Grafana 컨테이너는 상위 폴더의 [compose.service.yaml](../compose.service.yaml)에서 실행합니다.

| 경로 | 역할 |
| --- | --- |
| [prometheus/prometheus.yml](prometheus/prometheus.yml) | 추론 API의 `/metrics`를 15초마다 수집 (대상 `api:8000`) |
| [grafana/dashboards/backend-flow.json](grafana/dashboards/backend-flow.json) | 작업자 · 관리자 대시보드 "백엔드 처리 흐름". Prometheus와 검사 결과 DB(PostgreSQL)를 함께 씀 |
| [grafana/backend_alerts.py](grafana/backend_alerts.py) | Grafana API로 알림 수신자 · 경로 · 규칙 13개를 만드는 스크립트 |
| [kakao-relay/](kakao-relay/) | Grafana 알림을 카카오톡 "나에게 보내기"로 전달하는 중계 서버 |

## 접속과 인증

Grafana는 외부에 직접 열지 않고, Nginx의 `/grafana/` 경로로만 접근합니다. Nginx가 검사 화면 로그인 세션을 API로 확인한 뒤 사용자 이름을 헤더로 넘기고, Grafana는 Nginx의 고정 내부 IP(`172.30.250.2`)에서 온 헤더만 믿습니다(인증 프록시). 브라우저에서 보이는 주소는 서비스 `.env`의 `GF_ROOT_URL`입니다.

## 처음 설정하는 순서

1. **데이터 소스 추가 (Grafana 화면):** Prometheus(`http://prometheus:9090`)와 PostgreSQL(검사 결과 `inference` DB). PostgreSQL은 읽기 전용 계정 `grafana_reader`를 씁니다. 계정 생성 SQL은 [infra/mlops/db/003_grafana_reader.sql](../../mlops/db/003_grafana_reader.sql)에 있습니다.
2. **대시보드 가져오기:** Dashboards → Import에서 `backend-flow.json`을 올리고, 위 두 데이터 소스를 고릅니다.
3. **카카오 중계 서버 (선택):**
   ```bash
   cd kakao-relay
   cp .env.example .env    # 카카오 앱 키, RELAY_TOKEN 등 (항목 설명은 파일 안 주석)
   docker compose -f compose.kakao-relay.yaml run --rm kakao-relay python get_kakao_token.py   # 최초 1회 토큰 발급
   docker compose -f compose.kakao-relay.yaml up -d
   ```
4. **알림 규칙 만들기:** service 서버에서 `python3 grafana/backend_alerts.py`를 실행합니다. Grafana 관리자 비밀번호, Slack 웹훅 주소, `RELAY_TOKEN`은 실행할 때 입력받고 파일에 남기지 않습니다. 카카오 없이 Slack만 쓰려면 `--no-kakao`, 알림이 실제로 가는지 확인하려면 `--drill <번호|all>`을 씁니다.

## 데이터 보존

Prometheus와 Grafana 데이터는 Compose 볼륨(`prometheus-data`, `grafana-data`)에 저장됩니다. 대시보드는 저장소의 JSON이 원본이므로, 화면에서 고친 내용은 JSON으로 내보내 저장소에 반영합니다.
