# Grafana

| 파일 | 역할 |
| --- | --- |
| [dashboards/backend-flow.json](dashboards/backend-flow.json) | 대시보드 "백엔드 처리 흐름 (실시간)". 가져올 때 Prometheus와 PostgreSQL(`inference` DB) 데이터 소스를 고릅니다 |
| [backend_alerts.py](backend_alerts.py) | 알림 수신자(Slack · 카카오톡) · 경로 · 규칙 13개를 Grafana API로 만들거나 갱신합니다. 여러 번 실행해도 됩니다 |

설정 순서와 인증 방식은 [모니터링 README](../README.md)에 있습니다.
