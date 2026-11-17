# 모니터링 설정

이 디렉터리는 Prometheus와 Grafana 관련 설정을 관리한다.

- prometheus/prometheus.yml: API 메트릭 수집 설정
- grafana/: Grafana 설정 파일을 관리할 영역

Prometheus와 Grafana의 컨테이너 실행 설정은
상위 디렉터리의 compose.service.yaml에서 관리한다.
