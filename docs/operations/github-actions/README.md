# 운영 당시 CI/CD (기록)

NHN Cloud에서 운영할 때 쓰던 GitHub Actions 워크플로와 배포 스크립트입니다.
NHN 시크릿(NCR 키 등)과 서버마다 설치한 자가 호스팅 러너가 있어야 동작합니다.
그래서 공개 저장소에서는 실행되지 않도록 `.github/workflows/`에서 이곳으로 옮겨 기록으로만 보존합니다
(`.github/workflows/`에 두면 PR마다 실패하거나 러너를 기다리며 멈춥니다).
운영 당시 저장소 상태 그대로는 태그 [`v1.0-operated`](https://github.com/1230seongjun/ksdd2-mlops/tree/v1.0-operated)에서 볼 수 있습니다.

| 파일 | 하던 일 | 실행 위치 |
|---|---|---|
| `deploy.yaml` | 변경된 경로를 감지해 api · mlflow · experiments-ui 이미지를 빌드하고 NCR에 push한 뒤, mlops → service 순서로 서버에 배포 | 빌드: GitHub 호스트 러너 / 배포: 서버별 자가 호스팅 러너(`mlops`, `service`) |
| `deploy_stack.sh` | 서버에서 파일 · 이미지를 백업하고 compose로 교체한 뒤, 준비 검사가 실패하면 이전 구성으로 되돌림 | 자가 호스팅 러너 |
| `ncr-check.yaml` | NCR 로그인 · push 가능 여부 점검 | GitHub 호스트 러너 |
| `runner-check.yaml` | 서버별 자가 호스팅 러너 동작 점검 | 자가 호스팅 러너 |
| `train.yaml` | 학습 PC(Windows GPU)에서 모델 학습 실행 | 자가 호스팅 러너(`windows`, `gpu`) |

현재 main의 CI는 [`.github/workflows/ci.yaml`](../../../.github/workflows/ci.yaml)(스모크 테스트)과
[`.github/workflows/local-build.yaml`](../../../.github/workflows/local-build.yaml)(받은 사람과 같은 방법으로 이미지 3개를 빌드해 실행 확인)입니다.
