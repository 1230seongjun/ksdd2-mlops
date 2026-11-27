# ZeroDefect 로컬 실행 가이드 (`local-3tier` 브랜치)

# RHEL / Rocky 계열
dnf install -y podman && dnf install -y epel-release && dnf install -y podman-compose
# 또는 pip install podman-compose
# Ubuntu
apt install -y podman podman-compose
#########################################################
VM 한 대에서 3티어로 전체 스택을 띄웁니다. 외부 클라우드 서비스는 쓰지 않습니다.

| 티어 | 폴더 | 구성 |
| --- | --- | --- |
| 웹 | `infra/web` | nginx (검사 화면, TLS, `/api` · `/grafana` 전달) |
| 앱 | `infra/app` | 추론 API, 학습 서버(trainer), MLflow, Prometheus, Grafana |
| DB | `infra/db` | PostgreSQL (`mlflow`, `inference` DB) |

웹 티어는 DB에 닿지 않습니다 (웹 ↔ 앱: `ksdd2-web-app`, 앱 ↔ DB: `ksdd2-app-db`).

---

## 0. 준비물

- Podman + podman-compose (또는 Docker + Compose v2)
- git, openssl, curl
- 디스크 여유 20GB 이상 (이미지 빌드 + 데이터)

```bash
podman --version && podman-compose --version
```

## 1. 코드 받기

```bash
git clone https://github.com/1230seongjun/ksdd2-mlops.git
cd ksdd2-mlops
git checkout local-3tier
```

## 2. 실행

```bash
scripts/local.sh up
```

- `.env`, DB 비밀번호, 자체 서명 인증서는 처음 실행할 때 자동으로 만들어집니다.
- 처음에는 이미지 빌드(torch 등) 때문에 오래 걸립니다. **중간에 `Ctrl+C` 하지 마세요.**
- 기다리는 동안 `curl: (7) ... Connection refused`가 여러 줄 나오는 건 서비스가 뜨기를 기다리는 중이라 정상입니다.
- 마지막에 `준비됨: 웹 → API`와 접속 주소가 나오면 성공입니다.

확인:

```bash
podman ps -a --filter name=ksdd2 --format "table {{.Names}}\t{{.Status}}"   # 7개 모두 Up
```

## 3. 관리자 계정

```bash
scripts/local.sh admin <이메일>      # 비밀번호는 입력 프롬프트에서
```

## 4. 내 PC에서 접속 (SSH 터널)

포트는 서버 내부(`127.0.0.1`)에만 열려 있습니다. 보안그룹은 열지 않아도 됩니다.

```bash
ssh -p <SSH포트> -L 8443:localhost:8443 -L 5000:localhost:5000 root@<서버주소>
```

| 주소 | 용도 |
| --- | --- |
| https://localhost:8443 | 검사 화면 (인증서 경고 → 고급 → 계속) |
| https://localhost:8443/grafana/ | 모니터링 (검사 화면 로그인 후) |
| http://localhost:5000 | MLflow |

SSH 창을 닫으면 터널도 끊깁니다. MobaXterm은 `Tunneling` 메뉴의 Local port forwarding으로 같은 설정을 할 수 있습니다.

## 5. 데이터 넣기

데이터셋은 저장소에 포함되지 않습니다. 팀에서 직접 전달받아 저장소 루트의 `data/` 아래에 아래 구조로 넣습니다.

```text
data/
├── anomaly/      normal/, anomaly/
└── classifier/   갈적색결함/, 청색 결함/, 표면 결함/, 흑색 결함/, 정상/, 결함마스크/, coco.json
```

데이터를 다른 폴더에 두려면 `infra/app/.env`에 `DATASET_DIR=/절대/경로`를 넣고 `scripts/local.sh down && scripts/local.sh up` 하세요.

> 버킷 키를 가진 사람만 `scripts/local.sh fetch-data`로 받을 수도 있습니다. 이 방식은 다음 업데이트에서 제거됩니다.

## 6. 첫 학습

새로 만든 MLflow에는 모델이 없어서, 한 번 학습해야 검사가 동작합니다.

1. 검사 화면에 관리자로 로그인
2. 재학습 실행 ("모델 등록" 체크)
3. 학습이 끝나면 `scripts/local.sh restart-api`

학습은 CPU라 오래 걸립니다. 진행 상황은 `scripts/local.sh logs trainer`로 봅니다.

---

## 자주 쓰는 명령

| 명령 | 내용 |
| --- | --- |
| `scripts/local.sh status` | 컨테이너 상태 |
| `scripts/local.sh logs <서비스>` | 로그 (`api`, `trainer`, `mlflow`, `nginx`, `postgres` 등) |
| `scripts/local.sh restart-api` | 새로 등록한 모델 반영 |
| `scripts/local.sh backup` | DB 백업 (`~/backups/postgres`) |
| `scripts/local.sh down` | 중지 (데이터 유지) |
| `scripts/local.sh reset` | 볼륨까지 전부 삭제 |

컨테이너 이름은 `ksdd2-db_postgres_1`, `ksdd2-app_api_1`, `ksdd2-web_nginx_1` 형식입니다.

## 문제가 생기면

| 증상 | 해결 |
| --- | --- |
| `could not find free subnet` / `subnet ... already used` | 다른 Podman 네트워크와 대역이 겹친 것. 우리 대역은 `192.168.210/220/230.0/24`, `192.168.250.0/29`. `podman network ls`로 확인 |
| API 오류 `database "inference" does not exist` | `podman exec ksdd2-db_postgres_1 bash /docker-entrypoint-initdb.d/10-inference.sh` 실행 후 `scripts/local.sh restart-api` |
| 컨테이너가 `Exited` | `podman logs --tail 30 <컨테이너이름>`으로 원인 확인 |
| 8443 / 5000 접속 안 됨 | SSH 창이 열려 있는지, 내 PC 포트가 겹치는지 확인 (`-L 18443:localhost:8443`처럼 바꿔 접속) |
| `git apply` 가 `does not match index` | `git update-index --refresh` 후 다시 실행 |
| SELinux (RHEL 계열) | 따로 설정할 것 없음. compose 마운트에 `:z`가 들어 있음. **SELinux를 끄지 마세요** |

## 지켜 주세요

- `.env`, `infra/web/tls/`, `data/`는 git에 올리지 않습니다 (`.gitignore`에 포함).
- KolektorSDD2는 CC BY-NC-SA 4.0입니다. 전달받은 데이터는 공개 저장소나 공개 링크에 올리지 않습니다.
