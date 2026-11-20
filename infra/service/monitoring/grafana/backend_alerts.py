#!/usr/bin/env python3
"""불량제로 Grafana 알림 구성 스크립트 (Slack + 카카오톡 나에게 보내기)

service 서버에서 실행합니다. Grafana API로 아래를 만들거나 갱신합니다(여러 번 실행해도 됨).
  - 알림 폴더 "불량제로 알림"
  - 수신자(contact point)
      불량제로 장애 알림 : Slack #서버장애-알림 + 카카오톡
      불량제로 현장 알림 : Slack #검사품질-알림 + 카카오톡
      불량제로 테스트    : Slack #검사품질-알림 + 카카오톡
  - 알림 경로: 라벨 audience(admin / all / test)에 따라 수신자 선택
  - 알림 규칙 13개 (+ --test 옵션 시 1분 안에 울리는 테스트 규칙)
      서비스 장애 (#서버장애-알림): 검사 서버 중단, 모니터링 수집 중단, 오류율 5% 초과, DB 저장 실패, 응답 5초 초과
      모델 성능   (#검사품질-알림): 오늘 불량률 차이, 1시간 불량률 급변, 이상 점수 분포 변화,
                                    분류 확신도 비율 변화, 결함 종류 구성 변화, 운영 모델 버전 변경
      현장 검사   (#검사품질-알림): 검사 이미지 거부 증가, 재검토 필요 50건 이상
  - 모델 규칙은 운영 모델(champion) 검사만 봅니다 (challenger 비교 검사 제외)
  - 이전 버전에서 쓰던 규칙 '2시간째 검사 없음'은 실행 시 자동 삭제

사용법:
  python3 backend_alerts.py              # 수신자·경로·규칙 생성/갱신
  python3 backend_alerts.py --test       # 위 + 테스트 규칙 추가
  python3 backend_alerts.py --no-kakao   # 카카오 중계 서버 없이 Slack만
  python3 backend_alerts.py --remove-test
  python3 backend_alerts.py --drill 6      # 훈련: 6번 알림을 조건과 상관없이 실제로 한 번 울림
  python3 backend_alerts.py --drill all    # 훈련: 13개 전부 울림
  python3 backend_alerts.py --remove-drill # 훈련 규칙 삭제 (Slack·카카오에 '해결' 메시지가 감)

Grafana 비밀번호, Slack 웹훅 주소, 카카오 중계 토큰은 실행할 때 입력받습니다(화면·기록에 남지 않음).
"""
import base64
import copy
import getpass
import json
import sys
import urllib.error
import urllib.request

# ===================== 필요하면 바꾸세요 =====================
GRAFANA_URL = "http://localhost:3000/grafana"   # service 서버에서 실행할 때 주소
GRAFANA_USER = "admin"
KAKAO_RELAY_URL = "http://kakao-relay:8080/alert"  # Grafana 컨테이너 기준 중계 서버 주소
PROM_DS_NAME = None          # None이면 Prometheus 데이터 소스를 자동으로 찾음
INFERENCE_DS_NAME = None     # None이면 PostgreSQL 데이터 소스를 자동으로 찾음 (1개일 때)
# ==========================================================

FOLDER_UID = "zerodefect-alerts"
FOLDER_TITLE = "불량제로 알림"
# 분류 확신도가 이 값 미만이면 '재검토'(작업자가 이미지를 직접 확인)로 봅니다. 대시보드 low_conf와 같은 값.
LOW_CONF = 0.88
TODAY = "(date_trunc('day', now() AT TIME ZONE 'Asia/Seoul') AT TIME ZONE 'Asia/Seoul')"


# ---------------- Grafana API ----------------
class Grafana:
    def __init__(self, url, user, password):
        self.url = url.rstrip("/")
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json",
                        # 화면에서도 수정할 수 있게 (API로 만든 규칙 잠금 해제)
                        "X-Disable-Provenance": "true"}

    def call(self, method, path, body=None, ok404=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as res:
                text = res.read().decode()
                return json.loads(text) if text.strip() else {}
        except urllib.error.HTTPError as e:
            if ok404 and e.code == 404:
                return None
            raise SystemExit(f"[실패] {method} {path} → HTTP {e.code}\n{e.read().decode()[:800]}")


# ---------------- 알림 규칙 정의 ----------------
def prom_query(ref, expr, ds, seconds=300):
    return {"refId": ref, "relativeTimeRange": {"from": seconds, "to": 0}, "datasourceUid": ds,
            "model": {"refId": ref, "expr": expr, "instant": True, "range": False,
                      "intervalMs": 1000, "maxDataPoints": 43200}}


def sql_query(ref, text, ds):
    return {"refId": ref, "relativeTimeRange": {"from": 86400 * 8, "to": 0}, "datasourceUid": ds,
            "model": {"refId": ref, "rawSql": " ".join(text.split()), "format": "table",
                      "rawQuery": True, "editorMode": "code", "intervalMs": 1000, "maxDataPoints": 43200}}


def threshold(ref, src, op, value):
    """op: gt(초과) / lt(미만)"""
    return {"refId": ref, "relativeTimeRange": {"from": 0, "to": 0}, "datasourceUid": "__expr__",
            "model": {"refId": ref, "type": "threshold", "expression": src,
                      "conditions": [{"evaluator": {"type": op, "params": [value]}}]}}


def reduce_last(ref, src):
    return {"refId": ref, "relativeTimeRange": {"from": 0, "to": 0}, "datasourceUid": "__expr__",
            "model": {"refId": ref, "type": "reduce", "expression": src, "reducer": "last",
                      "settings": {"mode": "replaceNN", "replaceWithValue": 0}}}


def rule(uid, title, group, data, cond, for_, labels, summary, desc, no_data="OK", exec_err="Error"):
    return {"uid": uid, "orgID": 1, "title": title, "ruleGroup": group, "folderUID": FOLDER_UID,
            "condition": cond, "data": data, "for": for_, "noDataState": no_data,
            "execErrState": exec_err, "labels": labels, "isPaused": False,
            "annotations": {"summary": summary, "description": desc}}


def build_rules(prom, pg):
    crit = {"severity": "critical", "audience": "admin"}
    life = {"severity": "warning", "audience": "all"}
    work = {"severity": "info", "audience": "all"}
    R = []

    # ---- 1. 서비스 장애 (1분마다 확인, 관리자) ----
    R.append(rule("zd-api-down", "[장애] 검사 서버 중단", "1-서비스-장애",
                  [prom_query("A", 'max(up{job="defectcatch-api"})', prom), reduce_last("B", "A"),
                   threshold("C", "B", "lt", 1)], "C", "1m", crit,
                  "검사 서버(추론 API)가 1분 넘게 응답하지 않습니다.",
                  "작업자가 이미지 검사를 할 수 없는 상태입니다. service 서버에서 "
                  "docker ps 로 API 컨테이너 상태와 로그를 확인하세요.", no_data="Alerting", exec_err="OK"))
    R.append(rule("zd-monitor-down", "[장애] 모니터링 수집 중단", "1-서비스-장애",
                  [prom_query("A", "vector(1)", prom), reduce_last("B", "A"), threshold("C", "B", "lt", 0)],
                  "C", "1m", crit,
                  "Prometheus가 응답하지 않아 검사 서버 상태를 감시할 수 없습니다.",
                  "service 서버에서 Prometheus 컨테이너가 멈췄는지 확인하세요 "
                  "(docker start ksdd2-mlops-prometheus-1).", no_data="Alerting", exec_err="Alerting"))
    R.append(rule("zd-error-rate", "[장애] 검사 오류율 5% 초과", "1-서비스-장애",
                  # 5분간 요청이 20건 미만이면 계산하지 않음 (한두 건 실패로 울리지 않게)
                  [prom_query("A", '(100 * (sum(rate(defectcatch_predict_requests_total{status_code=~"5.."}[5m])) or vector(0))'
                                   ' / sum(rate(defectcatch_predict_requests_total[5m])))'
                                   ' and on() (sum(increase(defectcatch_predict_requests_total[5m])) >= 20)', prom),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 5)], "C", "5m", crit,
                  "최근 5분 검사 요청(20건 이상) 중 서버 오류(5xx)가 5%를 넘었습니다.",
                  "모델 로딩 실패나 MLflow 연결 문제일 수 있습니다. API 로그를 확인하세요."))
    R.append(rule("zd-save-fail", "[장애] 검사 결과 DB 저장 실패", "1-서비스-장애",
                  [prom_query("A", 'sum(increase(defectcatch_inference_save_total{result="error"}[5m])) or vector(0)', prom),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 0)], "C", "0s", crit,
                  "검사 결과가 추론 DB에 저장되지 않고 있습니다.",
                  "판정은 정상 응답되지만 이력·모니터링·재학습 데이터가 빠집니다. "
                  "mlops 서버 PostgreSQL 상태와 API의 DB 접속 정보를 확인하세요."))
    R.append(rule("zd-slow", "[주의] 검사 응답 5초 초과", "1-서비스-장애",
                  [prom_query("A", "1000 * histogram_quantile(0.95, sum by (le) "
                                   "(rate(defectcatch_predict_request_duration_seconds_bucket[5m])))", prom),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 5000)], "C", "5m",
                  {"severity": "warning", "audience": "admin"},
                  "최근 5분 검사 응답 시간(느린 쪽 5%)이 5초를 넘었습니다.",
                  "서버 CPU·메모리 사용량과 동시 요청 수를 확인하세요."))

    # ---- 2. 모델 성능 (5분마다 확인, #검사품질-알림) ----
    # 모델 비교용 challenger 검사가 섞이지 않게 운영 모델(champion) 검사만 봅니다.
    CH = "model_choice = 'champion'"

    R.append(rule("zd-defect-shift", "[모델] 오늘 불량률이 평소와 크게 다름", "2-모델-성능",
                  [sql_query("A", f"""
                      WITH t AS (SELECT count(*) AS n,
                                        100.0*count(*) FILTER (WHERE prediction='defect')/NULLIF(count(*),0) AS p
                                 FROM inference_requests WHERE {CH} AND created_at >= {TODAY}),
                           b AS (SELECT 100.0*count(*) FILTER (WHERE prediction='defect')/NULLIF(count(*),0) AS p
                                 FROM inference_requests
                                 WHERE {CH} AND created_at >= {TODAY} - interval '7 days' AND created_at < {TODAY})
                      SELECT CASE WHEN t.n >= 50 AND b.p IS NOT NULL THEN round(abs(t.p - b.p), 1) ELSE 0 END AS diff
                      FROM t, b""", pg),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 15)], "C", "0s", life,
                  "오늘 불량 판정 비율이 지난 7일 평균과 15%p 넘게 차이 납니다.",
                  "공정 변화(실제 불량 증가)이거나 모델 성능 저하일 수 있습니다. "
                  "대시보드의 '최근 불량 판정'을 전문가가 확인하고, 모델 문제라면 새 데이터 라벨링 후 재학습을 검토하세요."))

    R.append(rule("zd-defect-shift-1h", "[모델] 최근 1시간 불량률 급변", "2-모델-성능",
                  [sql_query("A", f"""
                      WITH t AS (SELECT count(*) AS n,
                                        100.0*count(*) FILTER (WHERE prediction='defect')/NULLIF(count(*),0) AS p
                                 FROM inference_requests WHERE {CH} AND created_at > now() - interval '1 hour'),
                           b AS (SELECT 100.0*count(*) FILTER (WHERE prediction='defect')/NULLIF(count(*),0) AS p
                                 FROM inference_requests
                                 WHERE {CH} AND created_at >= {TODAY} - interval '7 days' AND created_at < {TODAY})
                      SELECT CASE WHEN t.n >= 30 AND b.p IS NOT NULL THEN round(abs(t.p - b.p), 1) ELSE 0 END AS diff
                      FROM t, b""", pg),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 25)], "C", "0s", life,
                  "최근 1시간 불량 판정 비율이 지난 7일 평균과 25%p 넘게 차이 납니다.",
                  "하루 집계보다 빨리 잡는 신호입니다. 설비·조명·카메라 상태와 최근 불량 이미지를 바로 확인하세요."))

    R.append(rule("zd-score-drift", "[모델] 이상 점수 분포가 평소와 다름", "2-모델-성능",
                  [sql_query("A", f"""
                      WITH v AS (SELECT d.model_name, d.model_version
                                 FROM inference_requests r JOIN detector_results d ON d.request_id = r.id
                                 WHERE r.{CH} ORDER BY r.created_at DESC LIMIT 1),
                           x AS (SELECT r.created_at >= {TODAY} AS today, d.anomaly_score AS s
                                 FROM inference_requests r JOIN detector_results d ON d.request_id = r.id
                                 JOIN v ON v.model_name = d.model_name AND v.model_version = d.model_version
                                 WHERE r.{CH} AND r.created_at >= {TODAY} - interval '7 days'),
                           t AS (SELECT count(*) AS n, avg(s) AS m FROM x WHERE today),
                           b AS (SELECT count(*) AS n, avg(s) AS m, stddev_samp(s) AS sd FROM x WHERE NOT today)
                      SELECT CASE WHEN t.n >= 50 AND b.n >= 100 AND b.sd > 0
                                  THEN round((abs(t.m - b.m) / b.sd)::numeric, 2) ELSE 0 END AS shift
                      FROM t, b""", pg),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 0.5)], "C", "0s", life,
                  "같은 모델인데 오늘 이상 점수 평균이 지난 7일보다 크게 이동했습니다 (표준편차의 0.5배 초과).",
                  "판정이 바뀌기 전에 입력 이미지(조명·초점·카메라 위치)나 제품이 달라졌다는 신호일 수 있습니다. "
                  "최근 이미지를 평소 이미지와 비교해 보세요."))

    R.append(rule("zd-review-rate", "[모델] 분류 확신도 비율이 평소와 다름", "2-모델-성능",
                  [sql_query("A", f"""
                      WITH x AS (SELECT r.created_at >= {TODAY} AS today,
                                        (c.request_id IS NULL OR c.defect_type IS NULL OR c.confidence < {LOW_CONF}) AS review
                                 FROM inference_requests r LEFT JOIN classifier_results c ON c.request_id = r.id
                                 WHERE r.{CH} AND r.prediction = 'defect' AND r.created_at >= {TODAY} - interval '7 days')
                      SELECT CASE WHEN count(*) FILTER (WHERE today) >= 30 AND count(*) FILTER (WHERE NOT today) >= 30
                                  THEN round(abs(100.0*count(*) FILTER (WHERE today AND review)/NULLIF(count(*) FILTER (WHERE today),0)
                                           - 100.0*count(*) FILTER (WHERE NOT today AND review)
                                                      /NULLIF(count(*) FILTER (WHERE NOT today),0)), 1)
                                  ELSE 0 END AS diff
                      FROM x""", pg),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 10)], "C", "0s", life,
                  f"확신도가 낮은({LOW_CONF} 미만이거나 종류 미확인) 불량 판정 비율이 지난 7일과 10%p 넘게 차이 납니다 (증가 또는 감소).",
                  "늘었다면 분류 모델이 새로운 형태의 결함을 헷갈리는 것이고, 줄었다면 결함 크기 등 제조 과정의 변화일 수 있습니다. "
                  "전문가가 오늘 불량 이미지를 평소와 비교해 확인하고, 필요하면 재학습 데이터로 모아 주세요."))

    R.append(rule("zd-type-shift", "[모델] 결함 종류 구성이 평소와 다름", "2-모델-성능",
                  [sql_query("A", f"""
                      WITH x AS (SELECT r.created_at >= {TODAY} AS today, c.defect_type AS t
                                 FROM inference_requests r JOIN classifier_results c ON c.request_id = r.id
                                 WHERE r.{CH} AND r.prediction = 'defect' AND c.defect_type IS NOT NULL
                                   AND r.created_at >= {TODAY} - interval '7 days'),
                           n AS (SELECT count(*) FILTER (WHERE today) AS nt, count(*) FILTER (WHERE NOT today) AS nb FROM x),
                           k AS (SELECT t, count(*) FILTER (WHERE today) AS ct, count(*) FILTER (WHERE NOT today) AS cb
                                 FROM x GROUP BY t)
                      SELECT CASE WHEN n.nt >= 30 AND n.nb >= 100
                                  THEN COALESCE(round(max(abs(100.0*k.ct/NULLIF(n.nt,0) - 100.0*k.cb/NULLIF(n.nb,0))), 1), 0) ELSE 0 END AS diff
                      FROM n LEFT JOIN k ON true GROUP BY n.nt, n.nb""", pg),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 20)], "C", "0s", life,
                  "오늘 특정 결함 종류(brown/black/blue/surface)의 비율이 지난 7일과 20%p 넘게 차이 납니다.",
                  "특정 공정 단계에서 문제가 생겼을 수 있습니다. 대시보드의 '오늘 결함 종류'에서 늘어난 종류를 확인하세요."))

    R.append(rule("zd-model-change", "[모델] 운영 모델 버전이 바뀜", "2-모델-성능",
                  [sql_query("A", f"""
                      WITH r AS (SELECT r.created_at,
                                        d.model_name || ':' || d.model_version AS det,
                                        c.model_name || ':' || c.model_version AS cls
                                 FROM inference_requests r JOIN detector_results d ON d.request_id = r.id
                                 LEFT JOIN classifier_results c ON c.request_id = r.id
                                 WHERE r.{CH} AND r.created_at > now() - interval '7 days'),
                           pd AS (SELECT det FROM r WHERE created_at <= now() - interval '1 hour'
                                  ORDER BY created_at DESC LIMIT 1),
                           pc AS (SELECT cls FROM r WHERE created_at <= now() - interval '1 hour' AND cls IS NOT NULL
                                  ORDER BY created_at DESC LIMIT 1)
                      SELECT count(*) AS n FROM r
                      WHERE r.created_at > now() - interval '1 hour'
                        AND (r.det <> (SELECT det FROM pd) OR r.cls <> (SELECT cls FROM pc))""", pg),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 0)], "C", "0s", work,
                  "최근 1시간 운영 모델(champion) 검사에 이전과 다른 모델 버전이 쓰였습니다.",
                  "계획된 모델 교체라면 기록만 확인하세요. 계획에 없던 교체라면 MLflow의 champion 별칭과 배포 기록을 확인하세요. "
                  "1시간 뒤 자동으로 '해결'로 바뀝니다."))

    # ---- 3. 현장 검사 (5분~1시간마다, #검사품질-알림) ----
    R.append(rule("zd-input-reject", "[현장] 검사 이미지 거부 증가", "3-현장-검사",
                  [prom_query("A", '100 * (sum(increase(defectcatch_predict_requests_total{status_code=~"400|413|415|422"}[15m])) or vector(0))'
                                   ' / sum(increase(defectcatch_predict_requests_total[15m]))'
                                   ' and on() (sum(increase(defectcatch_predict_requests_total[15m])) >= 10)', prom, 900),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 20)], "C", "0s", life,
                  "최근 15분 검사 요청 중 20% 넘게 이미지가 거부됐습니다 (크기·형식 불일치, 빈 파일 등).",
                  "카메라 설정(해상도 184×597, JPEG/PNG)이나 촬영·업로드 방식이 바뀌었는지 현장에서 확인하세요."))

    R.append(rule("zd-review-backlog", "[현장] 오늘 재검토 필요 50건 이상", "3-현장-검사",
                  [sql_query("A", f"""
                      SELECT count(*) AS n
                      FROM inference_requests r LEFT JOIN classifier_results c ON c.request_id = r.id
                      WHERE r.prediction = 'defect' AND r.created_at >= {TODAY} AND r.reviewed_label IS NULL
                        AND (c.request_id IS NULL OR c.defect_type IS NULL OR c.confidence < {LOW_CONF})""", pg),
                   reduce_last("B", "A"), threshold("C", "B", "gt", 49)], "C", "0s", work,
                  f"오늘 작업자가 직접 확인해야 할 불량 판정(확신도 {LOW_CONF} 미만 또는 종류 미확인)이 50건을 넘었습니다.",
                  "대시보드의 재검토 목록을 확인해 주세요."))
    return R


def test_rule(prom):
    return rule("zd-test", "[테스트] 알림 동작 확인 (삭제 예정)", "0-테스트",
                [prom_query("A", "vector(1)", prom), reduce_last("B", "A"), threshold("C", "B", "gt", 0)],
                "C", "0s", {"severity": "info", "audience": "test"},
                "Slack·카카오톡 알림이 정상 동작하는지 확인하는 테스트입니다.",
                "이 메시지를 받았다면 알림 경로가 정상입니다. "
                "확인 후 python3 backend_alerts.py --remove-test 로 지우세요.")


# ---------------- 훈련(드릴) ----------------
# 실제 규칙과 같은 쿼리·라벨·문구를 쓰고, 기준만 "항상 참"으로 바꾼 복사본을 만듭니다.
# 확인할 수 있는 것: 쿼리가 오류 없이 도는지, audience에 맞는 채널로 가는지, 메시지 문구.
DRILL_GROUP = "9-훈련"


def drill_rule(r, no):
    d = copy.deepcopy(r)
    d["uid"] = f"zd-drill-{no:02d}"
    d["title"] = f"[훈련 {no}] " + r["title"]
    d["ruleGroup"] = DRILL_GROUP
    d["for"] = "0s"
    d["noDataState"] = "Alerting"      # 요청이 적어 값이 비어도 울리게
    d["execErrState"] = "Error"        # 쿼리 오류는 오류로 드러나게
    for q in d["data"]:
        if q["model"].get("type") == "threshold":
            q["model"]["conditions"] = [{"evaluator": {"type": "gt", "params": [-1e15]}}]
    d["annotations"]["summary"] = "[훈련 메시지] " + r["annotations"]["summary"]
    return d


# 이전 버전에서 만들었다가 이번 목록에서 뺀 규칙 (실행할 때 서버에서 지움)
REMOVED_RULES = ["zd-no-inspection"]

GROUP_INTERVAL = {"9-훈련": 60, "0-테스트": 60, "1-서비스-장애": 60, "2-모델-성능": 300, "3-현장-검사": 300}


# ---------------- 실행 ----------------
SLACK_TITLE = ('{{ if eq .Status "firing" }}:red_circle: 발생{{ else }}:white_check_mark: 해결{{ end }}'
               ' · {{ .CommonLabels.alertname }}')
SLACK_TEXT = ('{{ range .Alerts }}{{ if eq .Status "firing" }}*{{ .Annotations.summary }}*\n'
              '{{ .Annotations.description }}{{ else }}정상으로 돌아왔습니다.{{ end }}\n{{ end }}')


def upsert_contact(g, name, integrations):
    """같은 이름의 수신자 안에 여러 통로(Slack, 카카오 webhook)를 둠"""
    current = {c["uid"]: c for c in g.call("GET", "/api/v1/provisioning/contact-points")}
    keep = set()
    for uid, kind, settings in integrations:
        body = {"uid": uid, "name": name, "type": kind, "settings": settings, "disableResolveMessage": False}
        if uid in current:
            g.call("PUT", f"/api/v1/provisioning/contact-points/{uid}", body)
        else:
            g.call("POST", "/api/v1/provisioning/contact-points", body)
        keep.add(uid)
    # 이전 실행에서 만들었지만 이번에 빠진 통로(예: --no-kakao) 정리
    for uid, c in current.items():
        if c.get("name") == name and uid.startswith("zd-") and uid not in keep:
            g.call("DELETE", f"/api/v1/provisioning/contact-points/{uid}")
    print(f"  수신자 '{name}': {', '.join(k for _, k, _ in integrations)}")


def upsert_rule(g, r):
    if g.call("GET", f"/api/v1/provisioning/alert-rules/{r['uid']}", ok404=True):
        g.call("PUT", f"/api/v1/provisioning/alert-rules/{r['uid']}", r)
    else:
        g.call("POST", "/api/v1/provisioning/alert-rules", r)
    print(f"  규칙 {r['title']}")


def main():
    args = set(sys.argv[1:])
    pw = getpass.getpass(f"Grafana '{GRAFANA_USER}' 비밀번호: ")
    g = Grafana(GRAFANA_URL, GRAFANA_USER, pw)

    if "--remove-test" in args:
        g.call("DELETE", "/api/v1/provisioning/alert-rules/zd-test", ok404=True)
        print("테스트 규칙을 지웠습니다.")
        return

    if "--remove-drill" in args:
        for no in range(1, 100):
            if g.call("GET", f"/api/v1/provisioning/alert-rules/zd-drill-{no:02d}", ok404=True) is not None:
                g.call("DELETE", f"/api/v1/provisioning/alert-rules/zd-drill-{no:02d}")
                print(f"  삭제 zd-drill-{no:02d}")
        print("훈련 규칙을 지웠습니다. 1~2분 안에 '해결' 메시지가 갑니다.")
        return

    # 데이터 소스 UID
    ds = g.call("GET", "/api/datasources")
    def pick(name, types, label):
        if name:
            found = [d for d in ds if d["name"] == name]
        else:
            found = [d for d in ds if d["type"] in types]
        if len(found) != 1:
            raise SystemExit(f"[실패] {label} 데이터 소스를 하나로 정하지 못했습니다 (찾은 수 {len(found)}). "
                             f"현재 목록: {[(d['name'], d['type']) for d in ds]}\n"
                             f"파일 위쪽의 {'PROM_DS_NAME' if label == 'Prometheus' else 'INFERENCE_DS_NAME'}에 이름을 적어 주세요.")
        return found[0]

    prom = pick(PROM_DS_NAME, {"prometheus"}, "Prometheus")
    pg = pick(INFERENCE_DS_NAME, {"grafana-postgresql-datasource", "postgres"}, "추론 DB(PostgreSQL)")
    print(f"데이터 소스: Prometheus={prom['name']}({prom['uid']}), 추론 DB={pg['name']}({pg['uid']})")

    # 폴더
    if not g.call("GET", f"/api/folders/{FOLDER_UID}", ok404=True):
        g.call("POST", "/api/folders", {"uid": FOLDER_UID, "title": FOLDER_TITLE})
    print(f"폴더: {FOLDER_TITLE}")

    if "--drill" in args:
        pos = sys.argv.index("--drill")
        which = sys.argv[pos + 1] if pos + 1 < len(sys.argv) else ""
        rules = build_rules(prom["uid"], pg["uid"])
        if which == "all":
            nos = list(range(1, len(rules) + 1))
        elif which.isdigit() and 1 <= int(which) <= len(rules):
            nos = [int(which)]
        else:
            for i, r in enumerate(rules, 1):
                print(f"  {i:2d}. {r['title']}")
            raise SystemExit(f"--drill 뒤에 번호(1~{len(rules)}) 또는 all 을 적어 주세요.")
        print("훈련 규칙 (수신자·경로는 이미 만들어져 있어야 합니다)")
        for no in nos:
            upsert_rule(g, drill_rule(rules[no - 1], no))
        cur = g.call("GET", f"/api/v1/provisioning/folder/{FOLDER_UID}/rule-groups/{urllib.request.quote(DRILL_GROUP)}")
        cur["interval"] = 60
        g.call("PUT", f"/api/v1/provisioning/folder/{FOLDER_UID}/rule-groups/{urllib.request.quote(DRILL_GROUP)}", cur)
        print("1~2분 안에 Slack·카카오로 '[훈련 번호]' 알림이 옵니다. 확인 후 --remove-drill 로 지우세요.")
        return

    # 수신자
    slack_admin = getpass.getpass("Slack #서버장애-알림 웹훅 주소 (https://hooks.slack.com/...): ").strip()
    slack_field = getpass.getpass("Slack #검사품질-알림 웹훅 주소 (같은 채널이면 Enter): ").strip() or slack_admin
    use_kakao = "--no-kakao" not in args
    relay_token = getpass.getpass("카카오 중계 RELAY_TOKEN (.env와 같은 값): ").strip() if use_kakao else ""

    def slack(uid, url):
        return (uid, "slack", {"url": url, "title": SLACK_TITLE, "text": SLACK_TEXT, "username": "불량제로 알림"})

    def kakao(uid):
        return (uid, "webhook", {"url": KAKAO_RELAY_URL, "httpMethod": "POST",
                                 "authorization_scheme": "Bearer", "authorization_credentials": relay_token})

    print("수신자")
    for name, uid, url in (("불량제로 장애 알림", "zd-cp-admin", slack_admin),
                           ("불량제로 현장 알림", "zd-cp-all", slack_field),
                           ("불량제로 테스트", "zd-cp-test", slack_field)):
        integ = [slack(uid + "-slack", url)] + ([kakao(uid + "-kakao")] if use_kakao else [])
        upsert_contact(g, name, integ)

    # 알림 경로: audience 라벨로 수신자 선택
    policy = {
        "receiver": "불량제로 장애 알림",
        "group_by": ["grafana_folder", "alertname"],
        "group_wait": "30s", "group_interval": "5m", "repeat_interval": "4h",
        "routes": [
            {"receiver": "불량제로 테스트", "object_matchers": [["audience", "=", "test"]],
             "group_wait": "0s", "repeat_interval": "1h"},
            {"receiver": "불량제로 현장 알림", "object_matchers": [["audience", "=", "all"]],
             "repeat_interval": "12h"},
            {"receiver": "불량제로 장애 알림", "object_matchers": [["audience", "=", "admin"]]},
        ],
    }
    g.call("PUT", "/api/v1/provisioning/policies", policy)
    print("알림 경로: audience=admin → 장애 알림 / all → 현장 알림 / test → 테스트")

    # 규칙
    print("알림 규칙")
    rules = build_rules(prom["uid"], pg["uid"])
    if "--test" in args:
        rules.insert(0, test_rule(prom["uid"]))
    for r in rules:
        upsert_rule(g, r)
    for uid in REMOVED_RULES:
        if g.call("GET", f"/api/v1/provisioning/alert-rules/{uid}", ok404=True) is not None:
            g.call("DELETE", f"/api/v1/provisioning/alert-rules/{uid}")
            print(f"  삭제 {uid} (목록에서 뺀 규칙)")

    # 그룹별 확인 주기
    for group in {r["ruleGroup"] for r in rules}:
        cur = g.call("GET", f"/api/v1/provisioning/folder/{FOLDER_UID}/rule-groups/{urllib.request.quote(group)}")
        cur["interval"] = GROUP_INTERVAL.get(group, 60)
        g.call("PUT", f"/api/v1/provisioning/folder/{FOLDER_UID}/rule-groups/{urllib.request.quote(group)}", cur)
    print("완료. Grafana → Alerting → Alert rules 에서 '불량제로 알림' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
