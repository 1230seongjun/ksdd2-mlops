"""Grafana 알림 → 카카오톡 '나에게 보내기' 중계 서버

Grafana의 webhook 수신자가 POST /alert 로 알림을 보내면,
카카오톡 메시지 API(나에게 보내기)로 요약 메시지를 보냅니다.

- 표준 라이브러리만 사용 (pip 설치 없음)
- 카카오 토큰은 /data/kakao_token.json 에 저장하고, 만료되면 refresh token으로 자동 갱신
- Grafana 외의 요청을 막기 위해 Authorization: Bearer <RELAY_TOKEN> 확인

환경변수
  KAKAO_REST_API_KEY   카카오 앱 REST API 키
  KAKAO_CLIENT_SECRET  카카오 앱 클라이언트 시크릿
  RELAY_TOKEN          Grafana webhook과 맞춘 임의의 긴 문자열
  ALERT_LINK           메시지 버튼 링크 (예: https://<서비스 주소>/grafana/alerting/list)
  TOKEN_FILE           토큰 파일 경로 (기본 /data/kakao_token.json)
"""
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KAUTH = os.environ.get("KAKAO_AUTH_BASE", "https://kauth.kakao.com")
KAPI = os.environ.get("KAKAO_API_BASE", "https://kapi.kakao.com")
REST_KEY = os.environ.get("KAKAO_REST_API_KEY", "")
SECRET = os.environ.get("KAKAO_CLIENT_SECRET", "")
RELAY_TOKEN = os.environ.get("RELAY_TOKEN", "")
ALERT_LINK = os.environ.get("ALERT_LINK", "https://developers.kakao.com")
TOKEN_FILE = os.environ.get("TOKEN_FILE", "/data/kakao_token.json")
PORT = int(os.environ.get("PORT", "8080"))
KST = timezone(timedelta(hours=9))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kakao-relay")
_lock = threading.Lock()


# ---------------- 카카오 토큰 ----------------
def _post_form(url, data, headers=None):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read().decode())


def load_token():
    with open(TOKEN_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_token(tok):
    tmp = TOKEN_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tok, f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, TOKEN_FILE)


def refresh(tok):
    data = {"grant_type": "refresh_token", "client_id": REST_KEY, "refresh_token": tok["refresh_token"]}
    if SECRET:
        data["client_secret"] = SECRET
    res = _post_form(f"{KAUTH}/oauth/token", data)
    tok["access_token"] = res["access_token"]
    tok["expires_at"] = time.time() + int(res.get("expires_in", 21599)) - 300
    if res.get("refresh_token"):  # 만료가 가까우면 카카오가 새 refresh token을 줌
        tok["refresh_token"] = res["refresh_token"]
        tok["refresh_expires_at"] = time.time() + int(res.get("refresh_token_expires_in", 0))
        log.info("refresh token 갱신됨")
    save_token(tok)
    log.info("access token 갱신됨")
    return tok


def access_token(force=False):
    with _lock:
        tok = load_token()
        if force or time.time() >= tok.get("expires_at", 0):
            tok = refresh(tok)
        return tok["access_token"]


def send_kakao(text):
    template = {"object_type": "text", "text": text[:200],
                "link": {"web_url": ALERT_LINK, "mobile_web_url": ALERT_LINK},
                "button_title": "알림 확인"}
    body = {"template_object": json.dumps(template, ensure_ascii=False)}
    for attempt in (1, 2):
        try:
            res = _post_form(f"{KAPI}/v2/api/talk/memo/default/send", body,
                             {"Authorization": f"Bearer {access_token(force=attempt == 2)}"})
            if res.get("result_code") == 0:
                return True
            log.error("카카오 응답 오류: %s", res)
            return False
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            if e.code == 401 and attempt == 1:   # 토큰 만료 → 강제 갱신 후 한 번 더
                log.warning("토큰 만료로 재시도: %s", detail)
                continue
            log.error("카카오 전송 실패 HTTP %s: %s", e.code, detail)
            return False
    return False


# ---------------- Grafana 알림 → 메시지 ----------------
def to_kst(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(KST).strftime("%m/%d %H:%M")
    except Exception:
        return ""


def build_message(payload):
    alerts = payload.get("alerts", [])
    firing = [a for a in alerts if a.get("status") == "firing"]
    resolved = [a for a in alerts if a.get("status") == "resolved"]
    lines = []
    for group, mark, word in ((firing, "🔴", "발생"), (resolved, "✅", "해결")):
        for a in group[:3]:
            lab, ann = a.get("labels", {}), a.get("annotations", {})
            lines.append(f"{mark} {word} · {lab.get('alertname', '알림')}")
            if word == "발생" and ann.get("summary"):
                lines.append(ann["summary"])
            lines.append(to_kst(a.get("startsAt" if word == "발생" else "endsAt", "")))
        if len(group) > 3:
            lines.append(f"외 {len(group) - 3}건")
    text = "\n".join(x for x in lines if x)
    return ("[불량제로]\n" + text) if text else ""


# ---------------- HTTP 서버 ----------------
class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ok = os.path.exists(TOKEN_FILE)
            return self._reply(200 if ok else 503, {"ok": ok, "token_file": ok})
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/alert":
            return self._reply(404, {"error": "not found"})
        if not RELAY_TOKEN or self.headers.get("Authorization", "") != f"Bearer {RELAY_TOKEN}":
            return self._reply(401, {"error": "unauthorized"})
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except json.JSONDecodeError:
            return self._reply(400, {"error": "invalid json"})
        text = build_message(payload)
        if not text:
            return self._reply(200, {"sent": False, "reason": "no alerts"})
        try:
            sent = send_kakao(text)
        except FileNotFoundError:
            log.error("토큰 파일이 없어 보낼 수 없습니다. get_kakao_token.py로 먼저 발급하세요.")
            sent = False
        except Exception as e:  # 네트워크 오류 등: 서버는 계속 동작
            log.exception("카카오 전송 중 오류: %s", e)
            sent = False
        log.info("알림 %s건 → 카카오 %s", len(payload.get("alerts", [])), "전송" if sent else "실패")
        # 실패 시 502: Grafana 알림 기록에 '전송 실패'로 남아 원인을 찾을 수 있음
        self._reply(200 if sent else 502, {"sent": sent})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    missing = [k for k, v in {"KAKAO_REST_API_KEY": REST_KEY, "RELAY_TOKEN": RELAY_TOKEN}.items() if not v]
    if missing:
        raise SystemExit(f"환경변수 없음: {', '.join(missing)}")
    if not os.path.exists(TOKEN_FILE):
        log.warning("토큰 파일이 없습니다(%s). get_kakao_token.py로 먼저 발급하세요.", TOKEN_FILE)
    log.info("kakao-relay 시작 :%s", PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
