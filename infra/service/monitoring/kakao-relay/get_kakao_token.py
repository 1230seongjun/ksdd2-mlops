"""카카오 토큰 최초 발급 (한 번만 실행)

relay 컨테이너 안에서 실행합니다.
  docker compose -f compose.kakao-relay.yaml run --rm kakao-relay python get_kakao_token.py

1. 화면에 나온 주소를 브라우저에서 열고 카카오 로그인 → '카카오톡 메시지 전송'에 동의
2. 등록한 Redirect URI(예: https://localhost/kakao)로 이동하면서 페이지가 안 열려도 정상
3. 주소창의 전체 주소(…?code=XXXX)를 복사해 붙여 넣기
4. /data/kakao_token.json 에 토큰이 저장됨 (화면에 토큰을 출력하지 않음)
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

KAUTH = os.environ.get("KAKAO_AUTH_BASE", "https://kauth.kakao.com")
REST_KEY = os.environ["KAKAO_REST_API_KEY"]
SECRET = os.environ.get("KAKAO_CLIENT_SECRET", "")
REDIRECT = os.environ.get("KAKAO_REDIRECT_URI", "https://localhost/kakao")
TOKEN_FILE = os.environ.get("TOKEN_FILE", "/data/kakao_token.json")

auth_url = f"{KAUTH}/oauth/authorize?" + urllib.parse.urlencode(
    {"client_id": REST_KEY, "redirect_uri": REDIRECT, "response_type": "code", "scope": "talk_message"})
print("\n1) 아래 주소를 브라우저에서 열고 로그인·동의하세요.\n")
print(auth_url)
raw = input("\n2) 이동한 페이지의 전체 주소(또는 code 값)를 붙여 넣으세요: ").strip()
code = urllib.parse.parse_qs(urllib.parse.urlparse(raw).query).get("code", [raw])[0]

data = {"grant_type": "authorization_code", "client_id": REST_KEY, "redirect_uri": REDIRECT, "code": code}
if SECRET:
    data["client_secret"] = SECRET
req = urllib.request.Request(f"{KAUTH}/oauth/token", data=urllib.parse.urlencode(data).encode(), method="POST",
                             headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"})
try:
    with urllib.request.urlopen(req, timeout=10) as res:
        tok = json.loads(res.read().decode())
except urllib.error.HTTPError as e:
    raise SystemExit(f"토큰 발급 실패 HTTP {e.code}: {e.read().decode()[:300]}")

if "talk_message" not in tok.get("scope", ""):
    print("주의: 'talk_message' 동의가 빠졌습니다. 카카오 앱 동의항목 설정을 확인하세요.")
saved = {"access_token": tok["access_token"], "refresh_token": tok["refresh_token"],
         "expires_at": time.time() + int(tok.get("expires_in", 21599)) - 300,
         "refresh_expires_at": time.time() + int(tok.get("refresh_token_expires_in", 0))}
os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
with open(TOKEN_FILE, "w", encoding="utf-8") as f:
    json.dump(saved, f)
os.chmod(TOKEN_FILE, 0o600)
days = int(tok.get("refresh_token_expires_in", 0)) // 86400
print(f"\n저장 완료: {TOKEN_FILE} (refresh token 유효 약 {days}일, 사용 중 자동 연장)")
