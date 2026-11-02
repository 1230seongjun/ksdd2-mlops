# FastAPI 추론 API

이미지 이상 탐지 API를 구축하기 위한 초기 코드입니다.

## 파일 구성

- main.py: FastAPI 앱과 상태 확인 API
- requirements.txt: Python 의존성
- Dockerfile: 컨테이너 빌드·실행 설정

## 현재 구현

GET /health 요청에 다음 JSON을 반환합니다.

{"status":"ok"}

이 응답은 API 응답 확인용이며 모델 준비 상태를 의미하지 않습니다.

## 아직 구현하지 않은 기능

- 이미지 업로드
- 모델 로딩·이상 탐지
- 검사 결과 DB 저장
- Streamlit·Nginx 연결

## 검증 상태

코드 작성 완료.
이미지 빌드·컨테이너 실행 여부는 확인 후 기록합니다.
