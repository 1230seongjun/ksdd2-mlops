import os
import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "http://api:8000/predict"
)

st.set_page_config(
    page_title="결함캐치",
    layout="wide"
)

st.title("결함캐치")
st.write("AI 기반 제품 표면 결함 검사")

uploaded_file = st.file_uploader(
    "검사할 이미지를 업로드하세요.",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="업로드 이미지")

    if st.button("불량 검사 시작"):
        try:
            with st.spinner("검사 중..."):
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=30
                )
                response.raise_for_status()

            st.success("요청 처리 완료")
            st.json(response.json())

        except requests.RequestException as error:
            st.error(f"API 연결 오류: {error}")
