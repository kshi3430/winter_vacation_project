#chat.py
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

load_dotenv()


# =====================
# 페이지 설정
# =====================
st.set_page_config(
    page_title="지치니 챗봇",
    page_icon="💩",
)

st.title("💩지치니")
st.caption("고민을 말해보세요")

# =====================
# 지역 선택
# =====================
user_province = st.selectbox(
    "도/광역시 선택 (예: 경상북도)",
    [
    "모든 지역",    
    "서울특별시",
    "부산광역시",
    "대구광역시",
    "인천광역시",
    "광주광역시",
    "대전광역시",
    "울산광역시",
    "세종특별자치시",
    "경기도",
    "강원특별자치도",
    "충청북도",
    "충청남도",
    "전라북도",
    "전라남도",
    "경상북도",
    "경상남도",
    "제주특별자치도"
]
)

user_city = st.text_input(
    "시/군 입력 (예: 경산시)"
)

# 고민을 넣을 상태 만들기.
if "current_concern" not in st.session_state:
    st.session_state.current_concern = None


# =====================
# session_id 생성 (중요)
# =====================
if "session_id" not in st.session_state:
    st.session_state.session_id = "session_" + str(id(st.session_state))

# =====================
# 메시지 상태
# =====================
if "messages_list" not in st.session_state:
    st.session_state.messages_list = []

# =====================
# 기존 메시지 출력
# =====================
for message in st.session_state.messages_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =====================
# 사용자 입력
# =====================
if user_question := st.chat_input("고민 분석 ai 지치니"):
    with st.chat_message("user"):
        st.write(user_question)

    st.session_state.messages_list.append(
        {"role": "user", "content": user_question}
    )

    with st.spinner("AI가 답변을 작성하는 중..."):
        # llm.py의 get_ai_response 호출
        ai_response = get_ai_response(
            user_message=user_question,
            user_province=user_province,
            user_city=user_city,
            session_id=st.session_state.session_id
        )

        # Streamlit 채팅 메시지 스트리밍
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)

        # 세션 메시지 저장
        st.session_state.messages_list.append(
            {"role": "ai", "content": ai_message}
        )
        