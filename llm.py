#llm.py
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from dotenv import load_dotenv
import os

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda



load_dotenv()

# =========================
# 세션별 히스토리 저장소
# =========================
store = {}

session_concern_store = {}

session_seen_store = {}

def get_seen_ids(session_id):
    return session_seen_store.get(session_id, set())


def set_seen_ids(session_id, ids):
    session_seen_store[session_id] = ids


def get_session_concern(session_id):
    return session_concern_store.get(session_id, "")


def set_session_concern(session_id, concern):
    session_concern_store[session_id] = concern


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# =========================
# LLM
# =========================
def get_llm(model="solar-1-mini-chat"):
    return ChatUpstage(model=model)



def format_docs(docs):
    return "\n\n".join(
        f"- user_id: {doc.metadata['user_id']}\n"
        f"- province: {doc.metadata['province']}\n"
        f"- city: {doc.metadata['city']}\n"
        f"- 고민 내용: {doc.page_content}"
        for doc in docs
    )



# =========================
# Retriever (지역 필터 포함)
# =========================
def get_retriever(user_province: str, user_city: str):
    embeddings = UpstageEmbeddings(
        model="solar-embedding-1-large"
    )

    vectorstore = PineconeVectorStore(
        index_name="jichini-real-index",
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )

    # 🔥 필터 구조 명확하게
    filter_dict = {}

    # 🔥 전체 지역이면 필터 아예 없음
    if user_province != "모든 지역":

        if user_province and user_city:
            # 둘 다 있으면 둘 다 필터
            filter_dict = {
                "province": user_province,
                "city": user_city
            }

        elif user_province:
            # province만
            filter_dict = {
                "province": user_province
            }

    # 🔥 search_kwargs 분리
    search_kwargs = {
        "k": 15,
    }

    # 🔥 filter가 있을 때만 추가
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    return vectorstore.as_retriever(search_kwargs=search_kwargs)




# =========================
# History-aware retriever
# =========================
def get_history_retriever(user_province: str, user_city: str):
    llm = get_llm()
    retriever = get_retriever(user_province, user_city)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "이전 대화 내용을 참고해서 현재 질문을 독립적인 질문으로 바꿔라."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )



def get_classification_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
너는 사용자의 입력을 분류하는 AI"지치니"이다.

반드시 아래 중 하나만 정확히 출력하라 (다른 말 금지)
- 고민
- 인사
- 욕설
- 잡담

                                      
사용자 입력:
{input}
""")

    return prompt | llm



def get_guide_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
너는 사용자가 고민을 자연스럽게 말하도록 돕는 AI"지치니"이다.

규칙:
- 설교하지 마라
- 사용자가 자연스럽게 고민을 말하도록 유도하라
- 사용자가 인사, 욕설, 잡담을 했을 때는 자연스럽게 고민을 말하도록 유도하라
- 절대 사용자가 한 말을 무시하지 마라

입력 유형: {type}
사용자 입력: {input}
""")

    return prompt | llm



def get_intent_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
사용자의 고민 입력이 아래 중 무엇인지 분류하라.

반드시 아래 중 하나만 정확히 출력하라
- concern
- feedback


입력:
{input}
""")

    return prompt | llm



def get_merge_chain():
    llm = get_llm()

    merge_prompt = ChatPromptTemplate.from_template("""
기존 고민과 피드백을 반영해 하나의 자연스러운 고민으로 만들어라.

[규칙]
- 피드백을 반드시 반영
- 자연스럽게 이어붙이기
- 너무 길지 않게
- 핵심 고민이 무엇인지 명확하게 유지

기존 고민:
{original}

피드백:
{feedback}

새로운 고민:
""")

    return merge_prompt | llm


def is_more_request(text: str) -> bool:
    keywords = [
        "더 보여",
        "다른 사람",
        "추가",
        "더 추천",
        "또 있",
        "더 있",
        "더 없",
        "또 없"
    ]

    return any(k in text for k in keywords)




# =========================
# RAG Chain
# =========================
def get_rag_chain(user_province: str, user_city: str):
    llm = get_llm()
    system_prompt = f"""
[역할]
너는 Context를 그대로 출력하는 매칭 결과 생성기이다.

[절대 규칙]
- Context에 없는 정보는 절대 생성하지 마라
- 문장을 수정, 요약, 재작성하지 마라
- 반드시 Context의 '고민 내용'을 그대로 복사하라
- 다른 사용자와 내용을 섞지 마라
- 각 사용자 블록을 그대로 유지하라
- 출력 형식을 절대 변경하지 마라
- 줄바꿈을 유지하라

[출력 형식]
- user_id:
- province:
- city:
- 고민 내용:

[출력 방법]
Context의 [USER] 블록 단위로 최대 3개를 선택하여 그대로 복사하여 출력하라.
"""


    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("system", "{context}"),  
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)



    history_aware_retriever = get_history_retriever(user_province, user_city)

    def format_docs_safe(docs):
            if not docs:
                return ""
            return "\n\n".join(
                f"- user_id: {doc.metadata['user_id']}\n"
                f"- province: {doc.metadata['province']}\n"
                f"- city: {doc.metadata['city']}\n"
                f"- 고민 내용: {doc.page_content}"

                for doc in docs
            )

    document_chain = create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}\n\n[Context]\n{context}")
            ])
        )
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_chain



def string_to_stream(text):
    if not text:
        yield ""
    else:
        for line in text.split("\n"):
            yield line + "\n"




# =========================
# Streamlit에서 부르는 함수
# =========================
def get_ai_response(user_message, user_province, user_city, session_id="default"):
    classifier = get_classification_chain()
    intent_chain = get_intent_chain()
    merge_chain = get_merge_chain()
    guide_chain = get_guide_chain()

    # =========================
    # 0️⃣ 더보기 요청 (최우선)
    # =========================
    
    if is_more_request(user_message):
        prev = get_session_concern(session_id)

        if not prev:
            return string_to_stream("먼저 고민을 입력해 주세요.")

        # 정상 more 로직
        retriever = get_history_retriever(user_province, user_city)

        docs = retriever.invoke({
            "input": prev,
            "chat_history": []
        })

       

        seen_ids = get_seen_ids(session_id)
        if not seen_ids:
            seen_ids = set()

            

        result = ""
        count = 0

        for doc in docs:
            uid = doc.metadata['user_id']

            if uid in seen_ids:
                continue

            result += (            
                f"- 사용자 ID: {uid}\n"
                f"- 시/도: {doc.metadata['province']}\n"
                f"- 시/군: {doc.metadata['city']}\n"
                f"- 고민 내용: {doc.page_content}\n\n"
            )

            seen_ids.add(uid)
            count += 1

            if count == 3:
                break

        set_seen_ids(session_id, seen_ids)

        if count == 0:
            return string_to_stream("더 이상 추천할 사용자가 없습니다.")

        result += (
    "\n더 많은 사용자를 보고 싶다면 '더 보여줘'라고 말해 주세요."
    "\n고민을 조금 더 자세히 말해주면 더 비슷한 사람을 찾아드릴 수 있어요."
)


        return string_to_stream(result)



    # =========================
    # 1️⃣ category 먼저 판단
    # =========================
    category = classifier.invoke({"input": user_message}).content.strip()
    category = category.replace("\n", "").strip()

    if category not in ["고민", "인사", "욕설", "잡담"]:
        category = "잡담"

    # =========================
    # 2️⃣ 고민 아니면 guide
    # =========================
    if category != "고민":
        return guide_chain.stream({
            "input": user_message,
            "type": category
        })

    # =========================
    # 3️⃣ 고민일 때만 intent 판단
    # =========================
    intent = intent_chain.invoke({"input": user_message}).content.strip().lower()

    if intent == "feedback":
        intent = "feedback"
    else:
        intent = "concern"

    # =========================
    # 4️⃣ concern / feedback 처리
    # =========================
    if intent == "concern":
        current_concern = user_message

    elif intent == "feedback":
        prev = get_session_concern(session_id)

        if not prev:
            current_concern = user_message
        else:
            merged = merge_chain.invoke({
                "original": prev,
                "feedback": user_message
            })
            current_concern = merged.content.strip()

    # 저장
    set_session_concern(session_id, current_concern)

    # =========================
    # 5️⃣ retriever
    # =========================
    retriever = get_history_retriever(user_province, user_city)

    docs = retriever.invoke({
        "input": current_concern,
        "chat_history": []
    })

    if not docs:
        return string_to_stream("조건에 맞는 사용자가 없습니다.")

    # =========================
    # 6️⃣ 결과 출력
    # =========================
    seen_ids = get_seen_ids(session_id)
    if not seen_ids:
        seen_ids = set()

    result = ""
    count = 0

    for doc in docs:
        uid = doc.metadata['user_id']

        if uid in seen_ids:
            continue

        result += (
            f"- 사용자 ID: {uid}\n"
            f"- 시/도: {doc.metadata['province']}\n"
            f"- 시/군: {doc.metadata['city']}\n"
            f"- 고민 내용: {doc.page_content}\n\n"
        )

        seen_ids.add(uid)
        count += 1

        if count == 3:
            break

    set_seen_ids(session_id, seen_ids)

    result += (
    "\n더 많은 사용자를 보고 싶다면 '더 보여줘'라고 말해 주세요."
    "\n고민을 조금 더 자세히 말해주면 더 비슷한 사람을 찾아드릴 수 있어요."
)


    return string_to_stream(result)