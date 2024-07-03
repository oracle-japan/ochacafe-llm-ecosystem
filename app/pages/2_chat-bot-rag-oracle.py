import os
from dotenv import load_dotenv
import uuid

from modules.oci_generative_ai import OciGenerativeAi

import streamlit as st
from streamlit_feedback import streamlit_feedback
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

_ = load_dotenv()
compartment_id = os.getenv("COMPARTMENT_ID")
service_endpoint = os.getenv("GENAI_ENDPOINT")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST")
un = os.getenv("ORACLE_USERNAME")
pw = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_DSN")
config_dir = os.getenv("WALLET_DIR")
wallet_location = os.getenv("WALLET_DIR")
wallet_password = os.getenv("WALLET_PASSWORD")
table_name = os.getenv("TABLE_NAME")
content_column = os.getenv("CONTENT_COLUMN")
vector_column = os.getenv("VECTOR_COLUMN")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_id = st.session_state["session_id"]

if "langfuse" not in st.session_state:
    st.session_state["langfuse"] = Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=langfuse_host,
    )
langfuse = st.session_state["langfuse"]

if "langfuse_handler" not in st.session_state:
    st.session_state["langfuse_handler"] = CallbackHandler(
        secret_key=secret_key,
        public_key=public_key,
        host=langfuse_host,
        session_id=session_id
    )
langfuse_handler = st.session_state["langfuse_handler"]

st.title("🍔 OCHat w/ Oracle Database 23ai")
st.caption("""
    Cohere Command R+ と Oracle Database 23ai(ADB-S) を用いた RAG 構成のチャットボットです。 
    外部データは、[OCHaCafe Digest](https://thinkit.co.jp/series/10728)(ThinkIT) と [OCHaCafe Digest2](https://thinkit.co.jp/series/11131)(ThinkIT) を使用しています。  
    また、テキスト生成に用した時間や回答精度のスコア情報は、[Langfuse](https://langfuse.com/) に収集しています。
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar.container():
    with st.sidebar:
        st.sidebar.markdown("## Feedback Scale")
        feedback_option = (
            "thumbs" if st.sidebar.toggle(label="`Faces` ⇄ `Thumbs`", value=False) else "faces"
        )
        st.sidebar.markdown("## Vector Search Options")
        fetch_k = st.sidebar.slider(label="Fetch k", min_value=1, max_value=20, value=5, step=1)
        st.sidebar.markdown("## LLM Options")
        streaming = st.sidebar.radio(label="Streaming", options=[True, False], disabled=True, horizontal=True)
        max_tokens = st.sidebar.number_input(label="Max Tokens", min_value=10, max_value=1024, value=500, step=1)
        temperature = st.sidebar.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        k = st.sidebar.slider(label="Top k", min_value=0, max_value=500, value=0, step=1)
        p = st.sidebar.slider(label="Top p", min_value=0.0, max_value=0.99, value=0.75, step=0.01)
        frequency_penalty = st.sidebar.slider(label="Frequency Penalty", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        presence_penalty = st.sidebar.slider(label="Presence Penalty", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

cohere = OciGenerativeAi(
    compartment_id=compartment_id,
    service_endpoint=service_endpoint,
    streaming=streaming,
    max_tokens=max_tokens,
    temperature=temperature,
    k=k,
    p=p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
    fetch_k=fetch_k,
    oracle_username=un,
    oracle_password=pw,
    oracle_dsn=dsn,
    oracle_config_dir=wallet_location,
    oracle_wallet_dir=wallet_location,
    oracle_wallet_password=wallet_password,
    table_name=table_name,
    content_column=content_column,
    vector_column=vector_column,
    callback_handler=langfuse_handler,
)

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = cohere.chat_with_rag(
            input=prompt,
            streaming=streaming,
        )
        message = ""
        for content in response:
            message += content
            message_placeholder.markdown(message)
        st.session_state.messages.append({"role": "assistant", "content": message})

if session_id:
    trace_id = langfuse_handler.get_trace_id()
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[オプション] 理由を教えてください。",
        key=session_id
    )
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }
    scores = score_mappings[feedback_option]
    if feedback:
        score = scores.get(feedback.get("score"))
        comment = feedback.get("text")
        if score is not None:
            if comment is not None:
                trace_id = langfuse_handler.get_trace_id()
                langfuse.score(
                    trace_id=trace_id,
                    value=score,
                    name="user-feedback",
                    comment=comment
                )
            else:
                trace_id = langfuse_handler.get_trace_id()
                langfuse.score(
                    trace_id=trace_id,
                    value=score,
                    name="user-feedback",
                )
        else:
            st.warning("Invalid feedback score.")
