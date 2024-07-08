import os
from dotenv import load_dotenv
import uuid

import streamlit as st
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from modules.oci_generative_ai import OciGenerativeAi
from modules.session_handler import SessionHandler

_ = load_dotenv()
compartment_id = os.getenv("COMPARTMENT_ID")
service_endpoint = os.getenv("GENAI_ENDPOINT")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST")
milvus_uri = os.getenv("MILVUS_URI")
collection_name = os.getenv("COLLECTION_NAME")

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

st.title("🍔 OCHat w/ Milvus")
st.caption("""
    Cohere Command R+ と Milvus on OKE(Oracle Container Engine for Kubernetes) を用いた RAG 構成のチャットボットです。 
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
        search_type = st.sidebar.radio("Search type", ["similarity", "mmr", "similarity_score_threshold"], horizontal=True, disabled=True)
        return_k = st.sidebar.slider(label="Return k", min_value=1, max_value=10, value=4, step=1)
        fetch_k = st.sidebar.slider(label="Fetch k", min_value=1, max_value=20, value=5, step=1)
        score_threshold = st.sidebar.slider(label="Score threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.1, disabled=True)
        lambda_mult = st.sidebar.slider(label="Lambda milt", min_value=0.0, max_value=1.0, value=0.5, step=0.1, disabled=True)
        st.sidebar.markdown("## LLM Options")
        streaming = st.sidebar.radio(label="Streaming", options=[True, False], disabled=True, horizontal=True)
        max_tokens = st.sidebar.number_input(label="Max Tokens", min_value=10, max_value=1024, value=500, step=1)
        temperature = st.sidebar.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        k = st.sidebar.slider(label="Top k", min_value=0, max_value=500, value=0, step=1)
        p = st.sidebar.slider(label="Top p", min_value=0.0, max_value=0.99, value=0.75, step=0.01)
        frequency_penalty = st.sidebar.slider(label="Frequency Penalty", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        presence_penalty = st.sidebar.slider(label="Presence Penalty", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

oci_genai = OciGenerativeAi(
    compartment_id=compartment_id,
    service_endpoint=service_endpoint,
    streaming=streaming,
    max_tokens=max_tokens,
    temperature=temperature,
    k=k,
    p=p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
    search_type=search_type,
    return_k=return_k,
    score_threshold=score_threshold,
    lambda_mult=lambda_mult,
    milvus_uri=milvus_uri,
    collection_name=collection_name,
    callback_handler=langfuse_handler,
)

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = oci_genai.chat_with_rag(
            input=prompt,
            streaming=streaming,
        )
        message = ""
        for content in response:
            message += content
            message_placeholder.markdown(message)
        st.session_state.messages.append({"role": "assistant", "content": message})

if session_id:
    session_handler = SessionHandler(
        st=st,
        session_id=session_id,
        feedback_option=feedback_option
    )
    session_handler.handle()
