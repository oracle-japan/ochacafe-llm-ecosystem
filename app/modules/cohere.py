import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain_milvus import Milvus

class Cohere:
    def __init__(self, cohere_api_key: str, **kwargs):
        self.cohere_api_key = cohere_api_key
        self.cohere_chat_model = self._initialize_chat_cohere(cohere_api_key=cohere_api_key, kwargs=kwargs)
        self.cohere_embeddings_model = self._initialize_embed_cohere(cohere_api_key=cohere_api_key, model_name="embed-multilingual-v3.0")
        self.callback_handler = kwargs.get("callback_handler")
        if ("milvus_uri" in kwargs) & ("collection_name" in kwargs):
            self.milvus = Milvus(
                embedding_function=self.cohere_embeddings_model,
                collection_name=kwargs.get("collection_name"),
                connection_args={"uri": kwargs.get("milvus_uri")}
            )
            self.retriever = self.milvus.as_retriever(
                search_type=kwargs.get("search_type"),
                search_kwargs={
                    "k": kwargs.get("return_k"),
                    "score_threshold": kwargs.get("score_threshold"),
                    "fetch_k": kwargs.get("fetch_k"),
                    "lambda_mult": kwargs.get("lambda_mult")
                }
            )

    def chat(self, input: str, streaming: bool):
        if streaming == True:
            response = self.cohere_chat_model.stream(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            for chunk in response:
                if "generation_id" in chunk.response_metadata:
                    break
                yield chunk.content
        else:
            response = self.cohere_chat_model.invoke(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            return response

    def chat_with_rag(self, input: str, streaming: bool):
        langfuse = st.session_state["langfuse"]
        prompt_template = PromptTemplate.from_template(
            langfuse.get_prompt(name="demo-user-prompt", type="text").prompt,
            template_format="jinja2"
        )
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt_template
            | self.cohere_chat_model
            | StrOutputParser()
        )
        if streaming == True:
            response = chain.stream(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            for chunk in response:
                if len(chunk) > 20:
                    break
                yield chunk
        else:
            response = chain.invoke(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            return response

    def _initialize_chat_cohere(self, cohere_api_key: str, **kwargs) -> ChatCohere:
        return ChatCohere(
            cohere_api_key=cohere_api_key,
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            k=kwargs.get("k"),
            p=kwargs.get("p"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            metadata={
                "model_parameters": {
                    "streaming": kwargs.get("streaming"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                    "k": kwargs.get("k"),
                    "p": kwargs.get("p"),
                    "frequency_penalty": kwargs.get("frequency_penalty"),
                    "presence_penalty": kwargs.get("presence_penalty"),
                }
            }
        )

    def _initialize_embed_cohere(self, cohere_api_key, model_name: str) -> CohereEmbeddings:
        return CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model=model_name
        )
