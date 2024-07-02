import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings

class OciGenerativeAi:
    def __init__(self, compartment_id: str, service_endpoint: str, **kwargs):
        self.compartment_id = compartment_id
        self.service_endpoint = service_endpoint
        self.chat_model = self._initialize_chat_model(
            is_stream=kwargs.get("streaming"),
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),

        )
        self.embeddings_model = self._initialize_embedding_model()
        self.callback_handler = kwargs.get("callback_handler")
        if ("milvus_uri" in kwargs) & ("collection_name" in kwargs):
            self.milvus = Milvus(
                embedding_function=self.embeddings_model,
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
            response = self.chat_model.stream(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            for chunk in response:
                yield chunk.content
        else:
            response = self.chat_model.invoke(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            return response.content

    def chat_with_rag(self, input: str, streaming: bool):
        langfuse = st.session_state["langfuse"]
        prompt_template = PromptTemplate.from_template(
            langfuse.get_prompt(name="demo-user-prompt", type="text").prompt,
            template_format="jinja2"
        )
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        if streaming == True:
            response = chain.stream(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            for chunk in response:
                yield chunk
        else:
            response = chain.invoke(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
            )
            return response

    def _initialize_chat_model(self, is_stream: bool, max_tokens: int, temperature: float, top_k: int, top_p: float, frequency_penalty: float, presence_penalty: float) -> ChatOCIGenAI:
        return ChatOCIGenAI(
            auth_type="INSTANCE_PRINCIPAL",
            model_id="cohere.command-r-plus",
            compartment_id=self.compartment_id,
            service_endpoint=self.service_endpoint,
            is_stream=is_stream,
            model_kwargs={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            },
            metadata={
                "model_name": "cohere.command-r-plus",
                "model_parameters": {
                    "is_stream": is_stream,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
            }
        )

    def _initialize_embedding_model(self) -> OCIGenAIEmbeddings:
        return OCIGenAIEmbeddings(
            auth_type="INSTANCE_PRINCIPAL",
            compartment_id=self.compartment_id,
            service_endpoint=self.service_endpoint,
            model_id="cohere.embed-multilingual-v3.0"
        )
