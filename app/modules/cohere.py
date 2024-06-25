import streamlit as st
from langchain_cohere.chat_models import ChatCohere
from langfuse.decorators import observe

class Cohere:
    def __init__(self, cohere_api_key: str, **kwargs):
        self.cohere_api_key = cohere_api_key
        self.cohere_chat_model = self._initialize_chat_cohere(cohere_api_key=cohere_api_key, kwargs=kwargs)
        self.cohere_embeddings_model = self._initialize_embed_cohere(cohere_api_key=cohere_api_key, model_name="embed-multilingual-v3.0")
        self.callback_handler = kwargs.get("callback_handler")

    def chat(self, input: str, streaming: bool, max_tokens: int, temperature: float, k: int, p: float, frequency_penalty: float, presence_penalty: float):
        if streaming == True:
            response = self.cohere_chat_model.stream(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
                max_tokens=max_tokens,
                temperature=temperature,
                k=k,
                p=p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            for chunk in response:
                yield chunk.content
        else:
            response = self.cohere_chat_model.invoke(
                input,
                config={"callbacks": [self.callback_handler], "configurable": {"session_id": st.session_state["session_id"]}},
                max_tokens=max_tokens,
                temperature=temperature,
                k=k,
                p=p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
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
            presence_penalty=kwargs.get("presence_penalty")
        )

    def _initialize_embed_cohere(self, cohere_api_key, model_name: str) -> CohereEmbeddings:
        return CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model=model_name
        )
