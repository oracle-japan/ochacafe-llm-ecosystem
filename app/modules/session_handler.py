from typing import Literal

import streamlit as st
from streamlit_feedback import streamlit_feedback
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

class SessionHandler:
    def __init__(
        self,
        st,
        session_id: str,
        feedback_option: Literal['thumbs', 'faces']
    ) -> None:
        """Initialize SessionHandler"""
        self.st = st
        self.session_id = session_id
        self.feedback_option = feedback_option
        

    def handle(self) -> None:
        """Handle session"""
        langfuse = self.st.session_state["langfuse"]
        langfuse_handler = self.st.session_state["langfuse_handler"]
        trace_id = langfuse_handler.get_trace_id()
        feedback = streamlit_feedback(
            feedback_type=self.feedback_option,
            optional_text_label="[オプション] 理由を教えてください。",
            key=self.session_id
        )
        score_mappings = {
            "thumbs": {"👍": 1, "👎": 0},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }
        scores = score_mappings[self.feedback_option]
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
