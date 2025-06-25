import streamlit as st
from streamlit_javascript import st_javascript
import os
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from dataclasses import dataclass
from typing import Literal
from core.config import google_api
from Chatbot.ChromaDB import vector_store, document_content_description, metadata_field_info, df
from Chatbot.main import chain

def chatbot():
    USER_AVATAR = r"UA_United\user_icon.png"
    BOT_AVATAR = r"UA_United\ai_icon.png" 

    os.environ["GOOGLE_API_KEY"] = google_api

    llm= ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        max_tokens=None,  
        timeout=None,
        max_retries=2)

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=True)
    
    @dataclass
    class Message:
        origin: Literal["human", "ai"]
        message: str
    
    # ==== Load CSS ====
    def load_css():
        with open(r"styles.css", "r", encoding="utf-8") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    
    # ==== Initialize State ====
    def initialize_session_state():
        if "history" not in st.session_state:
            st.session_state.history = []
        if "user_prompt" not in st.session_state:
            st.session_state.user_prompt = ""
    # ==== Handle Chat Submission ====
    def on_click_callback():
        user_prompt = st.session_state.human_prompt
        if not user_prompt.strip():
            return
        
        st.session_state.history.append(Message("human", user_prompt))
        
        # Use your own logic for generating response
        reviews = retriever.invoke(user_prompt)
        result = chain.invoke({"reviews": reviews, "question": user_prompt})
        #ai_response = result_response["output"]
        
        st.session_state.history.append(Message("ai", result.content))
        
        # Làm trống ô nhập câu hỏi sau khi submit
        st.session_state.human_prompt = ""
    # ==== App UI ====  
    load_css()
    initialize_session_state()
    
    emotion_class = st_javascript("""
() => {
    const avatarImg = document.querySelector('img[alt="user avatar"]');
    if (!avatarImg) return null;
    const parentDiv = avatarImg.closest('div');
    if (!parentDiv) return null;
    const className = parentDiv.className;
    const emotionClass = className.split(' ').find(cls => cls.includes('st-emotion-cache'));
    return emotionClass;
}
""")
    if emotion_class:
        st.session_state["emotionClass"] = emotion_class
    st.title("🧠 Chatbot Gemini - Tùy biến giao diện")
    
    # Chat Container
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")        
    # ==== Display Chat History ====
    with chat_placeholder:
        for chat in st.session_state.history:
            role = "user" if chat.origin == "human" else "assistant"
            avatar = USER_AVATAR if role == "user" else BOT_AVATAR
            
            with st.chat_message(role, avatar=avatar):
                st.markdown(chat.message)
                if "emotionClass" in st.session_state:
                    st.markdown(f"""
                    <style>
                    .{st.session_state.emotionClass} {{
                        flex-direction: row-reverse;
                        text-align: right;
                    }}
                    </style>
                    """, unsafe_allow_html=True)
    
    # ==== Prompt Input Form ====
    with prompt_placeholder:
        st.markdown("**Nhập nội dung cần hỏi:**")
        cols = st.columns((10, 1))
        cols[0].text_input(
            "Chat",
            value="",
            label_visibility="collapsed",
            key="human_prompt"
        )
        cols[1].form_submit_button(
            "Submit",
            type="primary",
            on_click=on_click_callback,
        )
    
    # ==== Footer ====
    
