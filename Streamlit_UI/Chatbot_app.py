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
from Streamlit_UI.request_data import get_base64_logo
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def chatbot():
    # Design layout
    st.markdown("""
    <style>
    /* Nền tổng thể */
    body, .stApp {
        background: #f6f8fc !important; /* Màu nền tổng thể như ảnh bạn gửi (xám nhạt) */
    }

    /* Header gradient (giữ lại theo ảnh bạn gửi) */
    .header-gradient {
        background: linear-gradient(90deg, #e3f0ff 0%, #f6eaff 100%);
        border-radius: 16px;
        padding: 24px 0 18px 0;
        margin-bottom: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(44,62,80,0.08);
    }
    .header-title {
        color: #1a237e;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2em;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
    }
    .header-desc {
        color: #374151;
        font-size: 1.25rem;
        font-weight: 400;
        margin-top: 0.2em;
    }
    .logo-img {
        height: 64px;
        width: 64px;
        object-fit: contain;
        margin-right: 8px;
        vertical-align: middle;
    }

    /* Ô nhập liệu (Input) - Cụ thể cho st.text_area hoặc st.text_input */
    /* Target the container of text input/text area */
    div.stTextArea > div, div.stTextInput > div {
        background-color: #fff !important; /* Nền trắng cho ô nhập liệu */
        border-radius: 12px !important; /* Bo góc như ảnh */
        box-shadow: 0 1px 4px rgba(44,62,80,0.04); /* Đổ bóng nhẹ */
        padding: 18px 28px !important; /* Padding bên trong ô nhập liệu */
        margin-bottom: 18px !important;
        border: none !important; /* Loại bỏ viền mặc định nếu có */
    }
    /* Đảm bảo khung nhập liệu bên trong vẫn là màu trắng */
    .stTextInput input, .stTextArea textarea {
        background-color: #fff !important;
        color: #374151 !important; /* Màu chữ trong ô nhập liệu */
    }

    /* Giảm khoảng trống phía trên header */
    section.main, .block-container {
        padding-top: 45px !important;
        margin-top: 0 !important;
    }
    /* Loại bỏ nền trắng rộng hơn ở các container ngoài (giữ nguyên để đúng bố cục ảnh) */
    .block-container {
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Xóa nền không bo góc của các container ngoài (giữ nguyên) */
    .block-container, .stContainer, .element-container {
        background: transparent !important;
        box-shadow: none !important;
    }

    </style>
""", unsafe_allow_html=True)
    
    USER_AVATAR = r"Image/user_icon.png"
    BOT_AVATAR = r"Image/ai_icon.png" 
    
    logo_b64 = get_base64_logo(r"Image/united_logo.png")
    
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
    st.markdown(f"""
        <div class="header-gradient">
            <div class="header-title">
                <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="Logo" />
                Chat with AI - United Airlines
            </div>
            <div class="header-desc">
               🧠 To help you learn more about United Airlines
            </div>
        </div>
    """, unsafe_allow_html=True)
    
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
        st.markdown("**Ask Me:**")
        cols = st.columns((9.5, 1.5))
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
    
