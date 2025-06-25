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
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def chatbot():
    # Design layout
    st.markdown("""
        <style>
        /* Nền tổng thể */
        body, .stApp {
            background: #f6f8fc !important;
        }
        /* Header gradient đổi sang xanh nhạt */
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
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #fff !important;
            border-radius: 0 18px 18px 0;
            box-shadow: 2px 0 8px rgba(44,62,80,0.06);
        }
        /* Card, box, filter, metrics, plot, bảng... padding đều */
        .stMetric, .stDataFrame, .stPlotlyChart, .stSelectbox, .stRadio, .stButton, .stTextInput, .stNumberInput, .stSlider, .stMultiSelect {
            background: #fff !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 4px rgba(44,62,80,0.04);
            padding: 18px 28px !important;
            margin-bottom: 18px !important;
        }
        /* Tag màu nhạt */
        span[style*="background"] {
            filter: brightness(1.15);
        }
        /* Chỉnh màu text các subheader, label */
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stSubheader, .stLabel {
            color: #1a237e !important;
        }
        /* Bảng, table */
        table {
            background: #fff !important;
            border-radius: 10px;
            box-shadow: 0 1px 4px rgba(44,62,80,0.04);
        }
        /* Giảm bóng cho toàn bộ */
        .stApp {
            box-shadow: none !important;
        }
        /* Giảm khoảng trống phía trên header */
        section.main, .block-container {
            padding-top: 45px !important;
            margin-top: 0 !important;
        }
        /* Loại bỏ nền trắng rộng hơn ở các container ngoài */
        .block-container {
            background: transparent !important;
            box-shadow: none !important;
        }
        /* Xóa nền không bo góc của các container ngoài */
        .block-container, .stContainer, .element-container {
            background: transparent !important;
            box-shadow: none !important;
        }
        /* Chỉ giữ nền trắng bo góc cho từng card/box nhỏ */
        .stMetric, .stDataFrame, .stPlotlyChart, .stTable {
            background: #fff !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 4px rgba(44,62,80,0.04);
            padding: 18px 28px !important;
            margin-bottom: 18px !important;
        }
        /* Chỉ sửa phần nền của biểu đồ Plotly */
        .stPlotlyChart {
            background: #fff !important;
            border-radius: 18px !important;
            box-shadow: 0 3px 8px rgba(44,62,80,0.12) !important;
            border: 1px solid #e1e5e9 !important;
            padding: 0px 0px !important;
            margin-bottom: 18px !important;
            overflow: visible !important;
            max-width: 100% !important;
        }
        /* Kiểm soát iframe bên trong chart */
        .stPlotlyChart iframe {
            border-radius: 18px !important;
            max-width: 100% !important;
            overflow: visible !important;
        }
        /* Chỉnh sửa sidebar filters */
        .stSelectbox, .stRadio {
            background: #ffffff !important;
            border: 1px solid #e1e5e9 !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
            padding: 12px 16px !important;
            margin-bottom: 8px !important;
        }
        /* Giảm khoảng cách giữa các filter */
        .stSelectbox + .stSelectbox, .stRadio + .stRadio {
            margin-top: 4px !important;
        }
        /* Chỉnh màu nền sidebar rõ ràng hơn */
        section[data-testid="stSidebar"] {
            background: #f8fafc !important;
            border-radius: 0 18px 18px 0;
            box-shadow: 2px 0 8px rgba(44,62,80,0.06);
        }
        </style>
    """, unsafe_allow_html=True)
    
    USER_AVATAR = r"Image/user_icon.png"
    BOT_AVATAR = r"Image/ai_icon.png" 

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
    st.title("🧠 Chat with AI - United Airlines")
    
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
    
