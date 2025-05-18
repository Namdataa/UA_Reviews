import streamlit as st
import os
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import Literal
from core.config import google_api
from Chatbot.ChromaDB import retriever
from Chatbot.main import chain
st.set_page_config(page_title="United Airlines", page_icon="📊", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Chatbot"],
        icons=["bar-chart-line-fill", "robot"],
        menu_icon="cast",
        default_index=0)
    
if selected =="Dashboard":
    None

if selected =="Chatbot":
    USER_AVATAR = r"E:\UA_project\user_icon.png"
    BOT_AVATAR = r"E:\UA_project\ai_icon.png" 

    os.environ["GOOGLE_API_KEY"] = google_api

    @dataclass
    class Message:
        origin: Literal["human", "ai"]
        message: str
    
    # ==== Load CSS ====
    def load_css():
        with open(r"E:\UA_project\styles.css", "r", encoding="utf-8") as f:
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
        
        st.session_state.history.append(Message("ai", result))
        
        # Làm trống ô nhập câu hỏi sau khi submit
        st.session_state.human_prompt = ""
    # ==== App UI ====
    load_css()
    initialize_session_state()
    
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
                st.markdown(f"""
                    <style>
                    .{st.session_state.emotionClass} {{
                        flex-direction: row-reverse;   /* Đảo chiều khung chat */
                        text-align: right;             /* Nội dung lệch phải */
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
    
    # ==== Keyboard Submit Support (ENTER) ====
    components.html("""
<script>
    window.addEventListener('DOMContentLoaded', (event) => {
        // Tìm tất cả các thẻ <img> có alt là "user avatar"
        const userAvatars = document.querySelectorAll('img[alt="user avatar"]');

        userAvatars.forEach(img => {
            // Lấy thẻ <div> cha gần nhất
            const parentDiv = img.closest('div');
            
            if (parentDiv) {
                // Lấy class của div cha
                const parentClass = parentDiv.className;

                // Tìm phần "st-emotion-cache-1c7y2kd" trong class (là phần động mà bạn muốn trích xuất)
                const emotionClass = parentClass.split(' ').find(cls => cls.includes('st-emotion-cache'));

                console.log("🎯 Found avatar:", img);
                console.log("📦 Parent div:", parentDiv);
                console.log("🔑 Class found:", emotionClass);

                // Chuyển dữ liệu từ JavaScript về Python
                window.parent.postMessage({ emotionClass: emotionClass }, "*");
            }
        });
    });
</script>
""", height=0, width=0)
