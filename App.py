import streamlit as st
from streamlit_option_menu import option_menu
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="United Airlines Reviews", page_icon=r"Image/united_pageicon.png", layout="wide") # Yêu cầu chạy đầu tiên

from Streamlit_UI.Chart_app import dashboard # Đã có st. nên phải chạy sau
from Streamlit_UI.Chatbot_app import chatbot # Đã có st. nên phải chạy sau
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Dashboard", "Chatbot"],
            icons=["bar-chart-line-fill", "robot"],
            menu_icon="cast",
            default_index=0)
    if selected =="Dashboard":
        dashboard()
    if selected =="Chatbot":
        chatbot()

if __name__ == "__main__":
    main()
