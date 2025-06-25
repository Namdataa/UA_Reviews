import streamlit as st
from streamlit_option_menu import option_menu
from Streamlit_UI.Chart_app import dashboard
from Streamlit_UI.Chatbot_app import chatbot
def main():
    st.set_page_config(page_title="United Airlines Reviews", page_icon=r"E:\UA_United\united_pageicon.png", layout="wide")
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