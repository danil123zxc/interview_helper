import streamlit as st
from src.logging_config import setup_logging, init_sentry

setup_logging()
init_sentry()

pages = {
    "Interview Helper": [
        st.Page("pages/user_page.py", title="User info"),
        st.Page("pages/chat_page.py", title="Chat"),
    ],
}

pg = st.navigation(pages)
pg.run()
