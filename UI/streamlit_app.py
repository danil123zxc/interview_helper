import streamlit as st

pages = {
    "Interview Helper": [
        st.Page("pages/user_page.py", title="User info"),
        st.Page("pages/chat_page.py", title="Chat"),
    ],
}

pg = st.navigation(pages)
pg.run()
