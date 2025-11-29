import streamlit as st

switch_page = getattr(st, "switch_page", None)

from src.schemas import ContextSchema

from pypdf import PdfReader


def extract_pdf_text(uploaded_file, max_chars=8000):
    reader = PdfReader(uploaded_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars] 


def _ensure_state():
    """Initialize session state keys used across pages."""
    st.session_state.setdefault("context_model", None)
    st.session_state.setdefault("context_saved", False)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("histories", [])
    st.session_state.setdefault("current_history", 0)


def main():
    st.set_page_config(page_title="User Information", page_icon="🧑‍💻", layout="wide")
    _ensure_state()

    st.title("Your Interview Profile")
    st.caption("Fill this once; the Chat page will automatically use it.")

    levels = ["intern", "junior", "mid", "senior", "lead"]
    level_default = 0
    if st.session_state.context_model:
        try:
            level_default = levels.index(st.session_state.context_model.experience_level)
        except ValueError:
            level_default = 0

    with st.form("context_form", clear_on_submit=False):
        role = st.text_input(
            "Target role",
            st.session_state.context_model.role if st.session_state.context_model else "AI engineer",
        )
        
        resume = st.text_area(
            "Resume / background",
            st.session_state.context_model.resume if st.session_state.context_model else "",
            height=200,
        )
        uploaded_file = st.file_uploader("Upload files")
        experience_level = st.selectbox(
            "Experience level",
            levels,
            index=level_default,
        )
        years_of_experience = st.number_input(
            "Years of experience",
            min_value=0,
            max_value=50,
            value=st.session_state.context_model.years_of_experience if st.session_state.context_model else 0,
            step=1,
        )
        submitted = st.form_submit_button("Save")

    if submitted:
        resume_file = extract_pdf_text(uploaded_file)
        
        st.session_state.context_model = ContextSchema(
            role=role,
            resume=f"Resume:\n{resume}\n{resume_file}",
            experience_level=experience_level,
            years_of_experience=years_of_experience,
        )
        st.session_state.context_saved = True
        
        st.success("Saved. Head to Chat to start the conversation.")

    if st.session_state.context_saved:
        if switch_page:
            switch_page("pages/chat_page.py")
        else:
            st.page_link("streamlit_app.py", label="Chat")


if __name__ == "__main__":
    main()
