import itertools
import logging

import streamlit as st
from dotenv import load_dotenv

switch_page = getattr(st, "switch_page", None)

from src.db import workflow_ctx
from src.logging_config import setup_logging, init_sentry

load_dotenv()

logger = logging.getLogger(__name__)


def _ensure_state():
    """Initialize session state once."""
    st.session_state.setdefault("context_model", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("context_saved", False)
    # list of saved conversations; each entry: {"title": str, "messages": list[dict]}
    st.session_state.setdefault("histories", [])
    st.session_state.setdefault("current_history", 0)


def _reset_session():
    """Clear chat and context."""
    st.session_state.context_model = None
    st.session_state.context_saved = False
    st.session_state.messages = []
    st.session_state.histories = []
    st.session_state.current_history = 0


def main():
    st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")
    _ensure_state()

    # First-time visitors go straight to the User Information page.
    if not st.session_state.context_saved:
        if switch_page:
            switch_page("pages/user_page.py")
        else:
            st.warning("Please fill your info first on the User Information page.")
            st.stop()

    st.title("Interview Helper")
    st.caption("Chat UI powered by LangGraph + DeepAgents")

    # Sidebar: context preview + history controls
    with st.sidebar:
        st.subheader("Session")
        if st.session_state.context_saved and st.session_state.context_model:
            ctx = st.session_state.context_model
            st.markdown(
                f"**Role:** {ctx.role}\n\n"
                f"**Experience:** {ctx.experience_level}, {ctx.years_of_experience or 0} yrs\n\n"
                f"**Resume preview:** {ctx.resume[:50]}{'...' if ctx.resume and len(ctx.resume) > 200 else ''}"
            )
        else:
            st.warning("No context submitted yet.")
            if st.button("Fill user info"):
                switch_page("pages/user_page.py")

        if st.button("Reset session"):
            _reset_session()
            st.rerun()

        st.subheader("Chat history")
        if not st.session_state.histories:
            st.session_state.histories = [{"title": "Chat 1", "messages": list(st.session_state.messages)}]
        for idx, hist in enumerate(st.session_state.histories):
            label = hist["title"]
            if st.button(label, key=f"hist_{idx}"):
                st.session_state.current_history = idx
                st.session_state.messages = list(hist["messages"])
                st.rerun()

    # Chat view
    st.markdown("### Chat")
    chat_disabled = not st.session_state.context_saved

    # Render current conversation
    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about interview prep...", disabled=chat_disabled)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_box = st.chat_message("assistant")
        tool_placeholder = assistant_box.empty()
        error_placeholder = assistant_box.empty()

        def response_stream():
            """Stream agent output while showing a simple tool animation."""
            resp_parts = []
            spinner_cycle = itertools.cycle(
                ["🔍 Searching & planning", "📖 Reading context", "✅ Updating todos", "🤝 Coordinating subagents"]
            )
            try:
                with workflow_ctx() as wf:
                    for chunk in wf.stream_ai_response(
                        user_input=prompt,
                        context=st.session_state.context_model,
                    ):
                        tool_placeholder.info(next(spinner_cycle))
                        text = (
                            chunk
                            if isinstance(chunk, str)
                            else "".join(chunk) if isinstance(chunk, list) else str(chunk)
                        )
                        resp_parts.append(text)
                        yield text
                tool_placeholder.empty()
                if resp_parts:
                    st.session_state.messages.append({"role": "assistant", "content": "".join(resp_parts)})
                else:
                    msg = "No response from agent."
                    error_placeholder.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as exc:
                tool_placeholder.empty()
                error_placeholder.error(f"Agent error: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"Agent error: {exc}"})

        assistant_box.write_stream(response_stream())

        # Persist current conversation into histories
        current = st.session_state.current_history
        if current >= len(st.session_state.histories):
            st.session_state.histories.append({"title": f"Chat {len(st.session_state.histories)+1}", "messages": []})
        st.session_state.histories[current] = {
            "title": f"Chat {current+1}",
            "messages": list(st.session_state.messages),
        }


if __name__ == "__main__":
    main()
