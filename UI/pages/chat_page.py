import itertools
import logging
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

switch_page = getattr(st, "switch_page", None)

from src.db import workflow_ctx
from src.logging_config import setup_logging, init_sentry, logging_context

load_dotenv()

logger = logging.getLogger(__name__)


def _ensure_state():
    """Initialize session state once."""
    st.session_state.setdefault("context_model", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("context_saved", False)
    # list of saved conversations; each entry: {"title": str, "messages": list[dict], "config": dict}
    st.session_state.setdefault("histories", [])
    st.session_state.setdefault("current_history", 0)
    if not st.session_state.histories:
        st.session_state.histories = [
            {
                "title": "Chat 1",
                "messages": list(st.session_state.messages),
                "config": {"configurable": {"thread_id": f"thread_{uuid.uuid4()}"}},
                "files": [],
            }
        ]
    for hist in st.session_state.histories:
        hist.setdefault("files", [])


def _reset_session():
    """Clear chat and context."""
    st.session_state.context_model = None
    st.session_state.context_saved = False
    st.session_state.messages = []
    st.session_state.histories = []
    st.session_state.current_history = 0


def _new_chat():
    """Start a fresh chat with its own thread/config."""
    next_idx = len(st.session_state.histories)
    st.session_state.histories.append(
        {
            "title": f"Chat {next_idx + 1}",
            "messages": [],
            "config": {"configurable": {"thread_id": f"thread_{uuid.uuid4()}"}},
            "files": [],
        }
    )
    st.session_state.current_history = next_idx
    st.session_state.messages = []


def main():
    setup_logging()
    init_sentry()
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
        if st.button("New chat", use_container_width=True):
            _new_chat()
            st.rerun()
        for idx, hist in enumerate(st.session_state.histories):
            label = hist["title"]
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.current_history = idx
                st.session_state.messages = list(hist["messages"])
                hist.setdefault("files", [])
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

        current = st.session_state.current_history
        current_hist = st.session_state.histories[current]
        if not current_hist.get("config"):
            current_hist["config"] = {"configurable": {"thread_id": f"thread_{uuid.uuid4()}"}}
        current_config = current_hist["config"]
        conversation_messages = list(st.session_state.messages)

        assistant_box = st.chat_message("assistant")
        tool_placeholder = assistant_box.empty()
        error_placeholder = assistant_box.empty()
        md_files = []

        def response_stream():
            """Stream agent output while showing a simple tool animation."""
            nonlocal md_files
            resp_parts = []
            spinner_cycle = itertools.cycle(
                [
                    "Searching & planning",
                    "Reading context",
                    "Updating todos",
                    "Coordinating subagents",
                ]
            )
            try:
                thread_id = current_config.get("configurable", {}).get("thread_id")
                start = time.monotonic()
                with logging_context(thread_id=thread_id):
                    with workflow_ctx() as wf:
                        for chunk in wf.stream_ai_response(
                            user_input=prompt,
                            context=st.session_state.context_model,
                            messages=conversation_messages,
                            config=current_config,
                            include_md_files=False,
                        ):
                            tool_placeholder.info(next(spinner_cycle))
                            text = (
                                chunk
                                if isinstance(chunk, str)
                                else "".join(chunk) if isinstance(chunk, list) else str(chunk)
                            )
                            resp_parts.append(text)
                            yield text

                        md_files = wf.list_md_files(config=current_config)
                        history = wf.agent.get_state_history(current_config)
                        if history:
                            history_clean = " ".join(str(history).split())
                            logger.debug(
                                "State history (truncated): %s",
                                history_clean[:800] + ("..." if len(history_clean) > 800 else ""),
                            )

                elapsed = time.monotonic() - start
                logger.info("Chat response completed in %.2fs", elapsed)

                tool_placeholder.empty()
                if resp_parts:
                    st.session_state.messages.append({"role": "assistant", "content": "".join(resp_parts)})
                else:
                    msg = "No response from agent."
                    error_placeholder.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as exc:
                tool_placeholder.empty()
                logger.exception("Agent error")
                error_placeholder.error(f"Agent error: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"Agent error: {exc}"})

        assistant_box.write_stream(response_stream())
        

        # Persist current conversation into histories
        current = st.session_state.current_history
        if current >= len(st.session_state.histories):
            st.session_state.histories.append(
                {
                    "title": f"Chat {len(st.session_state.histories)+1}",
                    "messages": [],
                    "config": current_config,
                    "files": [],
                }
            )
        st.session_state.histories[current] = {
            "title": f"Chat {current+1}",
            "messages": list(st.session_state.messages),
            "config": current_config,
            "files": md_files or st.session_state.histories[current].get("files", []),
        }

    # Saved markdown files panel
    current_hist = st.session_state.histories[st.session_state.current_history]
    current_files = current_hist.get("files", [])
    st.markdown("### Saved markdown files")
    if current_files:
        for file_idx, file_data in enumerate(current_files):
            file_name = file_data.get("name", f"file_{file_idx}.md")
            file_text = file_data.get("text", "")
            row = st.container()
            cols = row.columns([0.7, 0.3])
            cols[0].markdown(f"📄 **{file_name}**")
            with cols[0].expander("Preview", expanded=False):
                st.code(file_text or "(empty file)", language="markdown")
            cols[1].download_button(
                "Download",
                data=file_text,
                file_name=file_name,
                mime="text/markdown",
                key=f"dl_{st.session_state.current_history}_{file_idx}",
            )
    else:
        st.info("Markdown files generated by the agent will appear here with preview/download options.")


if __name__ == "__main__":
    main()
