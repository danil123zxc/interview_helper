import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from workflow import Workflow
from schemas import ContextSchema
from tools import build_tools
load_dotenv()




async def run_workflow(user_input: str, ctx: ContextSchema) -> str:
    llm = init_chat_model(
        model="gpt-5-mini",
        model_provider="openai",
        temperature=0,
    )
    tools, tools_instructions = build_tools()

    subagents = [
        {
            "name": "research_agent",
            "description": "Used to make researches",
            "system_prompt": "You are a great researcher. Use tools to search information. Do not invent anything; check facts using tools.",
            "tools": tools,
        }
    ]

    db_url = os.getenv("DB_URL")

    if db_url:
        async with (
            AsyncPostgresStore.from_conn_string(db_url) as store,
            AsyncPostgresSaver.from_conn_string(db_url) as checkpointer,
        ):
            await store.setup()
            await checkpointer.setup()

            workflow = Workflow(
                llm=llm,
                tools=tools,
                tools_instructions=tools_instructions,
                store=store,
                checkpointer=checkpointer,
                subagents=subagents,
            )
            return await workflow.run_and_collect(
                input=user_input,
                context=ctx,
            )

    # Fallback to in-memory if no DB_URL is provided.
    workflow = Workflow(
        llm=llm,
        tools=tools,
        tools_instructions=tools_instructions,
        store=InMemoryStore(),
        checkpointer=InMemorySaver(),
        subagents=subagents,
    )
    return await workflow.run_and_collect(
        input=user_input,
        context=ctx,
    )


def main():
    st.set_page_config(page_title="Interview Helper", page_icon="🤝")
    st.title("Interview Helper")
    st.caption("Streamlit UI powered by LangGraph + DeepAgents")

    with st.form("agent_form"):
        prompt_text = st.text_area(
            "Your request",
            "Help me prepare for the interview...",
            height=120,
        )
        role = st.text_input("Target role", "AI engineer")
        resume = st.text_area("Resume / background", "", height=200)
        experience_level = st.selectbox(
            "Experience level",
            ["intern", "junior", "mid", "senior", "lead"],
            index=0,
        )
        years_of_experience = st.number_input(
            "Years of experience",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )
        submitted = st.form_submit_button("Run agent")

    if submitted:
        if not prompt_text.strip():
            st.warning("Please enter a request to send to the agent.")
            return

        context = ContextSchema(
            role=role,
            resume=resume,
            experience_level=experience_level,
            years_of_experience=int(years_of_experience),
        )

        st.info("Using Postgres checkpointing" if os.getenv("DB_URL") else "Using in-memory checkpointing")

        with st.spinner("Running agent..."):
            try:
                output = asyncio.run(run_workflow(prompt_text, context))
            except Exception as exc:
                st.error(f"Agent run failed: {exc}")
                return

        st.subheader("Agent response")
        st.markdown(output)


if __name__ == "__main__":
    main()
