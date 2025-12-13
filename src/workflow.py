import logging
from typing import Any, Dict, List, Optional

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langsmith import uuid7

from src.prompts import deep_agent_prompt, markdown_style_prompt
from src.schemas import ContextSchema
from src.tools.context_middleware import context_middleware
from src.tools.tools import build_tools

logger = logging.getLogger(__name__)


class Workflow:
    def __init__(
        self,
        agent: Optional[CompiledStateGraph] = None,
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Dict[str, BaseTool]] = None,
        tools_instructions: Optional[str] = "",
        store: Optional[BaseStore] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        subagents: Optional[Dict[str, Any]] = None,
        backend=None,
        middleware: Optional[List[Any]] = None,
    ):
        self.llm = llm if llm else init_chat_model(
            model="gpt-5-mini",
            model_provider="openai",
            temperature=0,
        )
        _tools, _tools_instructions = build_tools()

        self.system_prompt = system_prompt if system_prompt else deep_agent_prompt
        self.tools = tools if tools else _tools

        self.store = store
        self.checkpointer = checkpointer
        self.backend = backend
        self.middleware = middleware
        self.subagents = subagents if subagents else [
            {
                "name": "job_posting_ingestor",
                "description": "Normalize any provided job posting (text or link) into job_posting.md for downstream agents.",
                "system_prompt": f"""You are job_posting_ingestor. Capture the job posting into job_posting.md for others to reuse.

                Inputs: pasted job posting text or a link. If link, use tavily_extract. If text, save as-is.

                Steps:
                1) If a link is present, use tavily_extract to pull posting details. If it fails, note failure and continue with available text.
                2) Save the posting to job_posting.md using write_file. Include source URL if available.
                3) If nothing was provided, write "missing: job_posting" to job_posting.md.

                Constraints: no fabrication; keep raw posting content intact; be concise.
                Output only 'job_posting.md saved'.
                {markdown_style_prompt}
                """,
                "tools": [self.tools.get("tavily_extract")],
                "middleware": [context_middleware],
            },
            {
                "name": "analyze_agent",
                "description": "Analyze resume vs job posting; identify fit, gaps, rewrites with metrics placeholders.",
                "system_prompt": f"""You are analyze_agent. Analyze the candidate resume against the job posting and save analysis.md via write_file.

                Inputs:
                - Resume/context provided.
                - job_posting.md (read via read_file). If missing, state it and continue with best effort.
                - Job posting link or text (use tavily_extract only if needed and not already in job_posting.md).

                Tasks:
                1) Fit summary: candidate role, seniority, core skills, notable achievements (concise bullets).
                2) Gaps: missing/weak requirements vs posting (skills, tools, scope/impact, domain, metrics).
                3) Proof check: where claims lack metrics/context; call out sections needing evidence.
                4) Rewrite suggestions: concrete bullet edits the candidate can paste; action verbs, specific tech, scope, measurable impact; map to posting keywords without exaggeration.
                5) Red flags: mismatches (industry/domain, seniority, leadership vs IC expectations).
                6) Metrics to add: tailored placeholders like [X%], [N users], [time saved].
                7) Keywords to weave in: from the posting.

                analysis.md format:
                - Fit summary (2-3 bullets)
                - Gaps (bullets)
                - Improved bullet suggestions (3-5 bullets)
                - Metrics to add (bullets)
                - Keywords to weave in (bullets)
                - Red flags or risks (bullets, if any)

                Constraints:
                - Be concise and specific; no generic filler.
                - Never fabricate metrics; use placeholders when missing.
                - Ground everything in the provided resume/posting; if info is absent, say so.

                Output only 'analysis.md file saved'.

                {markdown_style_prompt}

                Example analysis.md:
                # Candidate vs Role Analysis

                ## Fit Summary
                - ...
                - ...

                ## Gaps & Mitigations
                - ...
                - ...

                ## Improved Bullet Suggestions (drop-in)
                - ...
                - ...

                ## Metrics to Add
                - ...
                - ...

                ## Keywords to Weave In
                - ...
                - ...

                ## Red Flags / Risks
                - ...
                - ...

                """,
                "tools": [
                    
                    self.tools.get("tavily_extract"),
                ],
                "middleware": [context_middleware],
            },
            {
                "name": "research_agent",
                "description": "Research company/role/industry; return concise bullets with sources and implications.",
                "system_prompt": f"""You are research_agent: a fast, factual researcher for interview prep.

                Tasks:
                - Read job_posting.md (via read_file) if present; otherwise note missing.
                - Use search tools to find current facts (prefer past 12-18 months): company news, products, tech stack, leadership moves, funding, market/competitors, role-specific expectations.
                - Extract 3-6 high-signal bullets that help the candidate tailor answers.
                - For each section, add a short implication for interview prep (e.g., "Emphasize X because company is pushing Y").

                Constraints/Style:
                - Be concise and specific; no fluff. Cite source titles or URLs briefly.
                - If data is unavailable, say so and offer best-effort guidance. Avoid speculation.

                Save to research.md via write_file.

                research.md sections:
                - Company/role insights
                - Interview process hints (if found)
                - Industry/market
                - Suggested focuses
                - Social proof (Reddit/forums) if relevant

                Output only 'research.md file saved'.

                {markdown_style_prompt}
                Example research.md:
                # Company / Role Research — Upstage

                ## Company / Role Insights
                - ...
                - ...

                ## Recent News (sources)
                - ...
                - ...

                ## Role Expectations
                - ...
                - ...

                ## Competitors & Implications
                - ...
                - ...

                ## Suggested Focus for Interviews
                - ...
                - ...

                ## Social Proof
                - ...
                - ...

                """,
                "tools": [

                    self.tools.get("tavily_search"),
                    self.tools.get("tavily_extract"),
                    self.tools.get("reddit_search"),
                ],
                "middleware": [context_middleware],
            },
            {
                "name": "question_writer",
                "description": "Generates a balanced set of 10 interview questions (behavioral + role-specific) with concise, structured example answers.",
                "system_prompt": f"""You are question_writer: write concise, role-specific practice questions with strong example answers.

                Inputs:
                - Role/company context, resume, experience level.
                - job_posting.md (read via read_file); if missing, note and proceed best-effort.

                Tasks:
                1) Produce 10 questions: mix behavioral (ownership, conflict, impact, leadership) and role-specific (projects/tech/architecture/design for eng, product sense for PM, etc.).
                2) Provide a 3-4 sentence example answer for each: brief setup -> concrete actions -> measurable outcome/impact. Name specific tools/tech/processes relevant to the role/company/domain. Use metrics placeholders like [X%].
                3) Save to questions.md via write_file.
                4) Output only 'questions.md file saved'.

                Constraints:
                - Be specific; avoid generic filler. Align terminology with role/company/domain.
                - Do not invent facts; use placeholders when needed.
                - Use tools if research is needed for role/company norms.

                questions.md format:
                # Practice Questions (10)

                - Q1: Design an agent for invoice extraction with exception handling. How ensure factuality and recovery?
                A: [3–4 sentences: architecture, grounding, validation, human-in-loop, outcome with metric placeholder]

                - Q2: How to add VLM to document ingestion? Describe tools/models, eval metrics, and safety.
                A: ...
                {markdown_style_prompt}
                """,
                "middleware": [context_middleware],
                "tools": [
                    
                    self.tools.get("tavily_search"),
                    self.tools.get("tavily_extract"),
                    self.tools.get("reddit_search"),
                ],
            },
            {
                "name": "planner_agent",
                "description": "Builds a concise, ordered prep plan (5–7 steps) with rationale and done criteria.",
                "system_prompt": f"""You are planner_agent: create a focused, ordered prep plan for the upcoming interview using the provided role, company, and candidate context.

                Tasks:
                - Read job_posting.md, analysis.md, research.md (via read_file); if missing, note it.
                - Identify the 5-7 highest-impact prep steps, ordered by urgency/impact.
                - Cover: role-specific gaps, practice areas (behavioral/technical), company/industry research, portfolio/code/artifact updates, logistics (questions to ask, docs to bring).
                - For each step, add a brief rationale, what "done" looks like, and a time-box hint. Highlight dependencies or blockers.
                - Save to prep_plan.md via write_file. Output only 'prep_plan.md file saved'.

                Constraints:
                - No generic fluff. Use role/company/domain terminology.
                - If info is missing, note assumptions.
                - Prefer fewer, higher-leverage steps over long checklists.

                {markdown_style_prompt}

                Example prep_plan.md:

                # Prep Plan — Upstage AI Agent Engineer

                ## Notes
                - Inputs: resume + job_posting.md + analysis/research (if available). Target 5–7 steps.

                ## Steps (ordered)
                1) Refresh resume/portfolio (PDF + demo links)
                2) ...

                """,
                "middleware": [context_middleware],
            },
            {
                "name": "synthesis_agent",
                "description": "Reads subagent markdown files and stitches them into a single final report with light dedupe and no new facts.",
                "system_prompt": f"""You are synthesis_agent: assemble the final report by reading saved markdown files and merging them without inventing new information.

                Mandatory behavior:
                - Use read_file on each: analysis.md, research.md, questions.md, prep_plan.md. If a file is missing, include "missing: <filename>".
                - Lightly deduplicate obvious repeats, but do not add new facts or paraphrase beyond trimming duplicates.
                - Use simple section headers labeling which file the following content came from.

                Output requirements:
                - Build final_response.md (write_file) by pasting contents in order: analysis.md, research.md, questions.md, prep_plan.md.
                - Preserve original formatting; only minimal dedupe allowed.
                - Output only 'final_response.md file saved'.

                Constraints:
                - No new facts, no extra commentary beyond headers and minor dedupe.
                - If nothing was read for a section, include only the "missing" note.

                {markdown_style_prompt}

                Example final_response.md:

                # Final Interview Prep Packet

                ## From analysis.md
                - [Paste fit/gaps/bullets; keep headings]

                ## From research.md
                - [Paste company/role insights, news, competitors, focuses]

                ## From questions.md
                - [Paste Q&A list]

                ## From prep_plan.md
                - [Paste ordered steps + risks]
                """,
            },
        ]
        self.agent = agent if agent else create_deep_agent(
            model=self.llm,
            tools=self.tools.values(),
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            store=self.store,
            subagents=self.subagents,
            middleware=self.middleware,
            backend=self.backend,
        )
        logger.info("Deep agent created with %d tool(s); subagents=%s", len(self.tools or []), bool(self.subagents))

    def _create_config(self) -> Dict[str, Any]:
        return {"configurable": {"thread_id": f"thread_{uuid7()}"}}

    def stream_all(
        self,
        user_input: str,
        context: ContextSchema,
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ):
        """Yield message chunks as the agent streams them."""
        config = self._create_config() if not config else config
        thread_id = config["configurable"].get("thread_id") if isinstance(config, dict) else None
        logger.info(
            "Starting stream",
            extra={"thread_id": thread_id, "role": getattr(context, "role", None)},
        )

        collected_text: List[str] = []
        payload = {"messages": messages} if messages else {"messages": [{"role": "user", "content": user_input}]}

        for message_chunk, metadata in self.agent.stream(payload, stream_mode="messages", config=config, context=context):
            content = message_chunk.content
            if isinstance(content, list):
                content = " ".join(str(c) for c in content if c)
            if content:
                collected_text.append(content)
            yield message_chunk

        final_snippet = (" ".join(collected_text) if collected_text else "")[:500].replace("\n", " ")
        md_files = self._iter_md_files_from_state(config) or self._iter_md_files_from_checkpoint(config)
        if md_files:
            file_names = [md.get("name") for md in md_files if isinstance(md, dict)]
            logger.info(
                "Final markdown files available",
                extra={
                    "thread_id": thread_id,
                    "role": getattr(context, "role", None),
                    "files": file_names,
                },
            )

        logger.info(
            "Stream completed",
            extra={"thread_id": thread_id, "role": getattr(context, "role", None), "snippet": final_snippet},
        )

    def stream_content(self, user_input: str, context: ContextSchema):
        """Yield content (any role) as strings."""
        for chunk in self.stream_all(user_input=user_input, context=context):
            yield chunk.content

    def stream_ai_response(
        self,
        user_input: str,
        context: ContextSchema,
        *,
        include_md_files: bool = True,
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ):
        """Yield only AI message chunks as plain text for streaming."""
        allowed_roles = {"ai", "assistant", "assistant_message"}

        def _flatten_text(content):
            if content is None:
                return []
            if isinstance(content, str):
                return [content]
            if isinstance(content, list):
                pieces = []
                for item in content:
                    if isinstance(item, str):
                        pieces.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        pieces.append(str(item.get("text", "")))
                    elif hasattr(item, "text"):
                        pieces.append(str(getattr(item, "text", "")))
                    else:
                        pieces.append(str(item))
                return pieces
            if hasattr(content, "text"):
                return [str(getattr(content, "text", ""))]
            return [str(content)]

        config = self._create_config() if not config else config

        for chunk in self.stream_all(
            user_input=user_input,
            context=context,
            config=config,
            messages=messages,
        ):
            role = getattr(chunk, "type", getattr(chunk, "role", "unknown"))
            role_l = str(role).lower()
            is_user = any(tag in role_l for tag in ("user", "human"))
            role_ok = role_l in allowed_roles or any(tag in role_l for tag in ("assistant", "ai")) or not is_user
            if role_ok:
                for piece in _flatten_text(getattr(chunk, "content", None)):
                    if piece:
                        yield piece

        if include_md_files:
            md_files = self._iter_md_files_from_state(config) or self._iter_md_files_from_checkpoint(config)
            if md_files:
                file_names = [md.get("name") for md in md_files if isinstance(md, dict)]
                logger.info(
                    "Streaming %d markdown files from state/checkpoint",
                    len(md_files),
                    extra={
                        "thread_id": config.get("configurable", {}).get("thread_id") if isinstance(config, dict) else None,
                        "files": file_names,
                    },
                )
            for md in md_files:
                yield f"\n[Saved file: {md['name']}]"
                yield "\n"
                yield md["text"]

    def list_md_files(self, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Return markdown files captured in state/checkpoint for the given config."""
        if not config:
            config = self._create_config()
        return self._iter_md_files_from_state(config) or self._iter_md_files_from_checkpoint(config)

    def _iter_md_files_from_state(self, config: Optional[Dict[str, Any]]):
        """Collect .md files from the latest graph state (state.files)."""
        if not config or not hasattr(self.agent, "get_state"):
            return []
        try:
            snapshot = self.agent.get_state(config)
        except Exception as exc:
            logger.debug("Failed to load graph state for md files: %s", exc)
            return []
        if snapshot is None:
            return []

        if isinstance(snapshot, dict):
            state_obj = snapshot.get("values") or snapshot.get("state") or snapshot
        else:
            state_obj = getattr(snapshot, "values", None) or getattr(snapshot, "state", None) or snapshot

        return self._extract_md_files_from_obj(state_obj)

    def _iter_md_files_from_checkpoint(self, config: Optional[Dict[str, Any]]):
        """Collect .md files from the latest checkpoint's state.files."""
        if not self.checkpointer or not config:
            return []
        try:
            ckt = self.checkpointer.get_tuple(config)
        except Exception as exc:
            logger.debug("Failed to load checkpoint for md files: %s", exc)
            return []
        if not ckt:
            return []

        checkpoint = getattr(ckt, "checkpoint", None) or {}
        channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}

        return self._extract_md_files_from_obj(channel_values)

    def _extract_md_files_from_obj(self, obj: Any):
        """Walk an object tree to collect .md files under any `files` mapping."""
        results = []
        seen = set()

        def _coerce_text(val):
            if isinstance(val, dict):
                for key in ("text", "content", "$", "page_content", "data"):
                    if key in val and val[key] is not None:
                        return str(val[key])
                return str(val)
            return "" if val is None else str(val)

        def _walk(current):
            if isinstance(current, dict):
                files = current.get("files")
                if isinstance(files, dict):
                    for name, val in files.items():
                        name_str = str(name)
                        if not name_str.endswith(".md") or name_str in seen:
                            continue
                        seen.add(name_str)
                        results.append({"name": name_str, "text": _coerce_text(val)})
                for v in current.values():
                    _walk(v)
            elif isinstance(current, (list, tuple)):
                for v in current:
                    _walk(v)

        _walk(obj)
        return results

    def _stream_response(self, user_input: str, context: ContextSchema) -> None:
        for chunk in self.stream_ai_response(user_input=user_input, context=context):
            print(chunk, end="|", flush=True)

    def execute_agent(self, input: str, context: ContextSchema) -> None:
        """Run the agent once and print streamed AI output."""
        self._stream_response(user_input=input, context=context)
