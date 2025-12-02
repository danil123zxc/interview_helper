
import logging

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from typing import List, Optional, Any, Union, Dict
from langchain.chat_models.base import BaseChatModel
from src.prompts import deep_agent_prompt

from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langsmith import uuid7
from src.schemas import ContextSchema
from langgraph.graph.state import CompiledStateGraph
from src.tools.tools import build_tools
from deepagents.backends import StoreBackend
from src.tools.context_middleware import context_middleware

logger = logging.getLogger(__name__)

class Workflow:
    def __init__(self, 
                 agent: Optional[CompiledStateGraph] = None,
                 llm: Optional[BaseChatModel]=None, 
                 system_prompt: Optional[str]=None, 
                 tools: Optional[Dict[str, BaseTool]]=None,
                 tools_instructions: Optional[str]="",
                 store: Optional[BaseStore] = None, 
                 checkpointer: Optional[BaseCheckpointSaver] = None,
                 subagents: Optional[Dict[str, Any]] = None,
                 backend = None,
                 middleware: Optional[List[Any]] = None,
                 ):
      
        self.llm = llm if llm else init_chat_model(
            model="gpt-5-mini",
            model_provider="openai",
            temperature=0,
        )
        _tools, _tools_instructions = build_tools()

        self.system_prompt = system_prompt if system_prompt else deep_agent_prompt.format(
            tools_instructions=tools_instructions if tools_instructions else _tools_instructions
        )
        self.tools = tools if tools else _tools
        self.configs: List[Dict[str, Any]] = []
        self.store = store
        self.checkpointer = checkpointer
        self.backends = backend if backend else lambda x: StoreBackend(x) 
        self.middleware = middleware if middleware else None
        self.subagents = subagents if subagents else [
            {
                "name": "analyze_agent",
                "description": "Takes user's resume and a job posting(can be link or text), analyzes and gives improvement suggestions.",
                "system_prompt": """You task is to analyze a resume and a job posting and write a report.

                Inputs you receive:
                - Resume: free-text, possibly extracted from PDF.
                - Job posting: text or link(use tavily_extract if link is given).

                Your tasks:
                1) Fit analysis: Summarize the candidate’s role, seniority, core skills, and notable achievements as stated.
                2) Gap check: Identify missing or weak requirements vs the posting (skills, tools, scope/impact, domain, metrics).
                3) Proof and evidence: Note where claims lack quantifiable proof or context; highlight sections needing metrics or outcomes.
                4) Rewrite suggestions: Provide concrete bullet edits the candidate can paste into their resume:
                - Use strong action verbs, specific tech, scope (team/scale), and measurable impact (%, $, time).
                - Map phrasing to the job’s keywords/requirements without exaggerating.
                5) Red flags: Call out any mismatches (industry/domain, seniority, leadership vs IC expectations).

                Output format:
                - Fit summary (2–4 bullets).
                - Gaps (bullets).
                - Improved bullet suggestions (3–8 bullets) referencing the job’s language.
                - Metrics to add (examples tailored to the candidate’s work).
                - Keywords to weave in (from the job posting).
                - Red flags or risks (if any).
                - Save it to analysis.md file (use `write_file` tool)
              
                Constraints:
                - Be concise and specific; avoid generic advice.
                - Never fabricate metrics; propose placeholders like “[X%]”, “[N users]”, “[time saved]” when missing.
                - Always ground suggestions in the given resume content; if something is absent, call it out as missing rather than inventing it.
                
                """,
                "tools":[self.tools.get('tavily_extract')],
                "middleware": [context_middleware],
            },
            {
                "name": "research_agent",
                "description": "Used to make researches. Use him when you need to research about the company, role, industry trends, recent news, competitors, products, tech stack, leadership, funding etc.",
                "system_prompt": f"""You are research_agent: a fast, factual researcher who pulls recent, relevant information about the company, role, and industry to support interview prep. 
                Tasks

                - Use search tools to find current facts: company news, products, tech stack, leadership moves, funding, market/competitors, role-specific expectations.
                - Extract 3–7 high-signal bullets that help the candidate tailor their answers.
                - Call out implications for the interview (e.g., “Emphasize X because company is pushing Y”).
                
                Constraints/Style

                - Be concise and specific; no fluff.
                - Cite source titles or URLs briefly when possible.
                - If data is unavailable, say so and offer best-effort guidance.
                - Avoid speculation; prefer verifiable facts.
                Output format

                - Company/role insights: 3–5 bullets (fact + implication).
                - Industry/market: 2–3 bullets (trend + why it matters for this role).
                - Suggested focuses: 3 bullets on what to emphasize in answers.
                Tooling guidance
                - Save it to research.md file(use `write_file` tool)

                Prefer tavily_search for discovery, tavily_extract for details; add Reddit only if social proof is requested or helpful

                Do not invent anything; check facts using tools.""",
                "tools": [self.tools.get('tavily_search'), self.tools.get('tavily_extract'), self.tools.get('reddit_search')],
                
            },
            {
                "name": "question_writer",
                "description":"Generates a balanced set of 10 interview questions (behavioral + role-specific) with concise, structured example answers.",
                "system_prompt":"""You are question_writer: an interview prep specialist who writes concise, role-specific practice questions with strong example answers.

                Inputs you’ll receive:
                - Role/company and any context (resume, experience level, job posting highlights).
                - Key skills/tech/competencies to cover.

                Your tasks:
                1) Produce a balanced set of 10 questions: mix behavioral (ownership, conflict, impact, leadership), technical/role-specific, and role craft (architecture/design for eng, product sense for PM, etc.).
                2) Provide tight example answers for each question that model structure and depth:
                - Structure: brief setup → concrete actions → measurable outcome/impact.
                - Name specific tools/tech/processes relevant to the role/company/domain.
                - Keep answers concise (3–6 sentences).
                3) Save it to questions.md file (use `write_file` tool)

                Style and constraints:
                - Be specific, avoid generic filler.
                - Align terminology with the given role/company/domain.
                - Include metrics/placeholders when the user hasn’t provided them (e.g., “[reduced latency by X%]”).
                - Don’t invent facts about the user; offer plausible phrasing with placeholders instead.

                Output format:
                - Q1: …
                A: …
                - Q2: …
                A: …
                ...
                """,
                "middleware": [context_middleware],
            },
            {
                "name": "planner_agent",
                "description":""" Builds a concise, ordered prep plan (5–8 high-impact steps) for the interview based on role, company, 
                and candidate context—covering practice focus, research, artifacts, and logistics—with rationales, “done” criteria, and notes on dependencies or blockers.""",
                "system_prompt":"""
                You are planner_agent: create a focused, ordered prep plan for the upcoming interview using the provided role, company, and candidate context (resume, experience level).

                Tasks:
                - Identify the 5–8 highest-impact prep steps, ordered by urgency/impact.
                - Cover: role-specific knowledge gaps, practice areas (behavioral/technical), company/industry research, portfolio/code/artefact updates, and logistics (questions to ask, docs to bring).
                - For each step, add a brief rationale and what “done” looks like; include time-box suggestions when helpful.
                - Highlight critical dependencies or blockers.
                - Keep it concise and actionable; no generic fluff.
                - Save it to prep_plan.md file (use `write_file` tool)

                Constraints:
                - Do not invent user-specific facts; if missing info, note assumptions.
                - Use role/company/domain terminology.
                - Prefer fewer, higher-leverage steps over long checklists.
                """,
                "middleware": [context_middleware],
            },  
            {   
                "name": "synthesis_agent",
                "description": """SynthesisAgent merges subagents’ outputs into one concise, deduped response aligned to the user’s role/company, 
                preserving structure (snapshot, prep plan, Q&A, insights, gap analysis, checklist) and avoiding new facts or fluff.""",
                "system_prompt": """You are synthesis_agent: merge outputs from all subagents into one concise, coherent answer for the user.
                
                Use `read_file` tool to read the subagent outputs(.md files).

                Inputs you receive:
                - research.md file insights (company/role/industry facts).
                - analysis.md file gaps and resume rewrites.
                - questions.md file practice Q&A.
                - prep_plan.md file prep plan/checklist.
                - User context (role, resume, experience level, job posting highlights).
               
                Your tasks:
                1) Deduplicate and prioritize: keep the highest-signal points, remove repeats.
                2) Align to the user’s role/company/domain; keep terminology consistent.
                3) Keep output lean and scannable.
                4) Save it to final_response.md file (use `write_file` tool)

                Output structure (adapt if something is missing):
                - Snapshot: role/company/user profile + top 3 risks/opportunities.
                - Prep plan: 5–8 bullets (from planner_agent).
                - Practice Q&A: list the 10 questions with concise example answers (from question_writer).
                - Company/industry insights: 3–5 bullets (from research_agent).
                - Gap analysis + suggested resume bullets: strengths, gaps, and 3–8 rewritten bullets with metrics placeholders (from analyze_agent).
                - Closing checklist: what to rehearse/research/prepare next.

                Constraints:
                - No new research or fabrication; only use provided inputs and user context.
                - Be concise; prefer bullets; avoid fluffy language.
                - If a section is missing inputs, note it briefly and move on.

                """,
                
            }
        ]
        self.agent = agent if agent else create_deep_agent(
            model=self.llm,
            tools=self.tools.values(),
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            store=self.store,
            subagents=self.subagents,
            middleware=self.middleware,
            backend=self.backends,
        )
        logger.info("Deep agent created with %d tool(s); subagents=%s", len(self.tools or []), bool(self.subagents))

    def _create_config(self) -> Dict[str, Any]:
        config = {
                "configurable": {
                    "thread_id": f"thread_{uuid7()}"
                    }
            }
        
        self.configs.append(config)

        return config
    

    def stream_all(
            self,
            user_input: str,
            context: ContextSchema,
            config: Optional[Dict[str, Any]] = None
        ):
        """Yield message chunks as the agent streams them."""
        config = self._create_config() if not config else config
        thread_id = config["configurable"].get("thread_id") if isinstance(config, dict) else None
        logger.info(
            "Starting stream",
            extra={"thread_id": thread_id, "role": getattr(context, "role", None)},
        )

        for message_chunk, metadata in self.agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ]
            },
            stream_mode="messages",
            config=config,
            context=context
        ):
            role = getattr(message_chunk, "type", getattr(message_chunk, "role", "unknown"))
            content = message_chunk.content
            if isinstance(content, list):
                content = " ".join(str(c) for c in content if c)
            snippet = (content or "")[:500].replace("\n", " ")
            logger.debug("Chunk role:\n%s\n Content:\n%s", role, snippet)
            yield message_chunk
        logger.info("Stream completed", extra={"thread_id": thread_id})

    def stream_content(
                self,
                user_input: str,
                context: ContextSchema
            ):
        """Yield content (any role) as strings."""
        for chunk in self.stream_all(
            user_input=user_input,
            context=context,
        ):
            yield chunk.content


    def stream_ai_response(
                self,
                user_input: str,
                context: ContextSchema,
                *,
                include_md_files: bool = True,
                config: Optional[Dict[str, Any]] = None,
            ):
        """Yield only AI message chunks as plain text for streaming.

        If include_md_files is True, any .md files found in the graph state
        (state.files) are streamed after the model output. Falls back to
        checkpoint lookup when available.
        """
        allowed_roles = {"ai", "assistant", "assistant_message"}

        def _flatten_text(content):
            if content is None:
                return []
            # Already a string
            if isinstance(content, str):
                return [content]
            # List of segments
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
            # Objects with .text
            if hasattr(content, "text"):
                return [str(getattr(content, "text", ""))]
            return [str(content)]
        
        config = self._create_config() if not config else config
        
        for chunk in self.stream_all(
            user_input=user_input,
            context=context,
            config=config,
        ):
            role = getattr(chunk, "type", getattr(chunk, "role", "unknown"))
            role_l = str(role).lower()
            # Accept common assistant roles, any role containing "assistant"/"ai",
            # and any non-user/human role with content (to avoid silently dropping output).
            is_user = any(tag in role_l for tag in ("user", "human"))
            role_ok = role_l in allowed_roles or any(tag in role_l for tag in ("assistant", "ai")) or not is_user
            if role_ok:
                for piece in _flatten_text(getattr(chunk, "content", None)):
                    if piece:
                        yield piece

        if include_md_files:
            md_files = self._iter_md_files_from_state(config) or self._iter_md_files_from_checkpoint(config)
            if md_files:
                logger.info(
                    "Streaming %d markdown files from state/checkpoint",
                    len(md_files),
                    extra={"thread_id": config.get("configurable", {}).get("thread_id") if isinstance(config, dict) else None},
                )
            for md in md_files:
                header = f"\n[Saved file: {md['name']}]"
                yield header
                yield "\n"
                yield md["text"]

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

        state_obj = None
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

    def _stream_response(
            self,
            user_input: str, 
            context: ContextSchema
            ) -> None:
        for chunk in self.stream_ai_response(
            user_input=user_input,
            context=context,
        ):
            print(chunk, end='|', flush=True)

    def execute_agent(
            self,
            input: str,
            context: ContextSchema,
        ) -> None:
        """Run the agent once and print streamed AI output."""
        self._stream_response(
            user_input=input,
            context=context,
        )
