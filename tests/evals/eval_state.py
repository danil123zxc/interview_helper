"""Final-state parsing helpers for evals."""

import logging
from typing import Any, Dict, List, Optional

from src.schemas import ContextSchema
from tests.evals.eval_constants import SUBAGENT_FILES, SUBAGENT_PROMPTS, USER_INPUT_KEYS

logger = logging.getLogger(__name__)


def extract_user_input(row: Dict[str, Any]) -> Optional[str]:
    """Return the first user input value found in a dataset row.

    Args:
        row: Dataset row dict that may include user input under keys in USER_INPUT_KEYS.

    Returns:
        The first non-empty user input string, or None if nothing is found.

    Example:
        ```python
        row = {"prompt": "Help me prepare for an interview"}
        value = extract_user_input(row)
        # value == "Help me prepare for an interview"
        ```
    """
    for key in USER_INPUT_KEYS:
        val = row.get(key)
        if val:
            return str(val)
    return None


def build_context(row: Dict[str, Any], defaults: Dict[str, Any]) -> ContextSchema:
    """Build a ContextSchema from dataset values with fallbacks.

    Args:
        row: Dataset row dict with optional "context" dict and top-level fields.
        defaults: Default values for role, resume, experience_level, years_of_experience.

    Returns:
        ContextSchema with merged values; role falls back to "unknown" if missing.

    Example:
        ```python
        row = {
            "context": {"role": "Backend Engineer", "resume": "..."},
            "experience_level": "mid",
        }
        defaults = {
            "role": None,
            "resume": None,
            "experience_level": "intern",
            "years_of_experience": 1,
        }
        ctx = build_context(row, defaults)
        # ctx.model_dump() == {
        #   "role": "Backend Engineer",
        #   "resume": "...",
        #   "experience_level": "mid",
        #   "years_of_experience": 1,
        # }
        ```
    """
    context = row.get("context") if isinstance(row.get("context"), dict) else {}
    role = context.get("role") or row.get("role") or defaults.get("role")
    resume = context.get("resume") or row.get("resume") or defaults.get("resume")
    experience_level = (
        context.get("experience_level")
        or row.get("experience_level")
        or defaults.get("experience_level")
        or "intern"
    )
    years_of_experience = (
        context.get("years_of_experience")
        or row.get("years_of_experience")
        or defaults.get("years_of_experience")
    )
    if not role:
        logger.warning("Missing role in dataset row; using 'unknown'.")
        role = "unknown"
    return ContextSchema(
        role=role,
        resume=resume,
        experience_level=experience_level,
        years_of_experience=years_of_experience,
    )


def md_files_to_map(md_files: List[Dict[str, str]]) -> Dict[str, str]:
    """Convert list of markdown file dicts into a name-to-text mapping.

    Args:
        md_files: List of dicts that include "name" and "text" keys.

    Returns:
        Mapping from file name to markdown text.

    Example:
        ```python
        md_files = [{"name": "analysis.md", "text": "Analysis text"}]
        md_map = md_files_to_map(md_files)
        # md_map == {"analysis.md": "Analysis text"}
        ```
    """
    md_map: Dict[str, str] = {}
    for item in md_files:
        name = item.get("name")
        text = item.get("text", "")
        if name:
            md_map[str(name)] = str(text)
    return md_map


def build_subagent_items(
    md_map: Dict[str, str],
    history_by_subagent: Dict[str, str],
) -> List[Dict[str, str]]:
    """Build per-subagent input/output pairs for evaluation.

    Args:
        md_map: Mapping of markdown file names to their content.
        history_by_subagent: Mapping of subagent name to execution history text.

    Returns:
        List of dicts with keys: subagent_name, input, output.

    Example:
        ```python
        md_map = {"analysis.md": "Analysis text"}
        history = {"analyze_agent": "Did step A"}
        items = build_subagent_items(md_map, history)
        item = next(i for i in items if i["subagent_name"] == "analyze_agent")
        # item["input"] == "Did step A"
        # item["output"].startswith("# analysis.md")
        ```
    """
    items: List[Dict[str, str]] = []
    for subagent_name in SUBAGENT_PROMPTS.keys():
        file_names = SUBAGENT_FILES.get(subagent_name, [])
        input_text = (
            history_by_subagent.get(subagent_name)
            or history_by_subagent.get(subagent_name.replace("-", "_"))
            or "No subagent history found."
        )
        outputs: List[str] = []
        for file_name in file_names:
            text = md_map.get(file_name)
            if text:
                outputs.append(f"# {file_name}\n{text}".strip())
        if not outputs and file_names:
            outputs.append("missing: " + ", ".join(file_names))
        output_text = "\n\n".join(outputs).strip()
        items.append(
            {
                "subagent_name": subagent_name,
                "input": input_text,
                "output": output_text,
            }
        )
    return items


def extract_state_obj(state: Any) -> Any:
    """Normalize a final state snapshot into a plain object/dict when possible.

    Args:
        state: Final state snapshot, dict, or object with "values"/"state" attributes.

    Returns:
        The inner values/state if present, otherwise the original object.

    Example:
        ```python
        state = {"values": {"files": {"analysis.md": "..."}}}
        obj = extract_state_obj(state)
        # obj == {"files": {"analysis.md": "..."}}
        ```
    """
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get("values") or state.get("state") or state
    return getattr(state, "values", None) or getattr(state, "state", None) or state


def collect_messages(obj: Any) -> List[Any]:
    """Collect all message lists under any "messages" key in an object tree.

    Args:
        obj: Arbitrary object tree containing nested dicts/lists.

    Returns:
        Flat list of message objects found in the tree.

    Example:
        ```python
        obj = {"messages": [{"role": "user", "content": "hi"}], "other": {"messages": []}}
        msgs = collect_messages(obj)
        # len(msgs) == 1
        # msgs[0]["content"] == "hi"
        ```
    """
    messages: List[Any] = []

    def _walk(current: Any) -> None:
        if isinstance(current, dict):
            msgs = current.get("messages")
            if isinstance(msgs, list):
                messages.extend(msgs)
            for value in current.values():
                _walk(value)
        elif isinstance(current, (list, tuple)):
            for value in current:
                _walk(value)

    _walk(obj)
    return messages


def extract_subagent_history(state_obj: Any, history: Any) -> Dict[str, str]:
    """Extract per-subagent execution history from final state or state history.

    Args:
        state_obj: Normalized state object (values/state) from extract_state_obj.
        history: Optional state history snapshot used as a fallback source.

    Returns:
        Mapping of subagent name to a newline-joined history string.

    Example:
        ```python
        state_obj = {
            "messages": [
                {
                    "tool_calls": [
                        {
                            "name": "task",
                            "id": "1",
                            "args": {"subagent_type": "analyze_agent", "description": "Analyze"},
                        }
                    ]
                },
                {"role": "tool", "tool_call_id": "1", "content": "analysis.md saved"},
            ]
        }
        hist = extract_subagent_history(state_obj, history=None)
        # hist["analyze_agent"] == "[task] Analyze\n[result] analysis.md saved"
        ```
    """
    histories: Dict[str, List[str]] = {name: [] for name in SUBAGENT_PROMPTS.keys()}
    tool_call_to_subagent: Dict[str, str] = {}

    def _msg_get(msg: Any, key: str) -> Any:
        if isinstance(msg, dict):
            return msg.get(key)
        return getattr(msg, key, None)

    messages = collect_messages(state_obj)
    if not messages and history is not None:
        messages = collect_messages(history)

    for msg in messages:
        tool_calls = _msg_get(msg, "tool_calls") or []
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_name = call.get("name") or call.get("tool")
            args = call.get("args") or {}
            call_id = call.get("id")
            if call_name == "task":
                subagent = (
                    args.get("subagent_type")
                    or args.get("subagent")
                    or args.get("name")
                )
                if subagent and subagent in histories:
                    if call_id:
                        tool_call_to_subagent[call_id] = subagent
                    desc = args.get("description") or args.get("prompt") or str(args)
                    histories[subagent].append(f"[task] {desc}")
            elif call_name in histories:
                if call_id:
                    tool_call_to_subagent[call_id] = call_name
                histories[call_name].append(f"[call] {args}")

        role = _msg_get(msg, "role") or _msg_get(msg, "type")
        role_l = str(role).lower() if role else ""
        if role_l == "tool" or _msg_get(msg, "name") == "task":
            tool_call_id = _msg_get(msg, "tool_call_id")
            subagent = tool_call_to_subagent.get(tool_call_id)
            if subagent and subagent in histories:
                content = _msg_get(msg, "content")
                if content:
                    histories[subagent].append(f"[result] {content}")

    return {name: "\n".join(lines).strip() for name, lines in histories.items() if lines}
