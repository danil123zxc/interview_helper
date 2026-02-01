import re
from typing import Any, Dict, List, Optional

import pytest

from src.schemas import ContextSchema
from src.workflow import Workflow


class DummyChunk:
    def __init__(self, content, role: str = "assistant"):
        self.content = content
        self.type = role
        self.role = role


class DummyAgent:
    def __init__(self, stream_results: List[DummyChunk], state: Optional[Dict[str, Any]] = None):
        self._stream_results = stream_results
        self.state = state or {}
        self.received_config = None

    def stream(self, *_args, **kwargs):
        self.received_config = kwargs.get("config")
        for item in self._stream_results:
            yield item, {}

    def get_state(self, _config):
        return self.state


class DummyCheckpoint:
    def __init__(self, checkpoint: Dict[str, Any]):
        self.checkpoint = checkpoint


class DummyCheckpointer:
    def __init__(self, checkpoint: Dict[str, Any]):
        self._checkpoint = checkpoint

    def get_tuple(self, _config):
        return DummyCheckpoint(self._checkpoint)


@pytest.fixture(autouse=True)
def stub_build_tools(monkeypatch):
    """Avoid constructing real external tools during Workflow init."""
    monkeypatch.setattr("src.workflow.build_tools", lambda: ({"tool": object()}, "instructions"))
    yield


@pytest.fixture
def context():
    return ContextSchema(
        role="engineer",
        resume="dummy resume",
        experience_level="junior",
        years_of_experience=1,
    )


def make_workflow(agent: DummyAgent, checkpointer: Optional[DummyCheckpointer] = None) -> Workflow:
    return Workflow(
        agent=agent,
        tools={"tool": object()},
        tools_instructions="instructions",
        checkpointer=checkpointer,
    )


def test_create_config_appends_and_prefixes_thread_id():
    wf = make_workflow(DummyAgent([]))
    config = wf._create_config()
    assert config["configurable"]["thread_id"].startswith("thread_")


def test_stream_all_uses_provided_config(context):
    config = {"configurable": {"thread_id": "thread_test"}}
    chunk = DummyChunk(content="hello", role="assistant")
    agent = DummyAgent([chunk])
    wf = make_workflow(agent)

    output = list(wf.stream_all("hi", context, config=config))

    assert output == [chunk]
    assert agent.received_config == config


def test_stream_content_returns_chunk_content(context):
    agent = DummyAgent([DummyChunk(content="hello")])
    wf = make_workflow(agent)

    output = list(wf.stream_content("hi", context))

    assert output == ["hello"]


def test_stream_ai_response_filters_roles_and_appends_md_from_state(context):
    state = {
        "values": {
            "files": {"final.md": {"text": "file content"}},
        }
    }
    chunks = [
        DummyChunk(content=["Hello", {"text": "world"}], role="assistant"),
        DummyChunk(content="ignored", role="user"),
    ]
    agent = DummyAgent(chunks, state=state)
    wf = make_workflow(agent)

    output = list(wf.stream_ai_response("hi", context, config={"configurable": {"thread_id": "t"}}))

    assert "Hello" in output
    assert "world" in output
    assert any("final.md" in item for item in output)
    assert any("file content" in item for item in output)


def test_iter_md_files_from_state_reads_nested_files():
    nested_state = {
        "values": {
            "other": [{"files": {"nested.md": {"content": "nested"}}}],
            "files": {"root.md": {"text": "root"}, "skip.txt": "nope"},
        }
    }
    agent = DummyAgent([], state=nested_state)
    wf = make_workflow(agent)

    md_files = wf._iter_md_files_from_state({"configurable": {"thread_id": "t"}})

    names = {m["name"] for m in md_files}
    assert names == {"root.md", "nested.md"}
    assert any(m["text"] == "root" for m in md_files)
    assert any(m["text"] == "nested" for m in md_files)


def test_iter_md_files_from_checkpoint_reads_channel_values():
    checkpoint = {"channel_values": {"files": {"from_checkpoint.md": {"page_content": "cp"}}}}
    checkpointer = DummyCheckpointer(checkpoint)
    wf = make_workflow(DummyAgent([]), checkpointer=checkpointer)

    md_files = wf._iter_md_files_from_checkpoint({"configurable": {"thread_id": "t"}})

    assert md_files == [{"name": "from_checkpoint.md", "text": "cp"}]


def test_extract_md_files_from_obj_collects_all_md_files():
    obj = {
        "files": {"a.md": {"text": "A"}, "b.txt": "skip"},
        "nested": [{"files": {"c.md": {"content": "C"}}}],
    }
    wf = make_workflow(DummyAgent([]))

    md_files = wf._extract_md_files_from_obj(obj)

    names = {m["name"] for m in md_files}
    assert names == {"a.md", "c.md"}
    assert any(m["text"] == "A" for m in md_files)
    assert any(m["text"] == "C" for m in md_files)


def test_stream_response_prints_delimited_output(capsys, context):
    agent = DummyAgent([DummyChunk(content="hello")])
    wf = make_workflow(agent)

    wf._stream_response("input", context)

    captured = capsys.readouterr().out
    assert captured.endswith("|")
    assert "hello" in captured


def test_execute_agent_delegates_to_stream_response(monkeypatch, context):
    calls = []

    def fake_stream_response(user_input, context):
        calls.append((user_input, context))

    wf = make_workflow(DummyAgent([]))
    monkeypatch.setattr(wf, "_stream_response", fake_stream_response)

    wf.execute_agent("input text", context)

    assert calls == [("input text", context)]
