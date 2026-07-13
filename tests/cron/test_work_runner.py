"""Bound cron jobs execute as isolated work runs.

Contract under test (replaces the in-session sentinel mechanism):

- the run gets the brain bootstrap, not the session — and persists no session;
- the user hears something only via an explicit ``message`` call, pre-bound to
  the job's origin (foreign targets refused);
- a run that sends nothing is a normal ``ok`` outcome — nothing is delivered,
  no sentinel exists anywhere;
- the final text is a run report written to the cron run record (+ ``sends``);
- tools inside the run observe ``CRON_TRIGGER_META`` request metadata and the
  bound session key (the no-scheduling-from-runs guard re-derives from it);
- the run registry honors the global tool allowlist.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import (
    current_request_context,
    current_request_session_key,
)
from nanobot.agent.tools.loader import ToolLoader
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.cron import work_runner
from nanobot.cron.session_turns import CRON_TRIGGER_META
from nanobot.cron.types import CronJob, CronPayload
from nanobot.cron.work_runner import run_cron_work_job
from nanobot.providers.base import LLMResponse, ToolCallRequest

BOUND_CHANNEL = "telegram"
BOUND_CHAT_ID = "user-1"
BOUND_SESSION_KEY = f"{BOUND_CHANNEL}:{BOUND_CHAT_ID}"


class RecordingCron:
    """Capture every run-record write in order."""

    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def write_run_record(self, run_id: str, record: dict[str, Any]) -> None:
        self.records.append((run_id, dict(record)))


def _make_agent(tmp_path: Path, replies: list[LLMResponse], **loop_kwargs: Any) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        **loop_kwargs,
    )
    agent.provider.chat_with_retry = AsyncMock(side_effect=replies)
    return agent


def _job(message: str = "Prüfe die Lage und melde dich nur wenn nötig.") -> CronJob:
    return CronJob(
        id="job-1",
        name="pulse",
        payload=CronPayload(
            message=message,
            session_key=BOUND_SESSION_KEY,
            origin_channel=BOUND_CHANNEL,
            origin_chat_id=BOUND_CHAT_ID,
        ),
    )


def _session_files(tmp_path: Path) -> list[Path]:
    sessions_dir = tmp_path / "sessions"
    if not sessions_dir.exists():
        return []
    return [p for p in sessions_dir.rglob("*") if p.is_file()]


def _final(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[])


def _tool_call(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="call-1", name=name, arguments=arguments)],
    )


def _sent_messages_json(agent: AgentLoop, call_index: int) -> str:
    calls = agent.provider.chat_with_retry.call_args_list
    return json.dumps(calls[call_index].kwargs["messages"], default=str)


async def test_silence_run_delivers_nothing_and_persists_no_session(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, [_final("Lage geprüft, kein Anlass für eine Nachricht.")])
    cron = RecordingCron()

    report = await run_cron_work_job(_job(), agent=agent, cron=cron)

    assert report == "Lage geprüft, kein Anlass für eine Nachricht."
    assert agent.bus.outbound_size == 0
    assert _session_files(tmp_path) == []

    assert cron.records[0][1]["status"] == "queued"
    run_id, final_record = cron.records[-1]
    assert final_record["status"] == "ok"
    assert final_record["report"] == report
    assert final_record["sends"] == []
    assert final_record["job_id"] == "job-1"
    assert final_record["session_key"] == BOUND_SESSION_KEY
    assert run_id.startswith("job-1:")


async def test_communicating_run_publishes_to_bound_target_and_records_send(
    tmp_path: Path,
) -> None:
    agent = _make_agent(
        tmp_path,
        [
            _tool_call("message", {"content": "Guten Morgen! Der Bericht ist fertig."}),
            _final("Reminder delivered."),
        ],
    )
    cron = RecordingCron()

    report = await run_cron_work_job(_job("Erinnere den Mandanten an den Bericht."), agent=agent, cron=cron)

    assert report == "Reminder delivered."
    assert agent.bus.outbound_size == 1
    outbound = agent.bus.outbound.get_nowait()
    assert outbound.channel == BOUND_CHANNEL
    assert outbound.chat_id == BOUND_CHAT_ID
    assert outbound.content == "Guten Morgen! Der Bericht ist fertig."
    # Carried recording path: the send is flagged so the gateway deliverer
    # mirrors it into the origin session.
    assert outbound.metadata.get("_record_channel_delivery") is True

    final_record = cron.records[-1][1]
    assert final_record["status"] == "ok"
    assert final_record["sends"] == [
        {
            "channel": BOUND_CHANNEL,
            "chat_id": BOUND_CHAT_ID,
            "content_preview": "Guten Morgen! Der Bericht ist fertig.",
        }
    ]
    # The report is a run record, not a delivery.
    assert _session_files(tmp_path) == []


async def test_foreign_target_send_is_refused_and_nothing_published(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        [
            _tool_call(
                "message",
                {"content": "hi", "channel": BOUND_CHANNEL, "chat_id": "someone-else"},
            ),
            _final("Could not send."),
        ],
    )
    cron = RecordingCron()

    await run_cron_work_job(_job(), agent=agent, cron=cron)

    assert agent.bus.outbound_size == 0
    assert cron.records[-1][1]["sends"] == []
    # The refusal is fed back to the model as the tool result.
    fed_back = _sent_messages_json(agent, 1)
    assert "may only message its own user" in fed_back


async def test_same_target_explicit_args_still_send(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        [
            _tool_call(
                "message",
                {"content": "hallo", "channel": BOUND_CHANNEL, "chat_id": BOUND_CHAT_ID},
            ),
            _final("done"),
        ],
    )
    cron = RecordingCron()

    await run_cron_work_job(_job(), agent=agent, cron=cron)

    assert agent.bus.outbound_size == 1
    assert cron.records[-1][1]["sends"][0]["content_preview"] == "hallo"


class ProbeTool(Tool):
    """Reports what the request-context contextvars look like inside the run."""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "probe_context"

    @property
    def description(self) -> str:
        return "probe request context"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        ctx = current_request_context()
        trigger = (ctx.metadata if ctx else {}).get(CRON_TRIGGER_META) or {}
        return (
            f"cron_meta={bool(trigger)} job={trigger.get('job_id')} "
            f"session={current_request_session_key()}"
        )


class _ProbeLoader(ToolLoader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(test_classes=[ProbeTool])


async def test_tools_inside_run_observe_cron_trigger_meta_and_session_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user_cron-style guard re-derives "inside a cron run" from request
    metadata; the read_user_conversation gate needs the session key. Both must
    be observable from tools executed by the run."""
    monkeypatch.setattr(work_runner, "ToolLoader", _ProbeLoader)
    agent = _make_agent(
        tmp_path,
        [_tool_call("probe_context", {}), _final("probed")],
    )
    cron = RecordingCron()

    await run_cron_work_job(_job(), agent=agent, cron=cron)

    fed_back = _sent_messages_json(agent, 1)
    assert "cron_meta=True" in fed_back
    assert "job=job-1" in fed_back
    assert f"session={BOUND_SESSION_KEY}" in fed_back


async def test_system_prompt_carries_brain_bootstrap_and_work_run_preamble(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("MARKER_PLATFORM_RULES", encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("MARKER_SOUL_PERSONA", encoding="utf-8")
    (tmp_path / "USER.md").write_text("MARKER_TENANT_PROFILE", encoding="utf-8")
    agent = _make_agent(tmp_path, [_final("ok")])

    await run_cron_work_job(_job("Puls-Auftrag."), agent=agent, cron=RecordingCron())

    first_call = agent.provider.chat_with_retry.call_args_list[0].kwargs["messages"]
    system = first_call[0]["content"]
    assert "MARKER_PLATFORM_RULES" in system
    assert "MARKER_SOUL_PERSONA" in system
    assert "MARKER_TENANT_PROFILE" in system
    # The work-run preamble states the polarity platform-side.
    assert "internal run report" in system
    assert f"{BOUND_CHANNEL}:{BOUND_CHAT_ID}" in system
    # The job text is the user message; no sentinel vocabulary anywhere.
    assert first_call[1]["role"] == "user"
    assert first_call[1]["content"].startswith("Puls-Auftrag.")
    assert "NO_MESSAGE" not in system
    assert "NO_MESSAGE" not in first_call[1]["content"]


async def test_run_registry_honors_global_allowlist(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        [_final("ok")],
        tools_config=ToolsConfig(allowed_tools=["read_file", "message"]),
    )

    registry, bound = work_runner._build_work_run_tools(
        agent,
        channel=BOUND_CHANNEL,
        chat_id=BOUND_CHAT_ID,
        send_callback=agent.bus.publish_outbound,
    )

    assert registry.has("read_file")
    assert registry.has("message")
    assert bound is not None
    assert not registry.has("write_file")
    assert not registry.has("exec")


async def test_bound_message_tool_respects_allowlist_exclusion(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        [_final("ok")],
        tools_config=ToolsConfig(allowed_tools=["read_file"]),
    )

    registry, bound = work_runner._build_work_run_tools(
        agent,
        channel=BOUND_CHANNEL,
        chat_id=BOUND_CHAT_ID,
        send_callback=agent.bus.publish_outbound,
    )

    assert bound is None
    assert not registry.has("message")


async def test_run_defers_while_bound_session_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(work_runner, "_DEFER_POLL_SECONDS", 0.01)
    agent = _make_agent(tmp_path, [_final("done after idle")])
    agent._pending_queues[BOUND_SESSION_KEY] = asyncio.Queue()
    cron = RecordingCron()

    task = asyncio.create_task(run_cron_work_job(_job(), agent=agent, cron=cron))
    await asyncio.sleep(0.05)
    assert agent.provider.chat_with_retry.call_count == 0

    del agent._pending_queues[BOUND_SESSION_KEY]
    report = await asyncio.wait_for(task, timeout=5)
    assert report == "done after idle"
    assert cron.records[-1][1]["status"] == "ok"


async def test_provider_failure_writes_error_record_and_raises(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, [_final("unused")])
    agent.provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("provider down"))
    cron = RecordingCron()

    with pytest.raises(RuntimeError, match="provider down"):
        await run_cron_work_job(_job(), agent=agent, cron=cron)

    final_record = cron.records[-1][1]
    assert final_record["status"] == "error"
    assert "provider down" in final_record["error"]
