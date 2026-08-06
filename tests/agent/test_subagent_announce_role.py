"""Subagent announces are user-voiced events, not assistant text.

Persisted/injected as assistant-role, an announce reads to the model as its own
prior words: the template's closing instruction is ignored (the turn re-narrates
the dispatch ack instead of relaying the result) and the announce block becomes a
learned pattern the model starts fabricating inside later replies — observed in
production 2026-08-06 as invented Result JSON carrying a stale run token. As
user-role the instruction is directed at the model and neither failure mode has a
surface.

The tests anchor on behavior (persisted role, the role the provider sees, what
channel surfaces display), not on loop internals — loop.py is high-churn.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.subagent_channel_display import scrub_subagent_announce_body


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="Bericht ist fertig.", tool_calls=[])
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)
    return loop


def _announce(content: str = "", task_id: str = "t-1") -> InboundMessage:
    return InboundMessage(
        channel="system",
        sender_id="subagent",
        chat_id="admin:operator",
        content=content
        or render_template(
            "agent/subagent_announce.md",
            label="Report Run",
            status_text="completed successfully",
            task="Task: monthly_report\nPeriod: 2026-06",
            result='{"status": "ok"}',
        ),
        session_key_override="admin:operator",
        metadata={"injected_event": "subagent_result", "subagent_task_id": task_id},
    )


def test_announce_template_instruction_precedes_result() -> None:
    rendered = render_template(
        "agent/subagent_announce.md",
        label="X",
        status_text="completed successfully",
        task="Task: t",
        result="r-body",
    )
    # The old trailing "Summarize this naturally…" line taught the model to
    # compress instead of continue, and as assistant voice it was ignored anyway.
    assert "Summarize this naturally" not in rendered
    assert rendered.startswith("[Subagent 'X' completed successfully]")
    # Continuation instruction present, and the Result section is last so the
    # data sits closest to generation.
    assert "not a message from the user" in rendered
    assert rendered.rstrip().endswith("r-body")
    assert rendered.index("finish whatever part") < rendered.index("Result:\n")


def test_persist_subagent_followup_uses_user_role(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("admin:operator")

    assert loop._persist_subagent_followup(session, _announce()) is True
    record = session.messages[-1]
    assert record["role"] == "user"
    assert record["injected_event"] == "subagent_result"
    assert record["subagent_task_id"] == "t-1"

    # Same task id is deduped, whatever role history it arrives with.
    assert loop._persist_subagent_followup(session, _announce()) is False


@pytest.mark.asyncio
async def test_system_announce_reaches_provider_as_user_turn(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    result = await loop._process_system_message(_announce())

    call = loop.provider.chat_with_retry.call_args
    messages = call.kwargs.get("messages") or call.args[0]
    announce_msgs = [
        m
        for m in messages
        if isinstance(m.get("content"), str) and "automated completion notice" in m["content"]
    ]
    assert announce_msgs, "announce content must reach the provider"
    assert all(m["role"] == "user" for m in announce_msgs)
    # The turn's own output is delivered back to the origin channel.
    assert result is not None
    assert result.channel == "admin"
    assert result.content == "Bericht ist fertig."

    # And the durable record keeps the user role for later replays.
    session = loop.sessions.get_or_create("admin:operator")
    persisted = [m for m in session.messages if m.get("injected_event") == "subagent_result"]
    assert persisted and all(m["role"] == "user" for m in persisted)


def test_channel_display_scrub_handles_new_template() -> None:
    rendered = render_template(
        "agent/subagent_announce.md",
        label="Report Run",
        status_text="completed successfully",
        task="Task: monthly_report\nPeriod: 2026-06",
        result="Build ok: 42 Monate verifiziert.",
    )
    scrubbed = scrub_subagent_announce_body(rendered)
    assert scrubbed.startswith("[Subagent 'Report Run' completed successfully]")
    assert "Build ok: 42 Monate verifiziert." in scrubbed
    # Internal scaffolding stays off human-facing surfaces.
    assert "automated completion notice" not in scrubbed
    assert "Task: monthly_report" not in scrubbed
