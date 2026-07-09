"""Channel inbounds can request ephemeral processing via metadata.

`TurnContext.ephemeral` used to be settable only through the
`process_direct(ephemeral=...)` parameter; messages arriving over a channel
were always persistent. A channel (e.g. an operator/admin webhook channel)
can now set `metadata={"ephemeral": True}` on its InboundMessage and the
turn behaves exactly like a direct ephemeral turn: no memory consolidation,
no leak into workspace-global memory.

System messages (e.g. a subagent announce spawned from an ephemeral turn)
branch off into `_process_system_message` before the normal path and honor
the same metadata flag: the turn still persists to the session for
multi-turn continuity, but consolidation/Dream is skipped.

The tests anchor on behavior (what ephemeral turns skip/mark), not on line
numbers — loop.py is high-churn.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
    loop.tools.get_definitions = MagicMock(return_value=[])
    # Spy on the two ephemeral-gated effects: token-based consolidation runs
    # for persistent turns only (both in _state_build and after saving).
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)
    return loop


def _msg(metadata: dict | None = None) -> InboundMessage:
    return InboundMessage(
        channel="admin",
        sender_id="operator",
        chat_id="operator",
        content="hello",
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_metadata_ephemeral_flag_makes_turn_ephemeral(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    result = await loop._process_message(_msg({"ephemeral": True}))

    loop.consolidator.maybe_consolidate_by_tokens.assert_not_called()
    assert result is not None
    # ephemeral turns tag their outbound with the stop reason (loop._state_respond)
    assert "_stop_reason" in result.metadata


@pytest.mark.asyncio
async def test_channel_message_without_flag_stays_persistent(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    result = await loop._process_message(_msg())

    assert loop.consolidator.maybe_consolidate_by_tokens.called
    assert result is not None
    assert "_stop_reason" not in result.metadata


@pytest.mark.asyncio
async def test_explicit_ephemeral_parameter_still_wins(tmp_path: Path) -> None:
    """The metadata flag ORs with — never overrides — the direct parameter."""
    loop = _make_loop(tmp_path)
    result = await loop._process_message(_msg({"ephemeral": False}), ephemeral=True)

    loop.consolidator.maybe_consolidate_by_tokens.assert_not_called()
    assert result is not None
    assert "_stop_reason" in result.metadata


# --- System messages (_process_system_message), e.g. subagent announces ---


def _system_msg(metadata: dict | None = None) -> InboundMessage:
    """A subagent result announce, shaped like SubagentManager._announce_result."""
    md = {"injected_event": "subagent_result", "subagent_task_id": "t1"}
    md.update(metadata or {})
    return InboundMessage(
        channel="system",
        sender_id="subagent",
        chat_id="admin:operator",
        content="Subagent [job] completed successfully.",
        session_key_override="admin:operator",
        metadata=md,
    )


@pytest.mark.asyncio
async def test_ephemeral_system_message_skips_consolidation_but_persists(
    tmp_path: Path,
) -> None:
    """An announce from an ephemeral spawn keeps the ephemeral memory posture,
    yet the turn is still saved to the session (multi-turn continuity)."""
    loop = _make_loop(tmp_path)
    result = await loop._process_message(_system_msg({"ephemeral": True}))

    loop.consolidator.maybe_consolidate_by_tokens.assert_not_called()
    assert result is not None
    assert result.channel == "admin"
    assert "_stop_reason" in result.metadata

    session = loop.sessions.get_or_create("admin:operator")
    assert any(m.get("injected_event") == "subagent_result" for m in session.messages)
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == "ok"


@pytest.mark.asyncio
async def test_system_message_without_flag_stays_persistent(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    result = await loop._process_message(_system_msg())

    assert loop.consolidator.maybe_consolidate_by_tokens.called
    assert result is not None
    assert "_stop_reason" not in result.metadata
