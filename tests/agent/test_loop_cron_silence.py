"""Scheduled cron turns may elect silence.

A bound cron job fires as a normal session turn, and upstream delivers
whatever the turn produces — including the English "couldn't produce a final
answer" placeholder when the model returns blank text. For conditional jobs
("check X, only message the user if Y") that is wrong: no user asked
anything, so "nothing to say" must mean *no delivery*, not an error text.

Contract under test: a turn carrying cron-trigger metadata whose final text
is blank or exactly ``NO_MESSAGE`` yields no outbound message. Interactive
(non-cron) turns keep the upstream behavior bit-for-bit.

The tests anchor on behavior (what gets delivered), not on line numbers —
loop.py is high-churn.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.cron.session_turns import CRON_TRIGGER_META
from nanobot.providers.base import LLMResponse
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE


def _make_loop(tmp_path: Path, reply: str) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content=reply, tool_calls=[]))
    loop.tools.get_definitions = MagicMock(return_value=[])
    # Background consolidation is irrelevant here and chokes on the mock provider.
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)
    return loop


def _cron_msg() -> InboundMessage:
    # Mirrors what cron.bound_runner.run_bound_cron_job constructs.
    return InboundMessage(
        channel="telegram",
        sender_id="cron",
        chat_id="user-1",
        content="The scheduled time has arrived. Execute this scheduled cron job now.",
        metadata={
            CRON_TRIGGER_META: {
                "job_id": "job-1",
                "job_name": "pulse",
                "run_id": "run-1",
                "persist_content": "Scheduled cron job triggered: pulse",
            },
        },
    )


def _user_msg() -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="user-1",
        content="hello",
    )


@pytest.mark.asyncio
async def test_cron_turn_sentinel_elects_silence(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, reply="NO_MESSAGE")
    result = await loop._process_message(_cron_msg())
    assert result is None


@pytest.mark.asyncio
async def test_cron_turn_sentinel_tolerates_whitespace(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, reply="  NO_MESSAGE\n")
    result = await loop._process_message(_cron_msg())
    assert result is None


@pytest.mark.asyncio
async def test_cron_turn_blank_final_is_silent_not_placeholder(tmp_path: Path) -> None:
    """Blank model output on a scheduled turn must not deliver the error placeholder."""
    loop = _make_loop(tmp_path, reply="")
    result = await loop._process_message(_cron_msg())
    assert result is None


@pytest.mark.asyncio
async def test_cron_turn_real_content_still_delivers(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, reply="Guten Morgen, der Bericht ist fertig.")
    result = await loop._process_message(_cron_msg())
    assert result is not None
    assert result.content == "Guten Morgen, der Bericht ist fertig."
    assert result.channel == "telegram"
    assert result.chat_id == "user-1"


@pytest.mark.asyncio
async def test_cron_turn_sentinel_inside_prose_still_delivers(tmp_path: Path) -> None:
    """Only an exact-sentinel reply is silence — mentions in prose deliver."""
    loop = _make_loop(tmp_path, reply="Ich habe NO_MESSAGE notiert.")
    result = await loop._process_message(_cron_msg())
    assert result is not None
    assert result.content == "Ich habe NO_MESSAGE notiert."


@pytest.mark.asyncio
async def test_interactive_turn_sentinel_delivers_verbatim(tmp_path: Path) -> None:
    """The sentinel has no meaning outside scheduled turns."""
    loop = _make_loop(tmp_path, reply="NO_MESSAGE")
    result = await loop._process_message(_user_msg())
    assert result is not None
    assert result.content == "NO_MESSAGE"


@pytest.mark.asyncio
async def test_interactive_turn_blank_final_keeps_placeholder(tmp_path: Path) -> None:
    """Upstream behavior for interactive turns is untouched."""
    loop = _make_loop(tmp_path, reply="")
    result = await loop._process_message(_user_msg())
    assert result is not None
    assert result.content == EMPTY_FINAL_RESPONSE_MESSAGE
