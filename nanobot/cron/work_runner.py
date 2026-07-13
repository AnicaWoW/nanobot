"""Isolated work runs for session-bound cron jobs.

A bound cron job used to execute as a turn *inside* its origin session:
deliver-by-default (silence required a sentinel reply) and every run's
machinery — cron prompt, tool dumps, suppressed finals — persisted into the
user's conversation. A work run inverts that polarity:

- **Fresh context.** The run gets the brain bootstrap
  (``ContextBuilder.build_system_prompt``) plus a work-run preamble plus the
  job text. It neither loads nor saves the bound session.
- **Speaking is an explicit act.** The user is contacted only through the
  ``message`` tool, pre-bound to the job's origin conversation; foreign
  targets are refused. Sends are mirrored into the origin session by the
  send callback (``_record_channel_delivery``).
- **Silence is the zero-action.** A run that decides nothing warrants contact
  simply finishes. No sentinel exists anywhere.
- **The final text is a run report**, written to the cron run record together
  with the list of actual sends — never delivered to the user.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from typing import Any, Awaitable, Callable, Protocol

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.runner import AgentRunSpec
from nanobot.agent.tools.context import (
    RequestContext,
    ToolContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.file_state import FileStates
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import OutboundMessage
from nanobot.cron.session_delivery import origin_delivery_context
from nanobot.cron.session_turns import CRON_TRIGGER_META
from nanobot.cron.types import CronJob
from nanobot.security.workspace_access import workspace_sandbox_status
from nanobot.utils.prompt_templates import render_template

_WORK_RUN_TEMPLATE = "agent/cron_work_run.md"

# Defer-until-idle: a work run must not start while its bound session has an
# active turn. The loop marks a session active by publishing a pending queue
# for it (``AgentLoop._pending_queues``); poll that signal and start once the
# session is idle. The wait is bounded: turns run seconds-to-minutes, and a
# run that has waited this long proceeds anyway (it touches no session state;
# the only overlap risk is a proactive send landing mid-exchange).
_DEFER_POLL_SECONDS = 1.0
_DEFER_MAX_WAIT_SECONDS = 600.0

_SEND_PREVIEW_CHARS = 200


class CronRunRecorder(Protocol):
    def write_run_record(self, run_id: str, record: dict[str, Any]) -> None:
        ...


class BoundMessageTool(MessageTool):
    """A ``message`` tool locked to one work run's origin conversation.

    Defaults are the job's ``(origin_channel, origin_chat_id)``; explicit
    ``channel``/``chat_id`` arguments naming any other target return an error
    string without sending. Accepted sends are counted for the run record.
    """

    def __init__(
        self,
        *,
        channel: str,
        chat_id: str,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None,
        workspace: Any,
        restrict_to_workspace: bool,
    ) -> None:
        super().__init__(
            send_callback=send_callback,
            default_channel=channel,
            default_chat_id=chat_id,
            workspace=workspace,
            restrict_to_workspace=restrict_to_workspace,
        )
        self._bound_channel = channel
        self._bound_chat_id = chat_id
        self.sends: list[dict[str, Any]] = []

    @property
    def description(self) -> str:
        return (
            "Send a message to the user this scheduled job belongs to "
            f"({self._bound_channel}:{self._bound_chat_id}). Call it with just "
            "'content' (plus optional 'media' file paths); the target is fixed, so "
            "omit channel/chat_id. Sending is the only way the user hears anything "
            "from this run — your final text is an internal report, never delivered."
        )

    def _foreign_target_error(self, requested: str) -> str:
        return (
            f"Error: this scheduled run may only message its own user "
            f"({self._bound_channel}:{self._bound_chat_id}); {requested} is not "
            "allowed. Omit channel/chat_id to reach the user."
        )

    async def execute(
        self,
        content: str = "",
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        if channel and str(channel).strip() and str(channel) != self._bound_channel:
            return self._foreign_target_error(f"channel {str(channel)!r}")
        if (
            chat_id is not None
            and str(chat_id).strip()
            and str(chat_id) != str(self._bound_chat_id)
        ):
            return self._foreign_target_error(f"chat_id {str(chat_id)!r}")

        result = await super().execute(content=content, channel=None, chat_id=None, **kwargs)
        if isinstance(result, str) and result.startswith("Message sent"):
            self.sends.append(
                {
                    "channel": self._bound_channel,
                    "chat_id": self._bound_chat_id,
                    "content_preview": content[:_SEND_PREVIEW_CHARS],
                }
            )
        return result


def _work_run_prompt_ref(preamble: str) -> dict[str, Any]:
    return {
        "id": "cron.agent_turn.work_run",
        "version": 1,
        "sha256": hashlib.sha256(preamble.encode("utf-8")).hexdigest(),
    }


def _build_work_run_tools(
    agent: Any,
    *,
    channel: str,
    chat_id: str,
    send_callback: Callable[[OutboundMessage], Awaitable[None]] | None,
) -> tuple[ToolRegistry, BoundMessageTool | None]:
    """Build the isolated registry for one work run.

    Subagent-scope tools via ``ToolLoader`` (the global allowlist applies, as in
    ``SubagentManager._build_tools``), except the ``ToolContext`` also carries
    ``bus``/``cron_service``/``sessions`` so scheduling/plugin tools that opt in
    to the ``subagent`` scope can construct. The stock ``message`` tool is
    core-scope; the run gets :class:`BoundMessageTool` instead (same allowlist
    name, target pre-bound).
    """
    cfg = agent.subagents._subagent_tools_config()
    root = agent.workspace
    registry = ToolRegistry()
    ctx = ToolContext(
        config=cfg,
        workspace=str(root.resolve()),
        bus=agent.bus,
        cron_service=agent.cron_service,
        sessions=agent.sessions,
        file_state_store=FileStates(),
        timezone=agent.context.timezone or "UTC",
        workspace_sandbox=workspace_sandbox_status(
            restrict_to_workspace=cfg.restrict_to_workspace,
            workspace=root,
        ),
        runtime_events=agent.runtime_events,
    )
    ToolLoader().load(ctx, registry, scope="subagent")

    allow = ToolLoader._allowed_tools(ctx)
    bound: BoundMessageTool | None = None
    if allow is None or "message" in allow:
        bound = BoundMessageTool(
            channel=channel,
            chat_id=chat_id,
            send_callback=send_callback,
            workspace=root,
            restrict_to_workspace=cfg.restrict_to_workspace,
        )
        registry.register(bound)
    return registry, bound


async def _wait_for_session_idle(agent: Any, session_key: str) -> None:
    """Bounded defer-until-idle against the loop's active-session signal."""
    pending = getattr(agent, "_pending_queues", None)
    if pending is None:
        return
    deadline = time.monotonic() + _DEFER_MAX_WAIT_SECONDS
    while session_key in pending:
        if time.monotonic() >= deadline:
            logger.warning(
                "Work run: session {} still active after {}s, starting anyway",
                session_key,
                _DEFER_MAX_WAIT_SECONDS,
            )
            return
        await asyncio.sleep(_DEFER_POLL_SECONDS)


async def run_cron_work_job(
    job: CronJob,
    *,
    agent: Any,
    cron: CronRunRecorder,
    send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
) -> str | None:
    """Execute a session-bound cron job as an isolated work run.

    ``agent`` is the live ``AgentLoop`` (brain, provider, subagent runner and
    the active-session signal all come from it); ``send_callback`` should be
    the gateway's recording deliverer so sends mirror into the origin session
    — it falls back to plain ``bus.publish_outbound``.
    """
    session_key = job.payload.session_key
    if not session_key:
        raise ValueError(f"cron job {job.id} is missing payload.session_key")
    channel, chat_id, _ = origin_delivery_context(job)

    preamble = render_template(
        _WORK_RUN_TEMPLATE,
        strip=True,
        job_name=job.name,
        channel=channel,
        chat_id=chat_id,
    )
    prompt_ref = _work_run_prompt_ref(preamble)
    run_id = f"{job.id}:{int(time.time() * 1000)}:{uuid.uuid4().hex[:8]}"
    run_record_base: dict[str, Any] = {
        "job_id": job.id,
        "job_name": job.name,
        "session_key": session_key,
        "prompt_ref": prompt_ref,
        "prompt_vars": {"message": job.payload.message},
        "rendered_prompt": preamble,
    }
    cron.write_run_record(run_id, {**run_record_base, "status": "queued"})

    if send_callback is None:
        send_callback = agent.bus.publish_outbound

    await _wait_for_session_idle(agent, session_key)

    registry, bound_message = _build_work_run_tools(
        agent,
        channel=channel,
        chat_id=chat_id,
        send_callback=send_callback,
    )
    if bound_message is not None and channel not in ("websocket", "cli"):
        # Mirror run sends into the origin session (carried recording path);
        # websocket self-persists via the webui and cli has no session.
        bound_message.set_record_channel_delivery(True)

    # The run itself never loads or saves the bound session: the session key
    # rides along for identity only (busy-defer above, tool gates below).
    system_prompt = "\n\n---\n\n".join(
        [
            agent.context.build_system_prompt(
                channel=channel,
                session_key=session_key,
                unified_session=getattr(agent, "_unified_session", False),
            ),
            preamble,
        ]
    )
    runtime_ctx = ContextBuilder._build_runtime_context(
        channel, chat_id, agent.context.timezone
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{job.payload.message}\n\n{runtime_ctx}"},
    ]

    request_ctx = RequestContext(
        channel=channel,
        chat_id=chat_id,
        session_key=session_key,
        metadata={
            CRON_TRIGGER_META: {
                "job_id": job.id,
                "job_name": job.name,
                "run_id": run_id,
                "prompt_ref": prompt_ref,
            }
        },
    )

    # Arm the in-cron guard on every CronTool-shaped tool in the run registry
    # (scheduling new jobs from inside a run is refused; list/remove stay
    # available). Plugins that re-derive the guard from request metadata get
    # the same answer from ``CRON_TRIGGER_META`` above.
    guard_tokens: list[tuple[CronTool, Any]] = []
    for name in registry.tool_names:
        tool = registry.get(name)
        if isinstance(tool, CronTool):
            guard_tokens.append((tool, tool.set_cron_context(True)))

    subagents = agent.subagents
    llm_timeout = None
    if getattr(subagents, "_llm_wall_timeout_for_session", None):
        llm_timeout = subagents._llm_wall_timeout_for_session(session_key)

    request_token = bind_request_context(request_ctx)
    try:
        result = await subagents.runner.run(
            AgentRunSpec(
                initial_messages=messages,
                tools=registry,
                model=subagents.model,
                max_iterations=subagents.max_iterations,
                max_tool_result_chars=subagents.max_tool_result_chars,
                max_iterations_message=(
                    "Run ended: max iterations reached before a final report."
                ),
                finalize_on_max_iterations=False,
                error_message=None,
                # Main-loop parity: feed tool errors back so one schema slip
                # doesn't abort a whole scheduled run.
                fail_on_tool_error=False,
                session_key=session_key,
                workspace=agent.workspace,
                llm_timeout_s=llm_timeout,
            )
        )
    except (Exception, asyncio.CancelledError) as exc:
        error_text = str(exc) or exc.__class__.__name__
        cron.write_run_record(
            run_id,
            {
                **run_record_base,
                "status": "error",
                "error": error_text,
                "sends": bound_message.sends if bound_message else [],
            },
        )
        raise
    finally:
        reset_request_context(request_token)
        for tool, token in guard_tokens:
            tool.reset_cron_context(token)

    report = result.final_content or ""
    sends = bound_message.sends if bound_message else []
    record: dict[str, Any] = {
        **run_record_base,
        "status": "error" if result.stop_reason == "error" else "ok",
        "report": report,
        "sends": sends,
    }
    if result.stop_reason == "error":
        record["error"] = result.error or "agent run failed"
    cron.write_run_record(run_id, record)
    logger.info(
        "Work run {} finished ({}): {} send(s), report {!r}",
        run_id,
        record["status"],
        len(sends),
        report[:120],
    )
    return report
