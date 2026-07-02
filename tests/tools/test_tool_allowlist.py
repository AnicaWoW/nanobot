"""Tests for the global tool allowlist (`tools.allowed_tools`, deny-by-default)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import ToolsConfig

# The accounting-rig surface: file + exec tools (skills run under exec), message
# (delivery), plus the `read_user_conversation` extension tool which is not part
# of this repo — the loader must simply not register anything outside this list.
RIG_ALLOWLIST = [
    "read_file", "write_file", "edit_file", "apply_patch", "find_files",
    "grep", "list_dir", "exec", "list_exec_sessions", "write_stdin",
    "message", "read_user_conversation",
]
RIG_BUILTINS = set(RIG_ALLOWLIST) - {"read_user_conversation"}


def _make_ctx(tmp_path, tools_config: ToolsConfig) -> ToolContext:
    return ToolContext(
        config=tools_config,
        workspace=str(tmp_path),
        bus=None,
        subagent_manager=SimpleNamespace(
            get_running_count=lambda: 0,
            max_concurrent_subagents=4,
        ),
        cron_service=None,
        timezone="UTC",
    )


def _named_tool(tool_name: str) -> type[Tool]:
    class _T(Tool):
        @property
        def name(self) -> str:
            return tool_name

        @property
        def description(self) -> str:
            return f"test tool {tool_name}"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> Any:
            return "ok"

    _T.__name__ = f"_Tool_{tool_name}"
    return _T


def test_default_allowlist_is_wildcard():
    assert ToolsConfig().allowed_tools == ["*"]


def test_wildcard_allowlist_registers_everything(tmp_path):
    """Default ["*"] keeps behavior identical to before the feature (backward-compatible)."""
    ctx = _make_ctx(tmp_path, ToolsConfig())
    registry = ToolRegistry()
    registered = ToolLoader().load(ctx, registry)
    assert "exec" in registered
    assert "spawn" in registered  # no off-switch tool: present without an allowlist


def test_rig_allowlist_registers_exactly_the_allowlisted_tools(tmp_path):
    """With the accounting rig's allowlist, ONLY listed tool names register.

    `read_user_conversation` is an external plugin, so from this repo's builtins
    exactly the 11 built-in names come out — and none of the no-off-switch tools
    (spawn, cron, long_task, complete_goal) sneak in.
    """
    cfg = ToolsConfig.model_validate({
        "allowed_tools": RIG_ALLOWLIST,
        "exec": {"enable": True, "sandbox": ""},
        "web": {"enable": False},
        "my": {"enable": False},
        "cli_apps": {"enable": False},
        "image_generation": {"enabled": False},
        "restrict_to_workspace": True,
    })
    ctx = _make_ctx(tmp_path, cfg)
    registry = ToolRegistry()
    registered = ToolLoader().load(ctx, registry)

    assert set(registered) == RIG_BUILTINS
    for denied in ("spawn", "cron", "long_task", "complete_goal", "web_search"):
        assert not registry.has(denied)


def test_allowlist_filters_plugin_tools_too(tmp_path):
    """The filter applies to entry-point plugins, not just builtins."""
    cfg = ToolsConfig.model_validate({"allowed_tools": ["good_tool"]})
    ctx = _make_ctx(tmp_path, cfg)
    loader = ToolLoader(test_classes=[_named_tool("good_tool"), _named_tool("bad_tool")])
    registry = ToolRegistry()
    registered = loader.load(ctx, registry)
    assert registered == ["good_tool"]
    assert not registry.has("bad_tool")


def test_empty_allowlist_falls_back_to_all(tmp_path):
    """An empty list is treated as unset (["*"]) rather than 'deny everything'."""
    cfg = ToolsConfig.model_validate({"allowed_tools": []})
    ctx = _make_ctx(tmp_path, cfg)
    loader = ToolLoader(test_classes=[_named_tool("anything")])
    registry = ToolRegistry()
    assert loader.load(ctx, registry) == ["anything"]
