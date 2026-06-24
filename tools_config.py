"""Configuration for chatbot capabilities: web tools and thinking levels.

Like model identifiers, the API version strings for server-side tools
(for example "web_search_20260318") are identifiers that change over
time. They are centralised here so that updating to a new tool version,
or adjusting a usage cap, is a one-line edit rather than a change buried
in the views.

This module covers two chatbot capabilities:
- Web tools: server-side tools Claude can call to read the live web.
- Thinking levels: how much budget the model may spend on extended
  reasoning before answering.

Docs:
- Web search: https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-search-tool
- Web fetch:  https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-fetch-tool
"""


# Web search server tool. "type" is the versioned API identifier; bump it
# here when Anthropic ships a newer version. "max_uses" caps how many
# searches Claude may run per message, which bounds the per-message cost
# (each search is billed separately, on top of token costs).
_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 3,
}

# Web fetch server tool: lets Claude read the full content of a URL the
# user provides or one found via search. Disabled by default. To enable
# it, pass web_fetch_enabled=True to build_chatbot_tools(); verify the
# version string and any required beta header against the docs first.
_WEB_FETCH_TOOL = {
    "type": "web_fetch_20260309",
    "name": "web_fetch",
    "max_uses": 3,
}


def build_chatbot_tools(
    web_search_enabled: bool,
    web_fetch_enabled: bool = False,
) -> list[dict]:
    """Build the list of server tools to enable for a chatbot call.

    Args:
        web_search_enabled: Whether to include the web search tool.
        web_fetch_enabled: Whether to include the web fetch tool. Off by
            default; web fetch lets Claude read arbitrary user-supplied
            URLs, which carries a small data-exfiltration risk, so it is
            opt-in.

    Returns:
        A list of tool-definition dicts, possibly empty. Each call gets a
        fresh copy of the definitions so callers cannot mutate the
        module-level templates.
    """
    tools: list[dict] = []
    if web_search_enabled:
        tools.append(dict(_WEB_SEARCH_TOOL))
    if web_fetch_enabled:
        tools.append(dict(_WEB_FETCH_TOOL))
    return tools


# Extended-thinking levels offered in the chatbot, in display order. The
# value is the thinking budget in tokens, or None to disable thinking.
# Higher budgets let the model reason more before answering, at higher
# cost and latency (thinking tokens are billed as output tokens).
THINKING_LEVELS: dict[str, int | None] = {
    "Off": None,
    "Standard": 4000,
    "Extended": 12000,
}


def thinking_budget(level_name: str) -> int | None:
    """Return the thinking budget in tokens for a named level.

    Args:
        level_name: One of the keys of THINKING_LEVELS ("Off",
            "Standard", "Extended").

    Returns:
        The budget in tokens, or None when thinking is off or the level
        is unknown.
    """
    return THINKING_LEVELS.get(level_name)
