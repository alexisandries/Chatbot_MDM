"""Central registry of all LLM models used by the application.

This module is the SINGLE place where model identifiers (API model strings)
are defined. No other module in the application may hard-code a model
string. The rest of the codebase refers to models exclusively through
stable ROLES (e.g. "economy", "standard"), which this module maps to
concrete models.

Why this design:
    When Anthropic releases a new model (or when we change which tier a
    role should use), the only file that needs editing is this one. The
    UI labels, sidebar descriptions and default parameters all travel
    with the model definition, so nothing else in the app has to change.

How to update a model:
    1. Add or edit a ModelSpec entry in MODEL_REGISTRY.
    2. If needed, repoint a role in ROLE_TO_MODEL.
    That's it - do not touch any other file.

A note on sampling parameters:
    The Messages API exposes no sampling controls (no temperature, no
    top_p, no top_k). How long and how deeply a model answers is steered
    entirely through the reasoning levels defined at the bottom of this
    module. A ModelSpec therefore carries only a token ceiling and a
    description of how the model lets thinking be switched off.

A note on thinking:
    Every current Claude model reasons ("thinks") before answering, using
    adaptive thinking: the model decides on its own when and how much to
    think, and the `effort` parameter is the dial that controls that
    amount. Models differ only in whether thinking can be switched off
    entirely, and in which API value does it. That difference is captured
    per model by ModelSpec.thinking_off_mode (see its docstring).
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one LLM model.

    Attributes:
        api_id: Exact model string expected by the provider's API
            (e.g. "claude-sonnet-5-5"). This is the only place in the
            codebase where such strings may appear.
        display_name: Human-friendly name shown in the UI
            (e.g. "Claude Sonnet 5.5").
        provider: Provider key, currently always "anthropic". Kept as a
            field so a second provider (e.g. "mistral") can be added
            later without changing the registry's structure.
        description: Short English text for the sidebar explaining the
            quality level and relative cost of the model. Written for
            end users, not developers. Kept generic (not specific to
            translation) so it reads well in every view that shows it.
        default_max_tokens: Maximum number of output tokens used when
            the caller does not specify a value. Sized generously
            because a translation can be as long as its source text,
            and because thinking tokens count towards this ceiling.
        thinking_off_mode: How the model lets a caller switch thinking
            off, i.e. which value the gateway must send as
            thinking={"type": ...} when a reasoning level asks for no
            thinking. One of:
              - "disabled": the model accepts thinking={"type": "disabled"}
                (only at effort "high" or below; the API rejects it at
                "xhigh" and "max").
              - "between_tools": the model rejects "disabled" and uses
                thinking={"type": "between_tools"} instead, which turns
                off up-front reasoning (same effort restriction as above).
              - None: thinking is always on and cannot be switched off.
                The gateway then falls back to adaptive thinking at the
                level's effort, which at "low" effort means very little
                reasoning in practice.
            Any model accepts thinking={"type": "adaptive"} and the
            output_config.effort parameter, so no flag is needed for those.
    """

    api_id: str
    display_name: str
    provider: str
    description: str
    default_max_tokens: int
    thinking_off_mode: str | None


# ---------------------------------------------------------------------------
# Registry: every model the app can use, keyed by a short internal name
# ---------------------------------------------------------------------------
# List prices (USD per million tokens, input / output) at the time of
# writing, used to word the relative-cost hints in the descriptions:
#   Haiku 5.5   0.10 / 0.50   (5x more for prompts above 100k tokens;
#                              the app's chunking keeps prompts far below)
#   Sonnet 5.5  2 / 10        (about 20x Haiku)
#   Opus 5.5    4 / 20        (about 2x Sonnet)
#   Fable 5.1   10 / 50       (about 2.5x Opus)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "haiku": ModelSpec(
        api_id="claude-haiku-5-5",
        display_name="Claude Haiku 5.5",
        provider="anthropic",
        description=(
            "Fast and nearly free (about 20x cheaper than the standard "
            "tier). Good for simple, everyday text and internal use. For "
            "anything shared externally, a higher tier is recommended."
        ),
        # Thinking is switched off for the mechanical tasks this model
        # runs (translation, glossary detection), so this ceiling only
        # has to hold the visible answer of one text chunk.
        default_max_tokens=32000,
        thinking_off_mode="disabled",
    ),
    "sonnet": ModelSpec(
        api_id="claude-sonnet-5-5",
        display_name="Claude Sonnet 5.5",
        provider="anthropic",
        description=(
            "High quality at a moderate cost. Strong on nuance, tone and "
            "natural phrasing. Recommended default for most work."
        ),
        default_max_tokens=64000,
        # Sonnet 5.5 rejects thinking={"type": "disabled"} with a 400
        # error at every effort level; "between_tools" is its equivalent.
        thinking_off_mode="between_tools",
    ),
    "opus": ModelSpec(
        api_id="claude-opus-5-5",
        display_name="Claude Opus 5.5",
        provider="anthropic",
        description=(
            "High-end quality at about 2x the standard tier. Best for "
            "complex tasks, demanding enterprise work, and for refining "
            "results. Slower than the standard and fast tiers."
        ),
        # Thinking is always on and counts towards max_tokens, so this
        # ceiling must leave room for both the reasoning and the answer.
        default_max_tokens=64000,
        thinking_off_mode=None,
    ),
    "fable": ModelSpec(
        api_id="claude-fable-5-1",
        display_name="Claude Fable 5.1",
        provider="anthropic",
        description=(
            "Top-of-the-range model, and the most expensive (about 2.5x "
            "the high-end tier). Best for open-ended thinking, writing and "
            "long, difficult problems. Reserve it for work where the extra "
            "quality is worth the cost."
        ),
        default_max_tokens=64000,
        thinking_off_mode=None,
    ),
}


# ---------------------------------------------------------------------------
# Roles: stable names used everywhere else in the application
# ---------------------------------------------------------------------------
# The application code NEVER asks for "sonnet" or "claude-sonnet-5-5";
# it asks for a role. Repointing a role to another model is a one-line
# change here.

ROLE_TO_MODEL: dict[str, str] = {
    # Fast, near-free translation (replaces the old Google Translate tier).
    "economy": "haiku",
    # Default quality translation.
    "standard": "sonnet",
    # Upgrade / refinement of an existing translation.
    "premium": "opus",
    # Highest capability available, at the highest cost. Offered in the
    # chatbot only, for open-ended or particularly demanding requests.
    "frontier": "fable",
    # Internal machinery (glossary term detection, language tasks, ...).
    # Never shown to the user as a choice.
    "utility": "haiku",
}

# Roles each view offers in its own model selector, in display order.
# The selectors are CONTEXTUAL: translation and chatbot can expose
# different tiers, because the same role does not mean the same thing in
# both. "utility" is never user-selectable (internal plumbing).
#
# Note: the translation "Upgrade" button always uses the "premium" role,
# regardless of which model is selected here. Offering "premium" in the
# translation selector simply lets the user also run the FIRST-PASS
# translation with the top model.
#
# "frontier" is deliberately absent from the translation selector: its
# added value is reasoning depth, which translation does not need, while
# its cost is the highest of all tiers.
#
# To change which tiers a view offers, edit only the relevant list here.

TRANSLATION_SELECTABLE_ROLES: list[str] = ["economy", "standard", "premium"]
CHATBOT_SELECTABLE_ROLES: list[str] = ["standard", "premium", "frontier"]


# ---------------------------------------------------------------------------
# Reasoning levels
# ---------------------------------------------------------------------------
# Two parameters steer how a model answers:
#   - thinking: "adaptive" lets the model decide when and how deeply to
#     reason; the model-specific "off" value (see ModelSpec.thinking_off_mode)
#     forbids up-front reasoning on models that allow it.
#   - output_config.effort: how many tokens the model may spend on the
#     whole response, thinking included.
#
# The levels below are what the chatbot sidebar slider exposes. To add,
# remove or retune a level, edit only this dictionary.

@dataclass(frozen=True)
class ReasoningLevel:
    """One position of the chatbot's "Reasoning" slider.

    Attributes:
        thinking_enabled: Whether the model may reason before answering.
            True maps to thinking={"type": "adaptive"}. False asks the
            gateway to switch thinking off using the model's own
            thinking_off_mode; on models where thinking is always on,
            the gateway keeps adaptive thinking and relies on the low
            effort below to keep reasoning minimal.
        effort: Effort level sent as output_config.effort. Valid values,
            from cheapest to most expensive: "low", "medium", "high",
            "xhigh", "max". Higher levels mean longer, deeper and more
            costly responses.
        help_text: Short English sentence shown next to the slider so
            users understand the speed/cost trade-off.
    """

    thinking_enabled: bool
    effort: str
    help_text: str


# Ordered from fastest and cheapest to slowest and most expensive.
# Note: "Off" is paired with low effort on purpose. The API only allows
# thinking to be switched off at effort "high" or below, so a level with
# thinking_enabled=False must never use "xhigh" or "max".
REASONING_LEVELS: dict[str, ReasoningLevel] = {
    "Off": ReasoningLevel(
        thinking_enabled=False,
        effort="low",
        help_text="Fastest and cheapest. Answers directly, minimal reasoning.",
    ),
    "Standard": ReasoningLevel(
        thinking_enabled=True,
        effort="medium",
        help_text="Balanced. Thinks briefly on harder questions.",
    ),
    "Deep": ReasoningLevel(
        thinking_enabled=True,
        effort="high",
        help_text="Thorough reasoning. Slower and more expensive.",
    ),
    "Extended": ReasoningLevel(
        thinking_enabled=True,
        effort="xhigh",
        help_text=(
            "Maximum depth for long or complex problems. Slowest and "
            "most expensive."
        ),
    ),
}

# Level applied when the user has not chosen one yet.
DEFAULT_REASONING_LEVEL: str = "Standard"


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_reasoning_level(level_name: str) -> ReasoningLevel:
    """Return the ReasoningLevel matching a slider label.

    Args:
        level_name: One of the keys of REASONING_LEVELS.

    Returns:
        The matching ReasoningLevel.

    Raises:
        KeyError: If the label is unknown. This is a configuration or UI
            bug and should fail loudly rather than silently downgrade
            the user's choice.
    """
    if level_name not in REASONING_LEVELS:
        raise KeyError(
            f"Unknown reasoning level '{level_name}'. "
            f"Valid levels: {list(REASONING_LEVELS)}"
        )
    return REASONING_LEVELS[level_name]


def get_model_for_role(role: str) -> ModelSpec:
    """Return the ModelSpec currently assigned to a role.

    Args:
        role: One of the keys of ROLE_TO_MODEL ("economy", "standard",
            "premium", "frontier", "utility").

    Returns:
        The ModelSpec the role is mapped to.

    Raises:
        KeyError: If the role is unknown, or if the role points to a
            model key that does not exist in MODEL_REGISTRY. Both cases
            are configuration bugs and should fail loudly at startup
            rather than silently fall back to another model.
    """
    if role not in ROLE_TO_MODEL:
        raise KeyError(
            f"Unknown model role '{role}'. "
            f"Valid roles: {sorted(ROLE_TO_MODEL)}"
        )

    model_key = ROLE_TO_MODEL[role]
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Role '{role}' points to unknown model key '{model_key}'. "
            f"Check ROLE_TO_MODEL and MODEL_REGISTRY in models_config.py."
        )

    return MODEL_REGISTRY[model_key]


def _models_for_roles(roles: list[str]) -> dict[str, ModelSpec]:
    """Build an ordered {role: ModelSpec} mapping for a list of roles.

    Args:
        roles: The roles to include, in the desired display order.

    Returns:
        An ordered mapping from each role to its ModelSpec.

    Raises:
        KeyError: If any role is unknown or points to a missing model
            (propagated from get_model_for_role).
    """
    return {role: get_model_for_role(role) for role in roles}


def get_translation_models() -> dict[str, ModelSpec]:
    """Return the models offered in the Translation view's selector.

    Returns:
        An ordered mapping {role: ModelSpec} for the translation tiers.
        The UI iterates over it to build the radio buttons and their
        quality/cost descriptions.
    """
    return _models_for_roles(TRANSLATION_SELECTABLE_ROLES)


def get_chatbot_models() -> dict[str, ModelSpec]:
    """Return the models offered in the Chatbot view's selector.

    Returns:
        An ordered mapping {role: ModelSpec} for the chatbot tiers. The
        UI iterates over it to build the radio buttons and their
        quality/cost descriptions.
    """
    return _models_for_roles(CHATBOT_SELECTABLE_ROLES)
