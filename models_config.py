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
            (e.g. "claude-sonnet-4-6"). This is the only place in the
            codebase where such strings may appear.
        display_name: Human-friendly name shown in the UI
            (e.g. "Claude Sonnet 4.6").
        provider: Provider key, currently always "anthropic". Kept as a
            field so a second provider (e.g. "mistral") can be added
            later without changing the registry's structure.
        description: Short English text for the sidebar explaining the
            quality level and relative cost of the model. Written for
            end users, not developers.
        default_temperature: Temperature used when the caller does not
            specify one. Translation work favours low values.
        default_max_tokens: Maximum number of output tokens used when
            the caller does not specify a value. Sized generously
            because translations can be as long as their source text.
    """

    api_id: str
    display_name: str
    provider: str
    description: str
    default_temperature: float
    default_max_tokens: int


# ---------------------------------------------------------------------------
# Registry: every model the app can use, keyed by a short internal name
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "haiku": ModelSpec(
        api_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        provider="anthropic",
        description=(
            "Fast and nearly free. Good for straightforward sentences, "
            "common phrases and internal drafts. For documents that will "
            "be shared externally, an upgrade with a higher tier is "
            "recommended."
        ),
        default_temperature=0.2,
        default_max_tokens=8192,
    ),
    "sonnet": ModelSpec(
        api_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        description=(
            "High-quality translations at a moderate cost (roughly 3x the "
            "economy tier). Handles nuance, tone and idiomatic phrasing "
            "well. Recommended default for most documents."
        ),
        default_temperature=0.3,
        default_max_tokens=16384,
    ),
    "opus": ModelSpec(
        api_id="claude-opus-4-8",
        display_name="Claude Opus 4.8",
        provider="anthropic",
        description=(
            "Best quality, highest cost (roughly 5x the standard tier). "
            "Use for publication-ready texts, sensitive communication and "
            "the refinement step. Slower than the other tiers."
        ),
        default_temperature=0.4,
        default_max_tokens=16384,
    ),
}


# ---------------------------------------------------------------------------
# Roles: stable names used everywhere else in the application
# ---------------------------------------------------------------------------

# The application code NEVER asks for "sonnet" or "claude-sonnet-4-6";
# it asks for a role. Repointing a role to another model is a one-line
# change here.
ROLE_TO_MODEL: dict[str, str] = {
    # Fast, near-free translation (replaces the old Google Translate tier).
    "economy": "haiku",
    # Default quality translation.
    "standard": "sonnet",
    # Upgrade / refinement of an existing translation.
    "premium": "opus",
    # Internal machinery (glossary term detection, language tasks, ...).
    # Never shown to the user as a choice.
    "utility": "haiku",
}

# Roles the user can pick from in the sidebar model selector, in display
# order. "utility" is deliberately excluded: it is internal plumbing.
USER_SELECTABLE_ROLES: list[str] = ["economy", "standard", "premium"]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_model_for_role(role: str) -> ModelSpec:
    """Return the ModelSpec currently assigned to a role.

    Args:
        role: One of the keys of ROLE_TO_MODEL ("economy", "standard",
            "premium", "utility").

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


def get_selectable_models() -> dict[str, ModelSpec]:
    """Return the models offered in the sidebar selector.

    Returns:
        An ordered mapping {role: ModelSpec} restricted to the roles in
        USER_SELECTABLE_ROLES. The UI iterates over this mapping to
        build the radio buttons and their quality/cost descriptions.
    """
    return {role: get_model_for_role(role) for role in USER_SELECTABLE_ROLES}
