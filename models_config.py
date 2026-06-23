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
            end users, not developers. Kept generic (not specific to
            translation) so it reads well in every view that shows it.
        default_temperature: Temperature used when the caller does not
            specify one. Lower values favour safe, consistent output.
        default_max_tokens: Maximum number of output tokens used when
            the caller does not specify a value. Sized generously
            because a translation can be as long as its source text.
        supports_temperature: Whether the model accepts a temperature
            parameter. Some models reject it; when this is False, the
            LLM gateway omits temperature from the request entirely.
    """

    api_id: str
    display_name: str
    provider: str
    description: str
    default_temperature: float
    default_max_tokens: int
    supports_temperature: bool = True


# ---------------------------------------------------------------------------
# Registry: every model the app can use, keyed by a short internal name
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "haiku": ModelSpec(
        api_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        provider="anthropic",
        description=(
            "Fast and nearly free. Good for simple, everyday text and "
            "internal use. For anything shared externally, a higher tier "
            "is recommended."
        ),
        default_temperature=0.2,
        default_max_tokens=8192,
    ),
    "sonnet": ModelSpec(
        api_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        description=(
            "High quality at a moderate cost (about 3x the economy tier). "
            "Strong on nuance, tone and natural phrasing. Recommended "
            "default for most work."
        ),
        default_temperature=0.3,
        default_max_tokens=16384,
    ),
    "opus": ModelSpec(
        api_id="claude-opus-4-8",
        display_name="Claude Opus 4.8",
        provider="anthropic",
        description=(
            "Top quality at the highest cost (about 5x the standard tier). "
            "Best for demanding work and for refining results. Slower than "
            "the other tiers."
        ),
        default_temperature=0.4,
        default_max_tokens=16384,
        # Opus 4.8 rejects the temperature parameter, so the gateway must
        # omit it for this model.
        supports_temperature=False,
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
# To change which tiers a view offers, edit only the relevant list here.
TRANSLATION_SELECTABLE_ROLES: list[str] = ["economy", "standard", "premium"]
CHATBOT_SELECTABLE_ROLES: list[str] = ["standard", "premium"]


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
