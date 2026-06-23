"""Session state, common sidebar, and contextual model selectors.

This module centralises three things that the two views (Translation and
Chatbot) share:

1. SESSION-STATE INITIALISATION
   Streamlit reruns the whole script on every interaction, but
   st.session_state persists across reruns within a browser session.
   Because both views share that single store, every key is namespaced by
   a prefix so the two views never collide:
       app_*           cross-cutting (reserved for future use)
       translation_*   owned by the Translation view
       chat_*          owned by the Chatbot view
   init_session_state() sets sensible defaults for all of them, once, so
   no view has to guard against missing keys. Keeping the full list here
   also makes it obvious, later, which keys would need to be persisted per
   user when user memory is added.

2. COMMON SIDEBAR
   The signed-in user's identity and a sign-out button. Deliberately
   minimal: the model selector is NOT here, because the two views need
   different tiers (see below).

3. CONTEXTUAL MODEL SELECTORS
   Translation and Chatbot expose different model tiers, so each view
   renders its own selector by calling the matching function here. The
   set of tiers per view lives in models_config (single source of truth);
   these functions only render it.
"""

import streamlit as st

import models_config
from auth import get_current_user, logout


# Session-state keys, grouped by owner. Listed here so the full surface of
# shared state is visible in one place.
_DEFAULTS: dict = {
    # Translation view.
    "translation_raw": "",                  # base translation text
    "translation_refined": "",              # upgraded translation text
    "translation_glossary_instructions": "",  # glossary block reused on upgrade
    "translation_source_text": "",          # the source text that was translated
    "translation_source_code": "",          # detected source ISO code
    "translation_source_name": "",          # detected source language name
    "translation_target_code": "",          # target ISO code used for the translation
    "translation_compliance": [],           # glossary discrepancies in the base translation
    "translation_refined_compliance": [],   # glossary discrepancies in the refined translation
    # Chatbot view.
    "chat_messages": [],                    # list of {"role", "content"} dicts
}


def init_session_state() -> None:
    """Initialise all shared session-state keys with their defaults.

    Call this once in app.py, after authentication and before any view
    runs. Uses setdefault so existing values are never overwritten on a
    rerun (which would wipe the user's work).
    """
    for key, default in _DEFAULTS.items():
        # Lists are mutable; give each session its own copy rather than
        # sharing the single default instance.
        st.session_state.setdefault(
            key, list(default) if isinstance(default, list) else default
        )


def render_common_sidebar() -> None:
    """Render the sidebar elements shared by every view.

    Shows the signed-in user's e-mail and a sign-out button. The model
    selector is intentionally left to each view, which renders its own
    contextual one.
    """
    with st.sidebar:
        user_email = get_current_user()
        if user_email:
            st.caption(f"Signed in as {user_email}")
        if st.button("Sign out", width="stretch"):
            logout()


def _render_model_selector(
    models: dict,
    title: str,
    state_key: str,
) -> str:
    """Render a sidebar radio selector for a set of model tiers.

    Shared implementation behind the per-view selectors. Each option is a
    role, labelled with its display name; the selected model's
    quality/cost description is shown beneath the radio.

    Args:
        models: Ordered mapping {role: ModelSpec} to offer, as returned by
            a models_config accessor.
        title: Bold heading shown above the selector.
        state_key: session_state key used to remember the chosen role and
            to key the widget (must be unique per selector).

    Returns:
        The role the user selected (a key of `models`).
    """
    roles = list(models)

    def _format(role: str) -> str:
        return models[role].display_name

    with st.sidebar:
        st.markdown(f"**{title}**")
        selected_role = st.radio(
            title,
            options=roles,
            format_func=_format,
            key=state_key,
            label_visibility="collapsed",
        )
        # Quality/cost guidance for the chosen tier.
        st.caption(models[selected_role].description)

    return selected_role


def select_translation_model() -> str:
    """Render the Translation view's model selector and return the choice.

    Offers the translation tiers (economy, standard). The premium tier is
    not offered here: for translation, premium is reached through the
    automatic upgrade step rather than chosen up front.

    Returns:
        The selected model role ("economy" or "standard").
    """
    return _render_model_selector(
        models=models_config.get_translation_models(),
        title="Translation model",
        state_key="app_translation_role",
    )


def select_chatbot_model() -> str:
    """Render the Chatbot view's model selector and return the choice.

    Offers the chatbot tiers (standard by default, premium as an upgrade).

    Returns:
        The selected model role ("standard" or "premium").
    """
    return _render_model_selector(
        models=models_config.get_chatbot_models(),
        title="Chatbot model",
        state_key="app_chatbot_role",
    )
