"""Authentication layer for the application.

OVERVIEW
========
This module implements a two-factor access control:

  1. A global password shared by all authorised users (APP_PASSWORD in
     secrets). Think of it as a door code for the tool — it prevents
     casual access by anyone who stumbles on the URL.

  2. An individual e-mail address that must appear in a whitelist
     (ALLOWED_EMAILS in secrets). This is what ties a session to a
     specific person and will serve as the user identity key when
     per-user memory is added later.

Both conditions must be met simultaneously. Failing either one is
treated the same way from the user's perspective (a single error
message) to avoid leaking information about which check failed.

SESSION PERSISTENCE
===================
Authentication state is stored in st.session_state. As long as the
browser tab stays open, the user stays logged in — Streamlit reruns
the script on every interaction but the session_state persists across
reruns within the same browser session. Closing the tab or the browser
clears session_state, so the user will be prompted again on their
next visit. No cookies, no tokens, no server-side session storage are
involved.

HOW TO USE THIS MODULE
======================
Call require_auth() at the very top of app.py, before any other
rendering. It either returns (user is authenticated) or calls st.stop()
(user sees the login page and nothing else renders):

    from auth import require_auth

    require_auth()
    # Everything below this line is only reached by authenticated users.
    st.sidebar.write(f"Logged in as {st.session_state['user_email']}")

After a successful login, two keys are always present in session_state:
    st.session_state["authenticated"]  -> True
    st.session_state["user_email"]     -> "alice@example.com"

SECRETS EXPECTED IN .streamlit/secrets.toml
============================================
    APP_PASSWORD   = "the-shared-door-code"
    ALLOWED_EMAILS = ["alice@org.be", "bob@org.be"]

See secrets.toml.example for the full expected structure.
"""

import hmac

import streamlit as st


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure the auth-related session_state keys exist.

    Called at the top of require_auth() on every script rerun. Using
    setdefault avoids overwriting values that are already set (which
    would log the user out on every rerun).
    """
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("user_email", None)


def _load_secrets() -> tuple[str, list[str]]:
    """Read auth credentials from st.secrets.

    Returns:
        A (password, allowed_emails) tuple where password is the global
        APP_PASSWORD string and allowed_emails is the list of authorised
        e-mail addresses, all lowercased and stripped for comparison.

    Raises:
        RuntimeError: If APP_PASSWORD or ALLOWED_EMAILS are missing from
            secrets. This is a configuration error that must be fixed
            before the app can run; it is not caught and converted to a
            user-friendly message because an operator (not an end user)
            is responsible for fixing it.
    """
    try:
        password = st.secrets["APP_PASSWORD"]
    except KeyError as exc:
        raise RuntimeError(
            "APP_PASSWORD is missing from .streamlit/secrets.toml."
        ) from exc

    try:
        raw_emails = st.secrets["ALLOWED_EMAILS"]
    except KeyError as exc:
        raise RuntimeError(
            "ALLOWED_EMAILS is missing from .streamlit/secrets.toml."
        ) from exc

    # Guard against a common configuration mistake: writing ALLOWED_EMAILS
    # as a single string instead of a list. Iterating over a string would
    # yield individual characters, silently breaking every login.
    if isinstance(raw_emails, str):
        raise RuntimeError(
            "ALLOWED_EMAILS must be a list of addresses, not a single string. "
            'Use ALLOWED_EMAILS = ["a@org.be", "b@org.be"] in secrets.toml.'
        )

    allowed_emails = [e.strip().lower() for e in raw_emails]
    return password, allowed_emails


def _check_credentials(password_input: str, email_input: str) -> bool:
    """Validate a login attempt against the stored secrets.

    Both the password and the e-mail must be correct simultaneously.
    The comparison is case-insensitive for e-mails and exact for the
    password (passwords are case-sensitive by convention).

    Args:
        password_input: The password the user typed.
        email_input: The e-mail address the user typed.

    Returns:
        True if both credentials are valid, False otherwise.
    """
    correct_password, allowed_emails = _load_secrets()

    # hmac.compare_digest performs a constant-time comparison, so the time
    # it takes does not reveal how many leading characters were correct.
    password_ok = hmac.compare_digest(password_input, correct_password)
    email_ok = email_input.strip().lower() in allowed_emails

    return password_ok and email_ok


def _render_login_page() -> None:
    """Render the full-page login form and handle submission.

    Displays a centred login form that occupies the whole viewport.
    The rest of the application is not rendered at all while this
    function is active (require_auth() calls st.stop() after this).

    On a valid submission, writes the authenticated state into
    session_state and calls st.rerun() so the app re-renders
    immediately without the login page. On an invalid submission,
    displays an error message without indicating which field was wrong.
    """
    # Centre the form horizontally using a 3-column layout.
    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.title("🔐 MdM IA Tool")
        st.write(
            "Please enter your credentials to access the application. "
            "Contact your administrator if you do not have access."
        )
        st.divider()

        with st.form("login_form", clear_on_submit=False):
            email_input = st.text_input(
                "Your e-mail address",
                placeholder="firstname.lastname@organisation.be",
                autocomplete="email",
            )
            password_input = st.text_input(
                "Password",
                type="password",
                placeholder="Shared access password",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button(
                "Sign in", width="stretch", type="primary"
            )

        if submitted:
            if not email_input or not password_input:
                st.error("Please fill in both fields.")
            elif _check_credentials(password_input, email_input):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = email_input.strip().lower()
                st.rerun()
            else:
                st.error(
                    "Invalid credentials. "
                    "Check your e-mail address and password, then try again."
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_auth() -> None:
    """Enforce authentication before any application content is rendered.

    Call this function as the very first statement in app.py, before
    any st.write(), st.sidebar, or tab rendering. It either returns
    silently (user is authenticated and the app can proceed) or renders
    the login page and calls st.stop() so that nothing below the
    require_auth() call is executed.

    After this function returns, the following keys are guaranteed to
    exist in st.session_state:
        "authenticated"  -> True
        "user_email"     -> the validated e-mail address (lowercase str)

    These keys are safe to read anywhere in the application:
        st.session_state["user_email"]   # use as a display name or
                                         # future memory storage key
    """
    _init_session_state()

    if st.session_state["authenticated"]:
        # Already logged in during this browser session — let the app run.
        return

    # Not yet authenticated: show the login page and stop rendering.
    _render_login_page()
    st.stop()


def get_current_user() -> str | None:
    """Return the e-mail address of the currently logged-in user.

    Convenience accessor so other modules do not have to read
    session_state directly. Returns None if called before authentication
    (which should not happen in normal flow, since require_auth() guards
    the whole app).

    Returns:
        The logged-in user's e-mail address (lowercase str), or None.
    """
    return st.session_state.get("user_email")


def logout() -> None:
    """Clear the authentication state and reload the login page.

    Intended to be called from a logout button in the sidebar. Clears
    only the auth keys so that other session_state keys (e.g. ongoing
    translation drafts) survive — this matches the principle of least
    surprise: the user explicitly chose to log out, not to lose their
    work.

    Note: because Streamlit reruns the script after st.rerun(), all
    rendering that came before logout() is discarded anyway. Clearing
    only the auth keys is therefore sufficient to show the login page.
    """
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.rerun()
