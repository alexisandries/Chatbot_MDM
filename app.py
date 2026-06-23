"""Application entry point.

This is the file Streamlit runs. It wires the application together in a
fixed boot sequence and then hands control to the selected page.

BOOT SEQUENCE
=============
1. set_page_config   Must be the very first Streamlit call.
2. require_auth      Gate the whole app behind the login page. If the
                     user is not authenticated, the login page is shown
                     and execution stops here.
3. init_session_state  Create all shared session keys with defaults,
                     once, before any page reads them.
4. st.navigation     Build the two-page navigation (Translation, Chatbot)
                     and render the menu in the sidebar.
5. render_common_sidebar  Add the signed-in identity and sign-out button
                     below the navigation menu.
6. pg.run()          Execute the page the user selected; the page renders
                     its own model selector and main content.

The two views always appear in the sidebar, so the user can switch
between them at any time. Switching pages reruns the script but preserves
session_state, so a translation in progress or a chat history is never
lost by navigating.

Each page is given an explicit url_path. This is required because both
view entry points are named render(); without distinct url_paths,
Streamlit would infer the same URL pathname from the callable name for
both pages and refuse to build the navigation.
"""

import streamlit as st

# set_page_config must run before any other Streamlit command, so it is
# placed at module import time, before the view modules are used.
st.set_page_config(page_title="MdM Translation & Chatbot", layout="wide")

import chatbot_view
import translation_view
from auth import require_auth
from session import init_session_state, render_common_sidebar


def main() -> None:
    """Run the application boot sequence and the selected page."""
    require_auth()
    init_session_state()

    pages = [
        st.Page(
            translation_view.render,
            title="Translation",
            icon="🌐",
            url_path="translation",
            default=True,
        ),
        st.Page(
            chatbot_view.render,
            title="Chatbot",
            icon="💬",
            url_path="chatbot",
        ),
    ]
    navigation = st.navigation(pages)

    render_common_sidebar()

    navigation.run()


if __name__ == "__main__":
    main()
