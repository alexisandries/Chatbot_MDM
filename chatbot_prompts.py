"""Prompt construction for the chatbot interface.

This module holds the prompts used by the conversational chatbot view.
It is kept separate from the translation prompts because the two
features evolve independently: the chatbot's persona, scope, and
behaviour (and, later, options such as reasoning depth) change without
any impact on the translation prompts, and vice versa.

All prompts are written in English for uniformity; the assistant still
replies in whatever language the user writes in.
"""


def build_chatbot_system_prompt() -> str:
    """Build the system prompt for the chatbot interface.

    Defines a general-purpose, helpful assistant for the organisation's
    staff. Deliberately broad: the chatbot is a general conversational
    tool, not a translation-specific one. Tune this wording here if the
    organisation wants a more specific persona or scope.

    Returns:
        The system prompt string.
    """
    return """
    You are a helpful, knowledgeable assistant for the staff of an international 
    medical-humanitarian organisation (Doctors of the World, Médecins du Monde, Dokters van de Wereld).  
    Answer clearly and accurately, and be honest about uncertainty. Adapt to the language the user writes in. 
    When a request is ambiguous, ask a brief clarifying question rather than guessing. 
    Be concise by default and expand when the user asks for depth.
    """
