"""Institutional glossary: detection, prompt injection, and compliance check.

WHY THIS MODULE EXISTS
======================
The organisation maintains an official terminology. Some everyday words
are deprecated in favour of institutional ones (for example, a source
text saying "personne en séjour illégal" must be translated as
"persoon zonder wettig verblijf", never "illegaal verblijf"). The
glossary is a NORM that overrides the source text: if the source uses a
deprecated term, the translation must still use the official one.

This module enforces that norm in two complementary ways, both built on
a single detection function:

  Upstream (preventive)
      Before translating, we detect glossary terms in the SOURCE text
      and inject their official target-language equivalents into the
      system prompt as binding instructions. This guides every text-
      producing step (base translation AND upgrade/refinement).

  Downstream (detective, passive)
      After a translation is produced, we verify that the official terms
      that were expected actually made it into the output. Any
      discrepancy is reported to the user as a warning. Nothing is
      corrected automatically — the human translator decides.

The downstream check is needed because an upgrade step, while improving
fluency, can inadvertently swap an official term for a more natural but
deprecated synonym. The norm must therefore be re-verified after every
text-producing step, not only once.

PUBLIC API
==========
    detect_glossary_terms(...)        -> list[dict]
        Core, reusable detector. Finds glossary terms present in a text.

    format_terminology_for_prompt(...) -> str
        Upstream use: turns detected terms into binding prompt
        instructions.

    check_translation_compliance(...)  -> list[dict]
        Downstream use: compares expected official terms against the
        produced translation and returns the discrepancies.

LANGUAGE CODES
==============
Throughout this module, languages are identified by the ISO codes used
as keys in GLOSSARY_DATA: "fr", "nl", "en". A target language outside
this set (the "Other" option in the UI) simply yields no glossary
matches, which is the correct, safe behaviour.
"""

import json

from llm_client import LLMError, complete_json


# ---------------------------------------------------------------------------
# Official glossary
# ---------------------------------------------------------------------------

# Each entry maps the same concept across languages. An empty string for
# a language means no official term is defined for that language; such
# entries are skipped when that language is the target.
GLOSSARY_DATA: list[dict] = [
    {"fr": "Personnes privées de titre de séjour ou sans titre de séjour", "nl": "Mensen zonder wettig verblijf", "en": "Undocumented person"},
    {"fr": "Personnes en séjour irrégulier", "nl": "Mensen zonder wettig verblijf", "en": "Undocumented person"},
    {"fr": "Personne privée de titre de séjour ou sans titre de séjour", "nl": "Persoon zonder wettig verblijf", "en": "Undocumented person"},
    {"fr": "Personnes usager.ère.s de drogues", "nl": "Drugsgebruikers", "en": "Drug user"},
    {"fr": "Personnes en situation de vulnérabilité", "nl": "Personen in maatschappelijk kwetsbare positie", "en": "People in vulnerable situations"},
    {"fr": "Personnes en situation de précarité", "nl": "Mensen in precaire situatie", "en": "People in precarious situations"},
    {"fr": "Personnes en situation de précarité", "nl": "Mensen in bestaansonzekerheid", "en": "People in precarious situations"},
    {"fr": "Personnes en situation de pauvreté", "nl": "Mensen in armoede", "en": "People experiencing poverty"},
    {"fr": "Personnes en situation de pauvreté", "nl": "Mensen die in armoede leven", "en": "People living in poverty"},
    {"fr": "Personnes refugiées", "nl": "Vluchtelingen", "en": "Refugee people"},
    {"fr": "Personnes en situation de migration", "nl": "Mensen met een migratieparcours", "en": "People in a migration situation"},
    {"fr": "Personne(s) sans abri et chez-soi", "nl": "Dak-en thuisloze persoon", "en": "Homeless people, person or people experiencing homelessness"},
    {"fr": "Sans-chez-soirisme", "nl": "dak- en thuisloosheid", "en": "Homelessness"},
    {"fr": "Aide médicale urgente", "nl": "Dringende medische hulp", "en": "Urgent medical aid"},
    {"fr": "AMU", "nl": "DMH", "en": "UMA"},
    {"fr": "Interruption volontaire de grossesse", "nl": "Vrijwillige zwangerschapsafbreking", "en": "Voluntary termination of pregnancy"},
    {"fr": "IVG", "nl": "VZA", "en": "VTP"},
    {"fr": "Demandeur·euses de Protection Internationale (DPI)", "nl": "Verzoeker om Internationale Bescherming (VIB)", "en": "Applicant for international protection"},
    {"fr": "Les travailleur.euse.s du sexe (TDS)", "nl": "Sekswerkers", "en": "Sex workers"},
    {"fr": "Les Enfants et Jeunes en Situation de Rue (EJSR)", "nl": "Kinderen en jongeren die op straat leven", "en": "Children and Young People in Street Situations (CYPS)"},
    {"fr": "Mineur (Etranger) Non Accompagné (MENA)", "nl": "Niet begeleide minderjarige (vreemdeling) (NBMV)", "en": "(Foreign) unaccompanied minor"},
    {"fr": "Ayant(s)-droit", "nl": "De rechthebbende(n)", "en": "The right-holder(s)"},
    {"fr": "Partie prenante", "nl": "De belanghebbende(n)", "en": "Stakeholders"},
    {"fr": "Santé et droit sexuels et reproductifs (SDSR)", "nl": "Seksuele en reproductieve gezondheid en rechten (SRGR)", "en": "Sexual and Reproductive Health and Rigths (SRHR)"},
    {"fr": "Santé et droits en Migration", "nl": "Gezondheid, rechten en migratie", "en": "Health and Rights in Migration (HRM)"},
    {"fr": "Réduction de risques", "nl": "Harm Reduction", "en": "Harm Reduction"},
    {"fr": "Réduction de risques", "nl": "Risicobeperking", "en": "Harm Reduction"},
    {"fr": "Mesures de réduction des risques et des dommages", "nl": "Schade- en risicobeperkende maatregelen", "en": "Harm Reduction measures"},
    {"fr": "Salle de Consommation (à Moindre Risque)", "nl": "(risicobeperkende) gebruikersruimte", "en": "Supervised injection site"},
    {"fr": "Comptoir d'échange (de matériel de réduction des risques)", "nl": "Spuitenruil project", "en": "Syringe Service Programs"},
    {"fr": "Les programmes de drug checking", "nl": "Drugstest programma's", "en": "Drug checking programs"},
    {"fr": "Psycho-médico-social", "nl": "Psycho-medisch-sociaal", "en": ""},
    {"fr": "Soins de premier ligne", "nl": "Eerstelijnszorg", "en": "Primary care"},
    {"fr": "Soins de premier ligne", "nl": "Eerstelijnsgezondheidszorg", "en": "Primary care"},
    {"fr": "Santé mentale", "nl": "Geestelijke gezondheid", "en": "Mental health"},
    {"fr": "Soins de santé mentale", "nl": "Geestelijke gezondheidszorg", "en": "Mental health care"},
    {"fr": "SSM", "nl": "GGZ", "en": ""},
    {"fr": "Services de Santé Mentale", "nl": "Centra voor Geestelijke Gezondheidszorg", "en": "Mental health services"},
    {"fr": "SSM", "nl": "CGZ", "en": ""},
    {"fr": "Problèmes de santé mentale", "nl": "Geestelijk gezondheidsproblemen", "en": "Mental health issues"},
    {"fr": "utilisateur de service", "nl": "zorggebruiker", "en": "Healthcare user"},
    {"fr": "Médecins du Monde", "nl": "Dokters van de Wereld", "en": "Doctors of the World"},
    {"fr": "CASO", "nl": "COZO", "en": ""},
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_terms_for_language(source_language: str) -> list[str]:
    """List the glossary terms that exist in a given source language.

    Args:
        source_language: ISO code ("fr", "nl", "en").

    Returns:
        The non-empty source-language terms from GLOSSARY_DATA. Empty if
        the language is not represented in the glossary.
    """
    return [
        entry[source_language]
        for entry in GLOSSARY_DATA
        if entry.get(source_language)
    ]


def _official_translation(source_term: str, source_language: str, target_language: str) -> str:
    """Look up the official target-language translation of a source term.

    Args:
        source_term: The exact glossary term in the source language.
        source_language: ISO code of the source language.
        target_language: ISO code of the target language.

    Returns:
        The official translation, or an empty string if none is defined.
    """
    for entry in GLOSSARY_DATA:
        if entry.get(source_language) == source_term:
            return entry.get(target_language, "")
    return ""


# ---------------------------------------------------------------------------
# Core detector (reused upstream and downstream)
# ---------------------------------------------------------------------------

def detect_glossary_terms(
    text: str,
    source_language: str,
    target_language: str,
) -> list[dict]:
    """Detect official glossary terms occurring in a text.

    Uses the "utility" model (Haiku) to scan the text for segments that
    match — exactly or closely (singular/plural, minor spelling
    variants, word-order changes) — the glossary terms defined for the
    source language. For each match, the official target-language
    translation is attached.

    This is the single detection primitive used by both the upstream
    prompt-injection step and the downstream compliance check.

    Args:
        text: The text to scan, written in source_language.
        source_language: ISO code of the text's language ("fr", "nl",
            "en"). Determines which glossary terms are searched for.
        target_language: ISO code of the desired translation language.
            Determines which official translation is attached to each
            match.

    Returns:
        A list of match dicts, each with:
            "source_text_term":   the segment as it appears in the text
            "glossary_term":      the official glossary term (source lang)
            "official_translation": the official term in target_language,
                                  or "" if none is defined
        Returns an empty list when no terms are found, when the source
        language is not in the glossary, or when detection fails. The
        function never raises: detection is a best-effort safeguard and
        must not block a translation.
    """
    if not text:
        return []

    candidate_terms = _source_terms_for_language(source_language)
    if not candidate_terms:
        # The glossary has nothing for this source language; nothing to do.
        return []

    terms_block = "\n".join(f"- {term}" for term in candidate_terms)

    system = (
        "You are a terminology recognition engine. You identify segments "
        "of a source text that match terms from a controlled glossary. "
        "You output only a JSON object, with no surrounding text, no "
        "explanation, and no Markdown code fences."
    )

    prompt = f"""\
Analyse the source text below (language: '{source_language}') and find every
segment that is identical or very similar to a glossary term. "Very similar"
covers singular/plural differences, minor spelling variations, and changes in
word order for multi-word terms.

Glossary terms (in '{source_language}'):
{terms_block}

Source text:
---
{text}
---

Return a JSON object with a single key "matches" whose value is a list of
objects. Each object must have exactly these two string keys:
- "source_text_term": the segment exactly as it appears in the source text
- "glossary_term": the glossary term (from the list above) that it matches

If no terms are found, return {{"matches": []}}.
"""

    try:
        result = complete_json(role="utility", system=system, prompt=prompt)
    except LLMError:
        # Detection is a safeguard, not a hard requirement. If the model
        # call or JSON parsing fails, we degrade gracefully to "no
        # matches" rather than breaking the translation flow.
        return []

    raw_matches = result.get("matches", []) if isinstance(result, dict) else []

    # Attach the official translation locally from GLOSSARY_DATA rather
    # than trusting the model to recall it. This keeps the source of
    # truth in the code, not in the model's output.
    validated: list[dict] = []
    for match in raw_matches:
        if not isinstance(match, dict):
            continue
        source_text_term = match.get("source_text_term")
        glossary_term = match.get("glossary_term")
        if not source_text_term or not glossary_term:
            continue
        validated.append({
            "source_text_term": str(source_text_term),
            "glossary_term": str(glossary_term),
            "official_translation": _official_translation(
                str(glossary_term), source_language, target_language
            ),
        })

    return validated


# ---------------------------------------------------------------------------
# Upstream use: prompt injection
# ---------------------------------------------------------------------------

def format_terminology_for_prompt(
    matches: list[dict],
    source_language: str,
    target_language: str,
) -> str:
    """Turn detected glossary matches into binding prompt instructions.

    The returned string is meant to be embedded in the system prompt of
    a translation or refinement call. It frames the glossary as an
    institutional norm that overrides the source text, not as an
    optional suggestion.

    Args:
        matches: The output of detect_glossary_terms() for the SOURCE
            text.
        source_language: ISO code of the source language (used for
            readable phrasing).
        target_language: ISO code of the target language (used for
            readable phrasing).

    Returns:
        A multi-line instruction block. If there are no matches with an
        official translation, returns a short note saying no specific
        terminology applies, so the caller can always embed the result
        unconditionally.
    """
    usable = [m for m in matches if m["official_translation"]]
    if not usable:
        return (
            "No specific glossary terminology was detected in this text. "
            "Translate using standard professional terminology."
        )

    lines = [
        "INSTITUTIONAL TERMINOLOGY (BINDING).",
        "The following terms are the organisation's official terminology. "
        "They OVERRIDE the source text: even if the source uses a different "
        "or deprecated wording, you MUST use the official translation below. "
        "This is a norm, not a suggestion.",
        "",
    ]
    for match in usable:
        lines.append(
            f"- When the text refers to '{match['glossary_term']}' "
            f"(it may appear as '{match['source_text_term']}'), "
            f"translate it as '{match['official_translation']}' "
            f"in {target_language}."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Downstream use: passive compliance check
# ---------------------------------------------------------------------------

def check_translation_compliance(
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> list[dict]:
    """Verify that expected official terms appear in the translation.

    Compares what SHOULD be in the translation (derived from glossary
    terms detected in the source) against what the translation actually
    contains. Reports every official term that is expected but missing.

    This is a passive check: it only reports discrepancies for the user
    to review. It never edits the translation.

    The comparison runs in two steps:
      1. Detect glossary terms in the source text (reusing
         detect_glossary_terms), giving the official target-language
         terms that are expected to appear.
      2. Ask the "utility" model whether each expected official term is
         present in the translation, allowing for normal inflection or
         agreement in the target language.

    Args:
        source_text: The original text.
        translated_text: The produced translation to verify.
        source_language: ISO code of the source language.
        target_language: ISO code of the target language.

    Returns:
        A list of discrepancy dicts, each with:
            "glossary_term":        the official term in the source language
            "official_translation": the official target-language term that
                                    is expected but appears to be missing
        Returns an empty list when the translation is compliant, when no
        glossary terms apply, or when the check cannot be performed. The
        function never raises.
    """
    if not source_text or not translated_text:
        return []

    source_matches = detect_glossary_terms(
        source_text, source_language, target_language
    )
    # Keep only the distinct official translations we expect to see.
    expected = {
        m["official_translation"]: m["glossary_term"]
        for m in source_matches
        if m["official_translation"]
    }
    if not expected:
        return []

    expected_block = "\n".join(f"- {term}" for term in expected)

    system = (
        "You are a terminology compliance checker. You verify whether "
        "required official terms appear in a translation. You output only "
        "a JSON object, with no surrounding text, no explanation, and no "
        "Markdown code fences."
    )

    prompt = f"""\
Below is a translation (language: '{target_language}') and a list of official
terms that MUST appear in it. A term counts as present even if it is inflected
or grammatically adjusted (agreement, plural, conjugation) as long as the same
official wording is clearly used. A term counts as MISSING if a different or
deprecated synonym was used instead, or if it is absent.

Translation:
---
{translated_text}
---

Required official terms (in '{target_language}'):
{expected_block}

Return a JSON object with a single key "missing" whose value is a list of the
required official terms (exactly as written above) that are NOT correctly
present in the translation. If all required terms are present, return
{{"missing": []}}.
"""

    try:
        result = complete_json(role="utility", system=system, prompt=prompt)
    except LLMError:
        return []

    missing_terms = result.get("missing", []) if isinstance(result, dict) else []

    discrepancies: list[dict] = []
    for term in missing_terms:
        official = str(term)
        if official in expected:
            discrepancies.append({
                "glossary_term": expected[official],
                "official_translation": official,
            })
    return discrepancies
