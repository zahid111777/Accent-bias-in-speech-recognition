"""Text normalisation applied to both the reference and the ASR hypotheses.

The rule is deliberately blunt so that the two sides are comparable: lowercase,
keep only letters and apostrophes, and collapse runs of whitespace. Unicode
letters are preserved (rather than ASCII only) so that a hypothesis returned in
Urdu, Arabic or Devanagari script survives normalisation and can still be
detected downstream by :mod:`src.metrics`.
"""

from __future__ import annotations

import re
import unicodedata

#: Curly quotes and similar marks that stand in for a plain apostrophe.
_APOSTROPHES = "\u2019\u02bc\u2018\u00b4\u0060"

#: A character worth keeping: a Unicode letter or an apostrophe.
#: ``[^\W\d_]`` is the standard way to spell "Unicode letter" with ``re``.
_LETTER_OR_APOSTROPHE = re.compile(r"[^\W\d_]|'", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Normalise a transcript or reference string for word-error scoring.

    Steps: Unicode NFKC folding, apostrophe unification, lowercasing, removal of
    every character that is not a Unicode letter or an apostrophe, and
    whitespace collapsing. Apostrophes that no longer sit inside a word (for
    example a stray leading quote) are dropped.

    Args:
        text: Raw text, possibly empty or ``None``-like.

    Returns:
        The normalised string; may be empty if the input held no letters.

    Examples:
        >>> normalise("Please call Stella!  Ask her...")
        "please call stella ask her"
        >>> normalise("Don\u2019t   stop")
        "don't stop"
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    for mark in _APOSTROPHES:
        text = text.replace(mark, "'")
    text = text.lower()

    kept = [ch if _LETTER_OR_APOSTROPHE.match(ch) else " " for ch in text]
    text = "".join(kept)
    text = _WHITESPACE.sub(" ", text).strip()

    # Drop apostrophes that are not word-internal (e.g. "'quoted'" -> "quoted").
    tokens = [token.strip("'") for token in text.split(" ")]
    return " ".join(token for token in tokens if token)


def tokenise(text: str) -> list[str]:
    """Normalise ``text`` and split it into whitespace-delimited word tokens.

    Args:
        text: Raw text.

    Returns:
        List of normalised tokens, empty if the input held no letters.
    """
    normalised = normalise(text)
    return normalised.split(" ") if normalised else []
