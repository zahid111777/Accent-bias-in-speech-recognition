"""Unit tests for :mod:`src.normalise`."""

from __future__ import annotations

import pytest

from src.normalise import normalise, tokenise


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Please call Stella.", "please call stella"),
        ("  MIXED   Case\tAND\nwhitespace  ", "mixed case and whitespace"),
        ("Bob's brother, Bob!", "bob's brother bob"),
        ("Don’t stop", "don't stop"),  # curly apostrophe folded to straight
        ("six spoons, 6 spoons", "six spoons spoons"),  # digits dropped
        ("well-known snow peas", "well known snow peas"),  # hyphen becomes a space
        ("'quoted'", "quoted"),  # edge apostrophes dropped
        ("", ""),
        ("!!! ??? 123", ""),
    ],
)
def test_normalise_examples(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


def test_normalise_is_idempotent() -> None:
    once = normalise("Please call Stella. Ask her to bring THESE things!")
    assert normalise(once) == once


def test_normalise_keeps_non_latin_letters() -> None:
    """Urdu output must survive normalisation so metrics can flag it."""
    assert normalise("اسٹیلا کو") == (
        "اسٹیلا کو"
    )


def test_normalise_handles_none_like_input() -> None:
    assert normalise(None) == ""  # type: ignore[arg-type]


def test_tokenise_splits_on_single_spaces() -> None:
    assert tokenise("Please   call  Stella.") == ["please", "call", "stella"]


def test_tokenise_of_empty_text_is_empty_list() -> None:
    assert tokenise("   ...   ") == []
