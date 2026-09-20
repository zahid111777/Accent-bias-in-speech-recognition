"""Unit tests for :mod:`src.metrics` using a short toy reference."""

from __future__ import annotations

import math

import pytest

from src.metrics import (
    ClipMetrics,
    compute_clip_metrics,
    dominant_script,
    is_wrong_language_output,
    non_latin_share,
    score_records,
    summarise_two_ways,
    word_outcomes,
)

REFERENCE = "please call stella ask her to bring these things"


def test_perfect_transcript_scores_zero() -> None:
    metrics = compute_clip_metrics(REFERENCE, REFERENCE)
    assert metrics.wer == 0.0
    assert (metrics.substitutions, metrics.deletions, metrics.insertions) == (0, 0, 0)
    assert metrics.hits == 9


def test_casing_and_punctuation_are_normalised_away() -> None:
    noisy = "Please, CALL Stella!  Ask her to bring these things."
    assert compute_clip_metrics(REFERENCE, noisy).wer == 0.0


def test_single_substitution() -> None:
    hypothesis = "please call steller ask her to bring these things"
    metrics = compute_clip_metrics(REFERENCE, hypothesis)
    assert metrics.substitutions == 1
    assert metrics.deletions == metrics.insertions == 0
    assert metrics.wer == pytest.approx(1 / 9)


def test_deletion_and_insertion_are_counted_separately() -> None:
    deleted = compute_clip_metrics(REFERENCE, "please call ask her to bring these things")
    assert deleted.deletions == 1 and deleted.substitutions == 0

    inserted = compute_clip_metrics(
        REFERENCE, "please call stella ask her to bring these lovely things"
    )
    assert inserted.insertions == 1 and inserted.substitutions == 0
    assert inserted.wer == pytest.approx(1 / 9)


def test_empty_hypothesis_is_all_deletions() -> None:
    metrics = compute_clip_metrics(REFERENCE, "")
    assert metrics == ClipMetrics(
        wer=1.0, substitutions=0, deletions=9, insertions=0, hits=0, ref_words=9, hyp_words=0
    )


def test_empty_reference_raises() -> None:
    with pytest.raises(ValueError):
        compute_clip_metrics("   ...   ", "anything at all")


def test_hits_plus_errors_account_for_every_reference_word() -> None:
    metrics = compute_clip_metrics(REFERENCE, "please call steller to bring things now")
    assert metrics.hits + metrics.substitutions + metrics.deletions == metrics.ref_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("please call stella", "latin"),
        ("اسٹیلا کو فون کریں اور اس سے کہیں", "arabic"),
        ("स्टेला को बुलाओ और उससे कहो", "devanagari"),
        ("12345 !!!", "none"),
    ],
)
def test_dominant_script(text: str, expected: str) -> None:
    assert dominant_script(text) == expected


def test_non_latin_share_bounds() -> None:
    assert non_latin_share("please call stella") == 0.0
    assert non_latin_share("اسٹیلا کو فون کریں") == 1.0
    assert non_latin_share("") == 0.0


def test_wrong_language_flag() -> None:
    assert is_wrong_language_output("اسٹیلا کو فون کریں اور اس سے کہیں") is True
    assert is_wrong_language_output("please call stella ask her to bring") is False


def test_short_non_latin_text_is_not_flagged() -> None:
    """A two-character return says nothing about the output language."""
    assert is_wrong_language_output("کو") is False


def test_word_outcomes_labels_every_reference_word() -> None:
    outcomes = word_outcomes(REFERENCE, "please call steller ask her bring these things")
    assert [word for word, _ in outcomes] == REFERENCE.split()
    labelled = dict(zip([w for w, _ in outcomes], [o for _, o in outcomes]))
    assert labelled["stella"] == "substituted"
    assert labelled["to"] == "missed"
    assert labelled["please"] == "correct"


def test_word_outcomes_with_empty_hypothesis_are_all_missed() -> None:
    outcomes = word_outcomes(REFERENCE, "")
    assert {outcome for _, outcome in outcomes} == {"missed"}


def test_score_records_builds_rows_and_flags() -> None:
    records = [
        {"key": "a/1.mp3", "accent": "a", "file": "audio/a/1.mp3", "text": REFERENCE},
        {
            "key": "b/2.mp3",
            "accent": "b",
            "file": "audio/b/2.mp3",
            "text": "اسٹیلا کو فون کریں اور اس سے کہیں",
        },
    ]
    rows = score_records(records, REFERENCE)
    assert [row["accent"] for row in rows] == ["a", "b"]
    assert rows[0]["wer"] == 0.0
    assert rows[0]["wrong_language_output"] is False
    assert rows[1]["wrong_language_output"] is True
    assert rows[1]["dominant_script"] == "arabic"


def test_summarise_two_ways_excludes_flagged_clips() -> None:
    rows = [
        {"wer": 0.0, "wrong_language_output": False},
        {"wer": 0.5, "wrong_language_output": False},
        {"wer": 1.0, "wrong_language_output": True},
    ]
    summary = summarise_two_ways(rows)
    assert summary["n_all"] == 3 and summary["n_clean"] == 2 and summary["n_flagged"] == 1
    assert summary["mean_wer_all"] == pytest.approx(0.5)
    assert summary["mean_wer_clean"] == pytest.approx(0.25)


def test_summarise_two_ways_on_all_flagged_clips_returns_nan_clean_mean() -> None:
    summary = summarise_two_ways([{"wer": 1.0, "wrong_language_output": True}])
    assert summary["n_clean"] == 0
    assert math.isnan(summary["mean_wer_clean"])
