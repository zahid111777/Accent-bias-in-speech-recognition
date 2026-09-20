"""Unit tests for the hand-rolled statistics in :mod:`src.analysis`."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    bootstrap_ci,
    cliffs_delta,
    error_type_breakdown,
    hardest_words,
    holm_correction,
    interpret_cliffs_delta,
    pairwise_vs_focus,
    summarise_by_accent,
    word_error_rates,
)

REFERENCE = "please call stella ask her to bring these things"


def _toy_frame() -> pd.DataFrame:
    """Two accents, one flagged clip, with hand-checkable error counts."""
    return pd.DataFrame(
        [
            {
                "key": "pakistani/1.mp3",
                "accent": "pakistani",
                "text_normalised": "please call steller ask her bring these things",
                "wer": 2 / 9,
                "substitutions": 1,
                "deletions": 1,
                "insertions": 0,
                "hits": 7,
                "ref_words": 9,
                "wrong_language_output": False,
                "dominant_script": "latin",
            },
            {
                "key": "pakistani/2.mp3",
                "accent": "pakistani",
                "text_normalised": "",
                "wer": 1.0,
                "substitutions": 0,
                "deletions": 9,
                "insertions": 0,
                "hits": 0,
                "ref_words": 9,
                "wrong_language_output": True,
                "dominant_script": "arabic",
            },
            {
                "key": "english_us/1.mp3",
                "accent": "english_us",
                "text_normalised": REFERENCE,
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 9,
                "ref_words": 9,
                "wrong_language_output": False,
                "dominant_script": "latin",
            },
            {
                "key": "english_us/2.mp3",
                "accent": "english_us",
                "text_normalised": "please call stella ask her to bring these thing now",
                "wer": 2 / 9,
                "substitutions": 1,
                "deletions": 0,
                "insertions": 1,
                "hits": 8,
                "ref_words": 9,
                "wrong_language_output": False,
                "dominant_script": "latin",
            },
        ]
    )


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    low, high = bootstrap_ci(values, n_resamples=2000, seed=1)
    assert low <= np.mean(values) <= high
    assert bootstrap_ci(values, n_resamples=2000, seed=1) == (low, high)


def test_bootstrap_ci_edge_cases() -> None:
    assert all(math.isnan(bound) for bound in bootstrap_ci([]))
    assert bootstrap_ci([0.42]) == (0.42, 0.42)


def test_cliffs_delta_extremes_and_ties() -> None:
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert math.isnan(cliffs_delta([], [1, 2]))


def test_interpret_cliffs_delta_labels() -> None:
    assert interpret_cliffs_delta(0.05) == "negligible"
    assert interpret_cliffs_delta(0.2) == "small"
    assert interpret_cliffs_delta(0.4) == "medium"
    assert interpret_cliffs_delta(-0.9) == "large"
    assert interpret_cliffs_delta(float("nan")) == "undefined"


def test_holm_correction_matches_worked_example() -> None:
    # Sorted p = .01, .02, .03 with m = 3 gives .03, .04, .04 after step-down.
    adjusted = holm_correction([0.01, 0.02, 0.03])
    assert adjusted == pytest.approx([0.03, 0.04, 0.04])


def test_holm_correction_is_monotonic_and_capped() -> None:
    adjusted = holm_correction([0.5, 0.6, 0.7])
    assert all(value <= 1.0 for value in adjusted)
    assert adjusted == sorted(adjusted)


def test_holm_correction_preserves_order_and_nans() -> None:
    adjusted = holm_correction([0.04, float("nan"), 0.01])
    assert math.isnan(adjusted[1])
    # Family size is 2, not 3: the nan does not inflate the correction.
    assert adjusted[2] == pytest.approx(0.02)
    assert adjusted[0] == pytest.approx(0.04)


def test_summarise_by_accent_reports_both_views() -> None:
    summary = summarise_by_accent(_toy_frame(), n_resamples=500)
    pakistani = summary.set_index("accent").loc["pakistani"]
    assert pakistani["n"] == 2 and pakistani["n_clean"] == 1
    assert pakistani["n_wrong_language"] == 1
    assert pakistani["mean_wer"] == pytest.approx((2 / 9 + 1.0) / 2)
    assert pakistani["mean_wer_clean"] == pytest.approx(2 / 9)


def test_pairwise_vs_focus_shape_and_correction() -> None:
    table = pairwise_vs_focus(_toy_frame(), focus="pakistani")
    assert list(table["group_b"]) == ["english_us"]
    assert table.iloc[0]["n_a"] == 2 and table.iloc[0]["n_b"] == 2
    assert 0.0 <= table.iloc[0]["p_value_holm"] <= 1.0


def test_pairwise_vs_focus_returns_empty_when_focus_missing() -> None:
    assert pairwise_vs_focus(_toy_frame(), focus="martian").empty


def test_error_type_breakdown_shares_sum_to_one() -> None:
    table = error_type_breakdown(_toy_frame())
    shares = table[["share_substitutions", "share_deletions", "share_insertions"]].sum(axis=1)
    assert shares.to_numpy() == pytest.approx(np.ones(len(table)))


def test_word_error_rates_counts_every_occurrence() -> None:
    table = word_error_rates(_toy_frame(), REFERENCE)
    pakistani = table[table["accent"] == "pakistani"].set_index("word")
    # Two Pakistani clips: "stella" substituted once and missed once -> rate 1.0
    assert pakistani.loc["stella", "n_occurrences"] == 2
    assert pakistani.loc["stella", "error_rate"] == pytest.approx(1.0)
    # "please" survives in clip 1 and is missed in the empty clip 2 -> 0.5
    assert pakistani.loc["please", "error_rate"] == pytest.approx(0.5)


def test_hardest_words_is_ranked_and_capped() -> None:
    table = hardest_words(_toy_frame(), REFERENCE, focus="pakistani", top_n=3)
    assert len(table) == 3
    rates = table["error_rate_focus"].tolist()
    assert rates == sorted(rates, reverse=True)
    assert set(table["focus_accent"]) == {"pakistani"}
    assert "error_rate_rest" in table.columns
