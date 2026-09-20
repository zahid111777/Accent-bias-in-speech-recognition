"""Group-level statistics: summaries, significance tests and word-level errors.

The design question is narrow: does Whisper make more errors on the Pakistani
accent group than on the comparison groups, on the same paragraph? Because WER
distributions are skewed and the groups are small, the comparison uses a rank
test (Mann-Whitney U) with Holm correction across the family of comparisons, and
reports Cliff's delta as a distribution-free effect size alongside a
bootstrapped confidence interval for each group mean.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import FOCUS_ACCENT, SEED
from .metrics import word_outcomes
from .normalise import tokenise

LOGGER = logging.getLogger(__name__)

#: Number of bootstrap resamples used for every confidence interval.
N_BOOTSTRAP: int = 10_000

#: Magnitude thresholds for interpreting Cliff's delta (Romano et al., 2006).
CLIFF_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.147, "negligible"),
    (0.330, "small"),
    (0.474, "medium"),
)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Args:
        values: Observed values, e.g. per-clip WERs.
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage of the interval, e.g. 0.95.
        seed: Seed for the resampling RNG, so runs are reproducible.

    Returns:
        ``(low, high)``; ``(nan, nan)`` for an empty input, and ``(x, x)`` when
        a single value is given.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return (float(low), float(high))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size between two independent samples.

    Args:
        a: First sample (here: the focus accent).
        b: Second sample (here: the comparison accent).

    Returns:
        A value in ``[-1, 1]``. Positive means values in ``a`` tend to be
        larger than values in ``b``. ``nan`` if either sample is empty.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    comparison = np.sign(x[:, None] - y[None, :])
    return float(comparison.sum() / (x.size * y.size))


def interpret_cliffs_delta(delta: float) -> str:
    """Label a Cliff's delta magnitude as negligible/small/medium/large."""
    if np.isnan(delta):
        return "undefined"
    magnitude = abs(delta)
    for threshold, label in CLIFF_THRESHOLDS:
        if magnitude < threshold:
            return label
    return "large"


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment of a family of p-values.

    Args:
        p_values: Raw p-values, possibly containing ``nan``.

    Returns:
        Adjusted p-values in the original order, each capped at 1.0. ``nan``
        inputs stay ``nan`` and are excluded from the family size.
    """
    raw = np.asarray(list(p_values), dtype=float)
    valid = ~np.isnan(raw)
    adjusted = np.full(raw.shape, np.nan, dtype=float)

    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return adjusted.tolist()

    order = indices[np.argsort(raw[indices], kind="stable")]
    m = order.size
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * raw[idx]
        running = max(running, scaled)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def summarise_by_accent(
    frame: pd.DataFrame,
    seed: int = SEED,
    n_resamples: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Build the per-accent WER summary table.

    Every statistic is reported twice: over all clips, and over clips excluding
    those flagged as wrong-language output.

    Args:
        frame: Per-clip results from :func:`src.metrics.score_records`.
        seed: Seed for the bootstrap.
        n_resamples: Bootstrap resamples per group.

    Returns:
        One row per accent, sorted by mean WER descending.
    """
    rows: list[dict[str, object]] = []
    for accent, group in frame.groupby("accent", sort=True):
        wers = group["wer"].astype(float).to_numpy()
        clean = group.loc[~group["wrong_language_output"].astype(bool), "wer"]
        clean_wers = clean.astype(float).to_numpy()
        low, high = bootstrap_ci(wers, n_resamples=n_resamples, seed=seed)
        clean_low, clean_high = bootstrap_ci(clean_wers, n_resamples=n_resamples, seed=seed)
        rows.append(
            {
                "accent": accent,
                "n": int(wers.size),
                "mean_wer": float(np.mean(wers)) if wers.size else float("nan"),
                "std_wer": float(np.std(wers, ddof=1)) if wers.size > 1 else float("nan"),
                "median_wer": float(np.median(wers)) if wers.size else float("nan"),
                "ci95_low": low,
                "ci95_high": high,
                "n_wrong_language": int(group["wrong_language_output"].astype(bool).sum()),
                "n_clean": int(clean_wers.size),
                "mean_wer_clean": (
                    float(np.mean(clean_wers)) if clean_wers.size else float("nan")
                ),
                "median_wer_clean": (
                    float(np.median(clean_wers)) if clean_wers.size else float("nan")
                ),
                "ci95_low_clean": clean_low,
                "ci95_high_clean": clean_high,
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("mean_wer", ascending=False, ignore_index=True)


def pairwise_vs_focus(
    frame: pd.DataFrame,
    focus: str = FOCUS_ACCENT,
    value_column: str = "wer",
) -> pd.DataFrame:
    """Compare the focus accent against every other accent.

    Each comparison is a two-sided Mann-Whitney U test; the resulting family of
    p-values is Holm-corrected, and Cliff's delta is reported as effect size.

    Args:
        frame: Per-clip results.
        focus: Accent group to compare against all others.
        value_column: Column holding the per-clip metric.

    Returns:
        One row per comparison, sorted by adjusted p-value. Empty if the focus
        group is absent or there is nothing to compare it with.
    """
    accents = sorted(frame["accent"].unique())
    if focus not in accents:
        LOGGER.warning(
            "Focus accent %r not present in the results (found: %s); "
            "skipping pairwise tests.",
            focus,
            ", ".join(accents) or "none",
        )
        return pd.DataFrame()

    focus_values = frame.loc[frame["accent"] == focus, value_column].astype(float).to_numpy()
    rows: list[dict[str, object]] = []
    for accent in accents:
        if accent == focus:
            continue
        other = frame.loc[frame["accent"] == accent, value_column].astype(float).to_numpy()
        if focus_values.size == 0 or other.size == 0:
            u_stat, p_value = float("nan"), float("nan")
        else:
            u_stat, p_value = stats.mannwhitneyu(
                focus_values, other, alternative="two-sided"
            )
        delta = cliffs_delta(focus_values, other)
        rows.append(
            {
                "group_a": focus,
                "group_b": accent,
                "n_a": int(focus_values.size),
                "n_b": int(other.size),
                "mean_a": float(np.mean(focus_values)) if focus_values.size else float("nan"),
                "mean_b": float(np.mean(other)) if other.size else float("nan"),
                "median_a": float(np.median(focus_values)) if focus_values.size else float("nan"),
                "median_b": float(np.median(other)) if other.size else float("nan"),
                "mannwhitney_u": float(u_stat),
                "p_value": float(p_value),
                "cliffs_delta": delta,
                "effect_size_label": interpret_cliffs_delta(delta),
            }
        )

    if not rows:
        LOGGER.warning("Only one accent group present; no pairwise tests to run.")
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    table["p_value_holm"] = holm_correction(table["p_value"].tolist())
    table["significant_holm_0.05"] = table["p_value_holm"] < 0.05
    return table.sort_values("p_value_holm", ascending=True, ignore_index=True)


def error_type_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of substitutions, deletions and insertions per accent.

    Args:
        frame: Per-clip results.

    Returns:
        One row per accent with raw counts, shares of total errors, and each
        error type expressed per reference word.
    """
    grouped = frame.groupby("accent", sort=True)[
        ["substitutions", "deletions", "insertions", "hits", "ref_words"]
    ].sum()
    grouped["total_errors"] = (
        grouped["substitutions"] + grouped["deletions"] + grouped["insertions"]
    )

    table = grouped.reset_index()
    for kind in ("substitutions", "deletions", "insertions"):
        table[f"share_{kind}"] = np.where(
            table["total_errors"] > 0, table[kind] / table["total_errors"], np.nan
        )
        table[f"{kind}_per_ref_word"] = np.where(
            table["ref_words"] > 0, table[kind] / table["ref_words"], np.nan
        )
    return table


def word_error_rates(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Per-accent error rate for every word of the reference paragraph.

    A reference word counts as an error on a clip when it was missed (deleted)
    or substituted. Words that occur more than once in the paragraph are pooled
    across their occurrences.

    Args:
        frame: Per-clip results, including ``text_normalised``.
        reference: The reference paragraph.

    Returns:
        Long-format table with ``accent``, ``word``, ``n_occurrences``,
        ``n_missed``, ``n_substituted``, ``n_errors`` and ``error_rate``.
    """
    vocabulary = sorted(set(tokenise(reference)))
    counters: dict[tuple[str, str], dict[str, int]] = {}

    for row in frame.itertuples(index=False):
        accent = getattr(row, "accent")
        hypothesis = getattr(row, "text_normalised", "") or ""
        for word, outcome in word_outcomes(reference, hypothesis):
            bucket = counters.setdefault(
                (accent, word), {"occurrences": 0, "missed": 0, "substituted": 0}
            )
            bucket["occurrences"] += 1
            if outcome == "missed":
                bucket["missed"] += 1
            elif outcome == "substituted":
                bucket["substituted"] += 1

    rows: list[dict[str, object]] = []
    for (accent, word), bucket in counters.items():
        errors = bucket["missed"] + bucket["substituted"]
        rows.append(
            {
                "accent": accent,
                "word": word,
                "n_occurrences": bucket["occurrences"],
                "n_missed": bucket["missed"],
                "n_substituted": bucket["substituted"],
                "n_errors": errors,
                "error_rate": errors / bucket["occurrences"] if bucket["occurrences"] else np.nan,
                "missed_rate": (
                    bucket["missed"] / bucket["occurrences"] if bucket["occurrences"] else np.nan
                ),
                "substituted_rate": (
                    bucket["substituted"] / bucket["occurrences"]
                    if bucket["occurrences"]
                    else np.nan
                ),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "accent",
                "word",
                "n_occurrences",
                "n_missed",
                "n_substituted",
                "n_errors",
                "error_rate",
                "missed_rate",
                "substituted_rate",
            ]
        )
    missing_words = set(vocabulary) - set(table["word"])
    if missing_words:
        LOGGER.debug("Reference words never scored: %s", sorted(missing_words))
    return table.sort_values(["accent", "error_rate"], ascending=[True, False], ignore_index=True)


def hardest_words(
    frame: pd.DataFrame,
    reference: str,
    focus: str = FOCUS_ACCENT,
    top_n: int = 15,
) -> pd.DataFrame:
    """Rank the reference words that the focus accent loses most often.

    Args:
        frame: Per-clip results.
        reference: The reference paragraph.
        focus: The accent group of interest.
        top_n: How many words to keep.

    Returns:
        Table with the focus group's error rate, the pooled error rate of every
        other group ("rest"), and the difference, sorted by focus error rate.
    """
    per_word = word_error_rates(frame, reference)
    if per_word.empty:
        return pd.DataFrame()

    focus_rows = per_word[per_word["accent"] == focus]
    rest_rows = per_word[per_word["accent"] != focus]
    if focus_rows.empty:
        LOGGER.warning(
            "Focus accent %r absent from the word-level table; "
            "hardest-word comparison skipped.",
            focus,
        )
        return pd.DataFrame()

    rest_pooled = (
        rest_rows.groupby("word", sort=True)[["n_occurrences", "n_errors"]]
        .sum()
        .reset_index()
    )
    rest_pooled["error_rate_rest"] = np.where(
        rest_pooled["n_occurrences"] > 0,
        rest_pooled["n_errors"] / rest_pooled["n_occurrences"],
        np.nan,
    )

    merged = focus_rows.merge(
        rest_pooled[["word", "error_rate_rest", "n_occurrences", "n_errors"]],
        on="word",
        how="left",
        suffixes=("_focus", "_rest"),
    )
    merged = merged.rename(
        columns={
            "error_rate": "error_rate_focus",
            "n_occurrences_focus": "n_occurrences_focus",
            "n_errors_focus": "n_errors_focus",
            "n_occurrences_rest": "n_occurrences_rest",
            "n_errors_rest": "n_errors_rest",
        }
    )
    merged["focus_accent"] = focus
    merged["difference_focus_minus_rest"] = (
        merged["error_rate_focus"] - merged["error_rate_rest"]
    )

    columns = [
        "focus_accent",
        "word",
        "n_occurrences_focus",
        "n_errors_focus",
        "error_rate_focus",
        "n_occurrences_rest",
        "n_errors_rest",
        "error_rate_rest",
        "difference_focus_minus_rest",
        "missed_rate",
        "substituted_rate",
    ]
    columns = [column for column in columns if column in merged.columns]
    ranked = merged.sort_values(
        ["error_rate_focus", "difference_focus_minus_rest"],
        ascending=[False, False],
        ignore_index=True,
    )
    return ranked.loc[: top_n - 1, columns]


def wrong_language_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count wrong-language-output clips per accent.

    Args:
        frame: Per-clip results.

    Returns:
        One row per accent with ``n_clips``, ``n_wrong_language``, ``share``
        and the scripts observed.
    """
    rows: list[dict[str, object]] = []
    for accent, group in frame.groupby("accent", sort=True):
        flagged = group[group["wrong_language_output"].astype(bool)]
        scripts = sorted(set(flagged.get("dominant_script", pd.Series(dtype=str)).tolist()))
        rows.append(
            {
                "accent": accent,
                "n_clips": int(len(group)),
                "n_wrong_language": int(len(flagged)),
                "share_wrong_language": len(flagged) / len(group) if len(group) else np.nan,
                "scripts_observed": "; ".join(scripts),
            }
        )
    return pd.DataFrame(rows)
