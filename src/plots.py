"""Figures for the write-up, written as PNGs to ``results/figures/``.

All three figures are static report images, so they follow the print-safe half
of the house style: one axis per chart, categorical hues assigned in fixed slot
order (never cycled), recessive grid and spines, ink-coloured text, a legend
whenever more than one series shares an axis, and a 2px surface gap between
touching fills.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no interactive display on a headless/Windows run

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend choice
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FOCUS_ACCENT, SEED  # noqa: E402

LOGGER = logging.getLogger(__name__)

#: Categorical slots in fixed order (light-mode steps of the reference palette).
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dcdbd6"

#: Width in points of the surface-coloured gap between touching fills.
SPACER_PT: float = 2.0

FIGURE_DPI: int = 200


def _style_axes(ax: plt.Axes, ylabel: str, xlabel: str, title: str) -> None:
    """Apply the shared recessive-chrome styling to one axes object."""
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _color_for(index: int) -> str:
    """Return the categorical colour for slot ``index`` (slots are not cycled)."""
    if index < len(SERIES_COLORS):
        return SERIES_COLORS[index]
    return "#7a7974"  # "Other": past slot 8 identity is carried by the label


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write ``fig`` to ``path`` and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)
    return path


def plot_wer_boxplot(
    frame: pd.DataFrame,
    out_path: Path,
    seed: int = SEED,
) -> Path:
    """Boxplot of per-clip WER by accent, with the individual clips overlaid.

    Args:
        frame: Per-clip results with ``accent`` and ``wer`` columns.
        out_path: Destination PNG path.
        seed: Seed for the horizontal jitter of the overlaid points.

    Returns:
        The path written.
    """
    accents = sorted(frame["accent"].unique())
    samples = [frame.loc[frame["accent"] == a, "wer"].astype(float).to_numpy() for a in accents]

    fig, ax = plt.subplots(figsize=(1.6 * max(len(accents), 3) + 2.5, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    boxes = ax.boxplot(
        samples,
        positions=range(1, len(accents) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT_PRIMARY, "linewidth": 2.0},
        whiskerprops={"color": GRID, "linewidth": 1.2},
        capprops={"color": GRID, "linewidth": 1.2},
    )
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(_color_for(index))
        patch.set_alpha(0.22)
        patch.set_edgecolor(_color_for(index))
        patch.set_linewidth(1.6)

    rng = np.random.default_rng(seed)
    for index, values in enumerate(samples):
        if values.size == 0:
            continue
        x = index + 1 + rng.uniform(-0.16, 0.16, size=values.size)
        ax.scatter(
            x,
            values,
            s=34,
            color=_color_for(index),
            edgecolor=SURFACE,  # 2px surface ring keeps overlapping points readable
            linewidth=SPACER_PT / 2,
            zorder=3,
            alpha=0.95,
        )

    ax.set_xticks(range(1, len(accents) + 1))
    # Group size is direct-labelled under each box rather than in a caption, so
    # it cannot collide with the axes and reads with the box it belongs to.
    ax.set_xticklabels([f"{a}\nn={len(s)}" for a, s in zip(accents, samples)])
    ax.set_ylim(bottom=0)
    _style_axes(
        ax,
        ylabel="Word error rate (lower is better)",
        xlabel="Accent group",
        title="Whisper large-v3 word error rate by accent group",
    )
    return _save(fig, Path(out_path))


def plot_error_types(breakdown: pd.DataFrame, out_path: Path) -> Path:
    """Stacked bars showing the mix of substitutions, deletions and insertions.

    Args:
        breakdown: Output of :func:`src.analysis.error_type_breakdown`.
        out_path: Destination PNG path.

    Returns:
        The path written.
    """
    table = breakdown.sort_values("accent", ignore_index=True)
    accents = table["accent"].tolist()
    kinds = ("substitutions", "deletions", "insertions")
    labels = {
        "substitutions": "Substitutions",
        "deletions": "Deletions",
        "insertions": "Insertions",
    }

    fig, ax = plt.subplots(figsize=(1.5 * max(len(accents), 3) + 3.0, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    positions = np.arange(len(accents), dtype=float)
    bottom = np.zeros(len(accents), dtype=float)
    for index, kind in enumerate(kinds):
        heights = table[f"share_{kind}"].astype(float).fillna(0.0).to_numpy()
        ax.bar(
            positions,
            heights,
            bottom=bottom,
            width=0.6,
            color=_color_for(index),
            edgecolor=SURFACE,  # surface gap between stacked segments
            linewidth=SPACER_PT,
            label=labels[kind],
        )
        for x, height, base in zip(positions, heights, bottom):
            if height >= 0.08:  # label only segments with room, never every value
                ax.text(
                    x,
                    base + height / 2,
                    f"{height:.0%}",
                    ha="center",
                    va="center",
                    color=TEXT_PRIMARY,
                    fontsize=9,
                )
        bottom += heights

    # A group with no errors at all has no stack to draw; say so rather than
    # leaving a blank column the reader has to interpret.
    for x, total in zip(positions, table["total_errors"].astype(float).to_numpy()):
        if total == 0:
            ax.text(
                x,
                0.02,
                "no errors",
                ha="center",
                va="bottom",
                color=TEXT_SECONDARY,
                fontsize=9,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(accents, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    _style_axes(
        ax,
        ylabel="Share of all errors",
        xlabel="Accent group",
        title="Error type mix by accent group",
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              labelcolor=TEXT_SECONDARY, fontsize=9)
    return _save(fig, Path(out_path))


def plot_hard_words(
    hard_words: pd.DataFrame,
    out_path: Path,
    focus: str = FOCUS_ACCENT,
) -> Path:
    """Grouped bars comparing per-word error rates: focus accent vs all others.

    Args:
        hard_words: Output of :func:`src.analysis.hardest_words`.
        out_path: Destination PNG path.
        focus: Name of the focus accent, used in the legend.

    Returns:
        The path written.
    """
    table = hard_words.sort_values("error_rate_focus", ascending=True, ignore_index=True)
    words = table["word"].tolist()
    focus_rates = table["error_rate_focus"].astype(float).to_numpy()
    rest_rates = table["error_rate_rest"].astype(float).fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * max(len(words), 6) + 2.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    positions = np.arange(len(words), dtype=float)
    height = 0.38
    ax.barh(
        positions + height / 2,
        focus_rates,
        height=height,
        color=_color_for(0),
        edgecolor=SURFACE,
        linewidth=SPACER_PT / 2,
        label=focus,
    )
    ax.barh(
        positions - height / 2,
        rest_rates,
        height=height,
        color=_color_for(1),
        edgecolor=SURFACE,
        linewidth=SPACER_PT / 2,
        label="all other accents pooled",
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(words)
    # Bars start at zero; the upper bound follows the data so low rates are
    # still readable, with a floor so a near-empty chart is not magnified.
    highest = float(np.nanmax(np.concatenate([focus_rates, rest_rates]))) if words else 0.0
    ax.set_xlim(0, min(1.0, max(0.1, highest * 1.15)))
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    _style_axes(
        ax,
        ylabel="Reference word",
        xlabel="Share of clips where the word was missed or substituted",
        title=f"Hardest reference words for the {focus} group",
    )
    ax.grid(axis="y", visible=False)
    # Below the axes, so the legend can never sit on top of a long bar.
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        labelcolor=TEXT_SECONDARY,
        fontsize=9,
    )
    return _save(fig, Path(out_path))


def make_all_figures(
    results: pd.DataFrame,
    breakdown: pd.DataFrame,
    hard_words: pd.DataFrame,
    figures_dir: Path,
    focus: str = FOCUS_ACCENT,
) -> list[Path]:
    """Render every figure, skipping any whose input table is empty.

    Args:
        results: Per-clip results.
        breakdown: Error-type breakdown table.
        hard_words: Hardest-words table.
        figures_dir: Folder to write the PNGs into.
        focus: Name of the focus accent.

    Returns:
        Paths of the figures actually written.
    """
    figures_dir = Path(figures_dir)
    written: list[Path] = []

    if results.empty:
        LOGGER.warning("No per-clip results; no figures written.")
        return written

    written.append(plot_wer_boxplot(results, figures_dir / "wer_by_accent.png"))
    if breakdown.empty:
        LOGGER.warning("Empty error-type breakdown; skipping that figure.")
    else:
        written.append(plot_error_types(breakdown, figures_dir / "error_types_by_accent.png"))
    if hard_words.empty:
        LOGGER.warning("Empty hard-words table; skipping that figure.")
    else:
        written.append(
            plot_hard_words(hard_words, figures_dir / "hardest_words.png", focus=focus)
        )
    return written
