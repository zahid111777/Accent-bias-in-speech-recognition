"""Per-clip word error rate, error counts and script-language checks.

Whisper occasionally transcribes accented English into the speaker's likely
first-language script (Urdu, Arabic, Devanagari). That is a qualitatively
different failure from mis-hearing a word, so those clips are flagged and every
headline number is reported twice: over all clips, and excluding flagged ones.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import jiwer

from .normalise import normalise, tokenise

LOGGER = logging.getLogger(__name__)

#: Share of letters that must be non-Latin before a clip is flagged.
NON_LATIN_THRESHOLD: float = 0.5

#: Minimum number of letters before the script check is meaningful.
MIN_LETTERS_FOR_SCRIPT_CHECK: int = 10

#: Unicode block prefixes reported alongside the flag, for the write-up.
SCRIPT_PREFIXES: tuple[str, ...] = ("ARABIC", "DEVANAGARI", "CYRILLIC", "CJK", "HAN", "HIRAGANA")


@dataclass(frozen=True)
class ClipMetrics:
    """Word-error statistics for a single clip.

    Attributes:
        wer: Word error rate, ``(S + D + I) / N`` against the reference.
        substitutions: Count of substituted reference words.
        deletions: Count of deleted (missed) reference words.
        insertions: Count of inserted hypothesis words.
        hits: Count of correctly recognised reference words.
        ref_words: Length of the normalised reference in words.
        hyp_words: Length of the normalised hypothesis in words.
    """

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int
    hyp_words: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dict, for DataFrame construction."""
        return asdict(self)


def dominant_script(text: str) -> str:
    """Return the most common Unicode script family among the letters in ``text``.

    Args:
        text: Any text, normalised or raw.

    Returns:
        A coarse script label such as ``"latin"``, ``"arabic"`` or
        ``"devanagari"``; ``"none"`` when the text holds no letters.
    """
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        label = name.split(" ")[0].lower()
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda item: item[1])[0]


def non_latin_share(text: str) -> float:
    """Compute the share of letters in ``text`` that are not Latin script.

    Args:
        text: Any text, normalised or raw.

    Returns:
        A value in ``[0, 1]``; ``0.0`` when the text holds no letters.
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    non_latin = 0
    for char in letters:
        try:
            name = unicodedata.name(char)
        except ValueError:
            non_latin += 1
            continue
        if not name.startswith("LATIN"):
            non_latin += 1
    return non_latin / len(letters)


def is_wrong_language_output(
    text: str,
    threshold: float = NON_LATIN_THRESHOLD,
    min_letters: int = MIN_LETTERS_FOR_SCRIPT_CHECK,
) -> bool:
    """Flag a transcript that came back mostly in a non-Latin script.

    Args:
        text: The transcript, raw or normalised.
        threshold: Share of non-Latin letters above which the clip is flagged.
        min_letters: Below this many letters the check is skipped, because a
            two-character transcript says nothing about the model's language.

    Returns:
        True if the transcript is mostly non-Latin script.
    """
    letters = [char for char in text if char.isalpha()]
    if len(letters) < min_letters:
        return False
    return non_latin_share(text) > threshold


def compute_clip_metrics(reference: str, hypothesis: str) -> ClipMetrics:
    """Score one hypothesis against the reference paragraph.

    Both sides are normalised first (see :mod:`src.normalise`). An empty
    hypothesis is scored directly as "everything deleted" rather than handed to
    jiwer, which rejects empty input.

    Args:
        reference: The reference paragraph, raw or normalised.
        hypothesis: The ASR transcript, raw or normalised.

    Returns:
        The clip's :class:`ClipMetrics`.

    Raises:
        ValueError: If the reference contains no words.
    """
    ref_tokens = tokenise(reference)
    hyp_tokens = tokenise(hypothesis)
    if not ref_tokens:
        raise ValueError("Reference text contains no words after normalisation.")

    if not hyp_tokens:
        return ClipMetrics(
            wer=1.0,
            substitutions=0,
            deletions=len(ref_tokens),
            insertions=0,
            hits=0,
            ref_words=len(ref_tokens),
            hyp_words=0,
        )

    output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    return ClipMetrics(
        wer=float(output.wer),
        substitutions=int(output.substitutions),
        deletions=int(output.deletions),
        insertions=int(output.insertions),
        hits=int(output.hits),
        ref_words=len(ref_tokens),
        hyp_words=len(hyp_tokens),
    )


def word_outcomes(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    """Label every reference word as correct, substituted or missed.

    Insertions are ignored here because they do not correspond to any reference
    word; they are counted separately in :class:`ClipMetrics`.

    Args:
        reference: The reference paragraph, raw or normalised.
        hypothesis: The ASR transcript, raw or normalised.

    Returns:
        One ``(reference_word, outcome)`` pair per reference word, in order,
        where outcome is ``"correct"``, ``"substituted"`` or ``"missed"``.
    """
    ref_tokens = tokenise(reference)
    hyp_tokens = tokenise(hypothesis)
    if not ref_tokens:
        return []
    if not hyp_tokens:
        return [(word, "missed") for word in ref_tokens]

    output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    outcomes: list[tuple[str, str]] = [(word, "missed") for word in ref_tokens]
    label_by_type = {
        "equal": "correct",
        "substitute": "substituted",
        "delete": "missed",
    }
    for chunk in output.alignments[0]:
        label = label_by_type.get(chunk.type)
        if label is None:  # insertions touch no reference word
            continue
        for position in range(chunk.ref_start_idx, chunk.ref_end_idx):
            if position < len(outcomes):
                outcomes[position] = (ref_tokens[position], label)
    return outcomes


def score_records(
    records: Iterable[dict[str, Any]],
    reference: str,
) -> list[dict[str, Any]]:
    """Score a sequence of transcription records.

    Args:
        records: Dicts with at least ``accent``, ``file``/``key`` and ``text``.
        reference: The reference paragraph.

    Returns:
        One row per record, carrying the metadata, the normalised transcript,
        the clip metrics and the ``wrong_language_output`` flag.
    """
    reference_normalised = normalise(reference)
    rows: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get("text", "") or "")
        normalised = normalise(text)
        metrics = compute_clip_metrics(reference_normalised, normalised)
        row: dict[str, Any] = {
            "key": record.get("key", record.get("file", "")),
            "accent": record.get("accent", "unknown"),
            "file": record.get("file", ""),
            "text_raw": text,
            "text_normalised": normalised,
            "wrong_language_output": is_wrong_language_output(text),
            "dominant_script": dominant_script(text),
            "non_latin_share": round(non_latin_share(text), 4),
        }
        row.update(metrics.as_dict())
        rows.append(row)

    flagged = sum(1 for row in rows if row["wrong_language_output"])
    LOGGER.info(
        "Scored %d clips; %d flagged as wrong-language output", len(rows), flagged
    )
    return rows


def summarise_two_ways(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Report the overall mean WER with and without flagged clips.

    Args:
        rows: Rows produced by :func:`score_records`.

    Returns:
        Dict with ``n_all``, ``mean_wer_all``, ``n_clean``, ``mean_wer_clean``
        and ``n_flagged``. Means are ``float('nan')`` when the subset is empty.
    """
    all_wers = [float(row["wer"]) for row in rows]
    clean_wers = [float(row["wer"]) for row in rows if not row["wrong_language_output"]]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "n_all": len(all_wers),
        "mean_wer_all": mean(all_wers),
        "n_clean": len(clean_wers),
        "mean_wer_clean": mean(clean_wers),
        "n_flagged": len(all_wers) - len(clean_wers),
    }
