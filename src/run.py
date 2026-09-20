"""End-to-end pipeline: transcribe, score, analyse, plot.

Usage::

    python -m src.run --audio_dir audio --out results

Transcription is cached, so rerunning after a code change re-scores the existing
transcripts without spending a single API call. Use ``--limit`` for a cheap
first run and ``--skip_transcribe`` to re-analyse the cache only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from .analysis import (
    error_type_breakdown,
    hardest_words,
    pairwise_vs_focus,
    summarise_by_accent,
    word_error_rates,
    wrong_language_counts,
)
from .config import DEFAULT_MODEL, FOCUS_ACCENT, SEED, load_reference, setup_logging
from .metrics import score_records, summarise_two_ways
from .plots import make_all_figures
from .transcribe import (
    BACKENDS,
    DEFAULT_BACKEND,
    TranscriptionError,
    describe_backends,
    records_for_clips,
    transcribe_all,
)

LOGGER = logging.getLogger(__name__)


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    """Write ``frame`` to ``path``, logging what was written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("Wrote %-26s (%d rows) -> %s", label, len(frame), path)


def run_pipeline(
    audio_dir: Path,
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
    focus: str = FOCUS_ACCENT,
    seed: int = SEED,
    skip_transcribe: bool = False,
    backend: str = DEFAULT_BACKEND,
    transcripts: Path | None = None,
    client: object | None = None,
    sleeper: object | None = None,
    reference_path: Path | None = None,
) -> dict[str, object]:
    """Run the whole study and write every table and figure.

    Args:
        audio_dir: Folder of ``<accent>/<clip>`` audio.
        out_dir: Destination folder for CSVs, the transcript cache and figures.
        model: ASR model id.
        limit: Transcribe at most this many new clips.
        focus: Accent group compared against all others.
        seed: Seed for the bootstrap and the plot jitter.
        skip_transcribe: Re-use the cache and make no API calls.
        backend: ``hf_api`` (hosted) or ``local`` (GPU/CPU via transformers).
        transcripts: Explicit transcript cache path. Every downstream table
            and figure is built from whichever cache is named here.
        client: Optional ASR client; used by the mocked dry run and the tests.
        sleeper: Optional sleep function; used to make tests instant.
        reference_path: Optional override for ``reference.txt``.

    Returns:
        Dict of the tables produced, keyed by output name.

    Raises:
        TranscriptionError: If no transcripts are available to analyse.
    """
    audio_dir, out_dir = Path(audio_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(reference_path)

    if skip_transcribe:
        LOGGER.info("--skip_transcribe: using cached transcripts only")
        records = records_for_clips(audio_dir, out_dir, transcripts=transcripts)
    else:
        kwargs = {} if sleeper is None else {"sleeper": sleeper}
        records = transcribe_all(
            audio_dir=audio_dir,
            out_dir=out_dir,
            client=client,
            model=model,
            limit=limit,
            backend=backend,
            transcripts=transcripts,
            **kwargs,
        )

    if not records:
        raise TranscriptionError(
            "No transcripts available to analyse. Run without --skip_transcribe, "
            "and check the log for failed clips."
        )

    provenance = describe_backends(records)
    LOGGER.info(
        "Analysing %d transcripts produced by: %s",
        len(records),
        ", ".join(f"{count} x {name}" for name, count in sorted(provenance.items())),
    )
    if len(provenance) > 1:
        LOGGER.warning(
            "These transcripts come from more than one backend. WERs from "
            "different backends are not comparable, so this analysis is not "
            "valid as a between-accent comparison."
        )

    results = pd.DataFrame(score_records(records, reference))
    overall = summarise_two_ways(results.to_dict("records"))
    LOGGER.info(
        "Overall mean WER: %.4f over %d clips; %.4f over %d clips excluding "
        "%d wrong-language outputs",
        overall["mean_wer_all"],
        overall["n_all"],
        overall["mean_wer_clean"],
        overall["n_clean"],
        overall["n_flagged"],
    )

    summary = summarise_by_accent(results, seed=seed)
    pairwise = pairwise_vs_focus(results, focus=focus)
    breakdown = error_type_breakdown(results)
    per_word = word_error_rates(results, reference)
    hard = hardest_words(results, reference, focus=focus)
    wrong_language = wrong_language_counts(results)

    _write_csv(results, out_dir / "results.csv", "per-clip results")
    _write_csv(summary, out_dir / "summary_by_accent.csv", "accent summary")
    _write_csv(pairwise, out_dir / "pairwise_tests.csv", "pairwise tests")
    _write_csv(breakdown, out_dir / "error_types_by_accent.csv", "error-type breakdown")
    _write_csv(per_word, out_dir / "word_error_rates.csv", "word error rates")
    _write_csv(hard, out_dir / "hard_words.csv", "hardest words")
    _write_csv(wrong_language, out_dir / "wrong_language_counts.csv", "wrong-language counts")
    _write_csv(
        pd.DataFrame([overall]), out_dir / "overall_summary.csv", "overall summary"
    )
    _write_csv(
        pd.DataFrame(
            [{"backend": name, "n_clips": count} for name, count in sorted(provenance.items())]
        ),
        out_dir / "run_provenance.csv",
        "run provenance",
    )

    figures = make_all_figures(
        results=results,
        breakdown=breakdown,
        hard_words=hard,
        figures_dir=out_dir / "figures",
        focus=focus,
    )
    LOGGER.info("Wrote %d figures to %s", len(figures), out_dir / "figures")

    return {
        "results": results,
        "summary_by_accent": summary,
        "pairwise_tests": pairwise,
        "error_types_by_accent": breakdown,
        "word_error_rates": per_word,
        "hard_words": hard,
        "wrong_language_counts": wrong_language,
        "overall": overall,
        "figures": figures,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m src.run`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.run",
        description="Transcribe the accent sample with Whisper and analyse the errors.",
    )
    parser.add_argument("--audio_dir", type=Path, default=Path("audio"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--limit", type=int, default=None, help="Transcribe at most N new clips"
    )
    parser.add_argument(
        "--focus", default=FOCUS_ACCENT, help="Accent group compared against all others"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for bootstrap and jitter")
    parser.add_argument(
        "--skip_transcribe",
        action="store_true",
        help="Analyse the cached transcripts without calling the API",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help=(
            "hf_api: hosted Inference API, needs HF_TOKEN and credits. "
            "local: run Whisper here with transformers, needs torch and a GPU "
            "to be practical (see colab/run_on_colab.ipynb)."
        ),
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help=(
            "Transcript cache to read and write (default: <out>/transcripts.jsonl). "
            "Give each backend its own file, e.g. results/transcripts_local.jsonl; "
            "all tables and figures are built from whichever file is named here."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for ``python -m src.run``.

    Returns:
        Process exit code: 0 on success, 2 on a user-fixable error.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    setup_logging(args.verbose)
    try:
        run_pipeline(
            audio_dir=args.audio_dir,
            out_dir=args.out,
            model=args.model,
            limit=args.limit,
            focus=args.focus,
            seed=args.seed,
            skip_transcribe=args.skip_transcribe,
            backend=args.backend,
            transcripts=args.transcripts,
        )
    except (TranscriptionError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
