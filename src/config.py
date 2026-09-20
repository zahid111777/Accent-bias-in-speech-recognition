"""Shared constants and logging setup for the accent-bias pipeline."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

#: Single seed used for every stochastic step (sampling, bootstrap, jitter).
SEED: int = 12345

#: Default ASR model queried through the Hugging Face Inference API.
DEFAULT_MODEL: str = "openai/whisper-large-v3"

#: Repository root, i.e. the directory that contains ``src/``.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Elicitation paragraph every speaker in the Speech Accent Archive reads.
REFERENCE_PATH: Path = REPO_ROOT / "reference.txt"

#: Audio extensions considered when matching metadata rows to recordings.
AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

#: Groups smaller than this are kept but flagged as under-powered.
MIN_GROUP_SIZE: int = 10

#: Name of the accent group the study contrasts against every other group.
FOCUS_ACCENT: str = "pakistani"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once, with a compact single-line format.

    The console stream is put into replace-on-error mode first. A Windows
    console defaults to cp1252, which cannot encode Urdu, Arabic or Devanagari;
    without this, a single log line carrying such text would raise
    ``UnicodeEncodeError`` and abort a long transcription run.

    Args:
        verbose: If True, emit DEBUG records as well as INFO and above.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # already detached or not a TTY
                pass

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


#: Optional file holding local secrets, e.g. ``HF_TOKEN=hf_...``. Git-ignored.
ENV_PATH: Path = REPO_ROOT / ".env"


def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    A deliberately small stdlib reader rather than a dependency: it handles
    comments, blank lines, ``export`` prefixes and quoted values, which is all
    this project needs. A real environment variable wins over the file unless
    ``override`` is set, so a value exported in the shell still takes priority.

    Args:
        path: File to read; defaults to ``.env`` beside this repository's root.
        override: If True, values in the file replace existing environment
            variables instead of deferring to them.

    Returns:
        The keys that were set, mapped to their values. Empty if no file exists.
    """
    env_path = Path(path) if path is not None else ENV_PATH
    if not env_path.is_file():
        return {}

    applied: dict[str, str] = {}
    for line_no, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            logging.getLogger(__name__).warning(
                "Ignoring line %d of %s: no '=' found", line_no, env_path
            )
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    return applied


def load_reference(path: Path | None = None) -> str:
    """Read the reference paragraph from disk.

    Args:
        path: Optional override for the reference file location.

    Returns:
        The raw (un-normalised) reference paragraph.

    Raises:
        FileNotFoundError: If the reference file does not exist.
    """
    ref_path = Path(path) if path is not None else REFERENCE_PATH
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Reference paragraph not found at {ref_path}. "
            "It ships with the repository as reference.txt."
        )
    text = ref_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Reference paragraph at {ref_path} is empty.")
    return text
