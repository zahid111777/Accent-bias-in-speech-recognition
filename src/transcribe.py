"""Transcribe audio clips with Whisper through the Hugging Face Inference API.

Every successful call is appended to ``results/transcripts.jsonl`` as one JSON
object per line. That file is the cache: a rerun reads it first and only calls
the API for clips it has not seen, so repeated analysis runs cost nothing.

No model is downloaded or run locally; the machine only needs an internet
connection and an ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from .config import AUDIO_EXTENSIONS, DEFAULT_MODEL, load_dotenv

LOGGER = logging.getLogger(__name__)

#: Seconds to wait between successive API calls, to stay a polite client.
CALL_INTERVAL_SECONDS: float = 1.0

#: How many times a single clip is retried before it is recorded as failed.
MAX_ATTEMPTS: int = 5

#: First backoff delay; doubles on each retry.
INITIAL_BACKOFF_SECONDS: float = 4.0

#: Phrases that mark an error as transient and worth retrying.
RETRYABLE_MARKERS: tuple[str, ...] = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "currently loading",
    "service unavailable",
    "bad gateway",
    "connection",
    "overloaded",
)

#: HTTP status codes worth retrying.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

#: HTTP status codes that will never succeed on retry: a missing or rejected
#: token, exhausted credits, a wrong model id. Retrying these wastes time and,
#: for 402, can look like abuse.
FATAL_STATUS: frozenset[int] = frozenset({400, 401, 402, 403, 404})

#: Matches an HTTP status code as a standalone token, so a request id such as
#: "Root=1-6ab01117-4e2d355e230c11fd76c31269" cannot be mistaken for one.
_STATUS_PATTERN = re.compile(r"(?<!\w)(\d{3})(?!\w)")


class TranscriptionError(RuntimeError):
    """Raised for setup problems such as a missing token or empty audio folder."""


#: Transcription backends selectable from the command line.
BACKENDS: tuple[str, ...] = ("hf_api", "local")

#: Default backend: the hosted API, which needs no GPU and no local model.
DEFAULT_BACKEND: str = "hf_api"


class ASRClient(Protocol):
    """Minimal interface this module needs from an inference client.

    Matching :class:`huggingface_hub.InferenceClient` lets the tests substitute
    a stub without touching the network, and lets the local GPU backend drop in
    behind the same call.
    """

    def automatic_speech_recognition(self, audio: str, *, model: str) -> Any:
        """Transcribe the audio at ``audio`` with ``model``."""


class LocalASRClient:
    """Run Whisper locally through ``transformers``, on a GPU when one exists.

    Exposes the same ``automatic_speech_recognition`` call as the hosted client
    so the transcription loop, the cache and the whole analysis are unchanged.

    ``torch`` and ``transformers`` are imported lazily, inside ``__init__``, so
    a laptop that has neither installed can still use the ``hf_api`` backend
    and run the test suite.

    The model is loaded once, when the client is constructed, and reused for
    every clip. Loading per clip would dominate the runtime.

    Attributes:
        model: The model id this client was built for.
        device: ``0`` for the first CUDA device, ``-1`` for CPU.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        chunk_length_s: int = 30,
    ) -> None:
        """Build the pipeline once.

        Args:
            model: Whisper model id to load.
            chunk_length_s: Window used for long-form audio.

        Raises:
            TranscriptionError: If torch or transformers are not installed.
        """
        try:
            import torch
            from transformers import pipeline
        except ImportError as error:  # pragma: no cover - environment dependent
            raise TranscriptionError(
                "The 'local' backend needs torch and transformers, which are "
                "not installed. Either install them:\n"
                "    pip install torch transformers accelerate\n"
                "or use the hosted API instead with --backend hf_api.\n"
                "See colab/run_on_colab.ipynb for a free GPU setup."
            ) from error

        self.model = model
        cuda = torch.cuda.is_available()
        self.device = 0 if cuda else -1
        dtype = torch.float16 if cuda else torch.float32

        if cuda:
            LOGGER.info(
                "Loading %s on GPU (%s), float16", model, torch.cuda.get_device_name(0)
            )
        else:
            LOGGER.warning(
                "No CUDA device found: loading %s on CPU in float32. This works "
                "but is slow for a full sample; see colab/run_on_colab.ipynb.",
                model,
            )

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=self.device,
            torch_dtype=dtype,
            chunk_length_s=chunk_length_s,
        )
        LOGGER.info("Model loaded once and reused for every clip")

    def automatic_speech_recognition(self, audio: str, *, model: str) -> dict[str, Any]:
        """Transcribe one audio file.

        No language is forced: Whisper auto-detects, exactly as the hosted API
        does. That matters here, because a clip transcribed into the speaker's
        first language instead of English is one of the failure modes this
        study measures, and forcing ``language="en"`` would hide it.

        Args:
            audio: Path to the audio file.
            model: Ignored; the pipeline is already bound to a model. Present
                so this class satisfies :class:`ASRClient`.

        Returns:
            A dict with a ``text`` key, matching the hosted client's shape.
        """
        if model and model != self.model:
            LOGGER.debug(
                "Ignoring per-call model %r; this client is loaded with %r",
                model,
                self.model,
            )
        return self._pipe(str(audio), generate_kwargs={"task": "transcribe"})


def get_backend_client(
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_MODEL,
) -> ASRClient:
    """Build the client for the requested backend.

    Args:
        backend: Either ``hf_api`` (hosted, metered) or ``local`` (GPU/CPU).
        model: Model id.

    Returns:
        A client satisfying :class:`ASRClient`.

    Raises:
        TranscriptionError: For an unknown backend, a missing token, or missing
            local dependencies.
    """
    if backend == "hf_api":
        return get_client(model)
    if backend == "local":
        return LocalASRClient(model)
    raise TranscriptionError(
        f"Unknown backend {backend!r}. Choose one of: {', '.join(BACKENDS)}."
    )


@dataclass(frozen=True)
class Clip:
    """One audio file queued for transcription.

    Attributes:
        accent: Accent group name, taken from the parent folder.
        path: Path to the audio file.
        key: Cache key, the POSIX path relative to the audio root.
    """

    accent: str
    path: Path
    key: str


def get_client(model: str = DEFAULT_MODEL) -> ASRClient:
    """Build a Hugging Face inference client from ``HF_TOKEN``.

    Args:
        model: Model id, used only for the log line.

    Returns:
        A configured :class:`huggingface_hub.InferenceClient`.

    Raises:
        TranscriptionError: If ``HF_TOKEN`` is unset or empty.
    """
    # A real environment variable wins; .env fills in when one is not exported.
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise TranscriptionError(
            "HF_TOKEN is not set. Create a token at "
            "https://huggingface.co/settings/tokens, then either write it into "
            "a .env file in the repository root:\n"
            "    HF_TOKEN=hf_your_token_here\n"
            "or set it as an environment variable on Windows:\n"
            '    setx HF_TOKEN "hf_your_token_here"\n'
            "(setx only affects NEW terminals, so reopen the shell afterwards).\n"
            ".env is git-ignored and is never read from anywhere else."
        )

    from huggingface_hub import InferenceClient  # imported lazily for fast tests

    LOGGER.info("Using Hugging Face Inference API with model %s", model)
    return InferenceClient(provider="auto", api_key=token)


def discover_clips(audio_dir: Path, interleave: bool = True) -> list[Clip]:
    """Collect every audio file under ``audio_dir/<accent>/``.

    Clips are returned round-robin across accents by default: one from the
    first accent, one from the second, and so on. A run that stops early —
    exhausted API credits, a rate limit, an interrupted session — then leaves a
    roughly balanced sample across every group instead of a complete first
    accent and nothing for the last. With ``--limit`` it also means a cheap run
    covers all groups rather than only the alphabetically first one.

    Ordering is deterministic either way, so the cache stays stable.

    Args:
        audio_dir: Root folder produced by :mod:`src.prepare_data`.
        interleave: If False, return clips grouped by accent instead.

    Returns:
        Clips in a deterministic order.

    Raises:
        TranscriptionError: If the folder is missing or holds no audio.
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise TranscriptionError(
            f"Audio folder not found: {audio_dir}. Run 'python -m src.prepare_data' first."
        )

    by_accent: dict[str, list[Clip]] = {}
    for accent_dir in sorted(p for p in audio_dir.iterdir() if p.is_dir()):
        files = [
            p
            for p in sorted(accent_dir.iterdir())
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]
        if not files:
            LOGGER.warning("Accent folder %s contains no audio files", accent_dir.name)
            continue
        by_accent[accent_dir.name] = [
            Clip(
                accent=accent_dir.name,
                path=path,
                key=path.relative_to(audio_dir).as_posix(),
            )
            for path in files
        ]

    clips: list[Clip] = []
    if interleave:
        for position in range(max((len(v) for v in by_accent.values()), default=0)):
            for accent in by_accent:  # insertion order is the sorted accent order
                if position < len(by_accent[accent]):
                    clips.append(by_accent[accent][position])
    else:
        for accent_clips in by_accent.values():
            clips.extend(accent_clips)

    if not clips:
        raise TranscriptionError(
            f"No audio files found under {audio_dir}. Expected {audio_dir}/<accent>/*.mp3"
        )
    LOGGER.info(
        "Found %d clips across %d accents",
        len(clips),
        len({clip.accent for clip in clips}),
    )
    return clips


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Read ``transcripts.jsonl`` into a dict keyed by clip key.

    Malformed lines are skipped with a warning so one bad append cannot make the
    whole cache unreadable.

    Args:
        cache_path: Path to the JSONL cache.

    Returns:
        Mapping of clip key -> cached record. Empty if the file is absent.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed cache line %d in %s", line_no, cache_path)
                continue
            key = record.get("key") or record.get("file")
            if key:
                cache[str(key)] = record
    LOGGER.info("Loaded %d cached transcripts from %s", len(cache), cache_path)
    return cache


def _append_record(cache_path: Path, record: dict[str, Any]) -> None:
    """Append one record to the JSONL cache, flushing immediately."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_text(response: Any) -> str:
    """Pull the transcript string out of whatever the client returned.

    The API has returned plain strings, dicts with a ``text`` key and objects
    with a ``.text`` attribute across versions, so all three are accepted.

    Args:
        response: Raw return value of ``automatic_speech_recognition``.

    Returns:
        The transcript, stripped. Empty string if none could be found.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        return str(response.get("text", "")).strip()
    text = getattr(response, "text", None)
    return str(text).strip() if text is not None else ""


def _is_retryable(error: Exception) -> bool:
    """Decide whether an exception is transient enough to retry.

    A status code that can never succeed on retry (a rejected token, exhausted
    credits) wins over every other signal, so the run fails fast on those
    instead of sleeping through five pointless attempts per clip.

    Args:
        error: The exception raised by the inference client.

    Returns:
        True if the call is worth repeating.
    """
    message = f"{type(error).__name__}: {error}"
    codes = {int(match) for match in _STATUS_PATTERN.findall(message)}
    if codes & FATAL_STATUS:
        return False
    if codes & RETRYABLE_STATUS:
        return True
    lowered = message.lower()
    return any(marker in lowered for marker in RETRYABLE_MARKERS)


def _fatal_status(error: Exception) -> bool:
    """Return True for an error that will recur identically on every clip.

    Args:
        error: The exception raised by the inference client.

    Returns:
        True for a rejected token, exhausted credits or an unknown model.
    """
    message = f"{type(error).__name__}: {error}"
    codes = {int(match) for match in _STATUS_PATTERN.findall(message)}
    return bool(codes & FATAL_STATUS)


def transcribe_clip(
    client: ASRClient,
    clip: Clip,
    model: str = DEFAULT_MODEL,
    max_attempts: int = MAX_ATTEMPTS,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    sleeper: Any = time.sleep,
) -> str:
    """Transcribe one clip, retrying transient failures with exponential backoff.

    Args:
        client: Any object exposing ``automatic_speech_recognition``.
        clip: The clip to transcribe.
        model: Model id passed to the API.
        max_attempts: Total attempts, including the first.
        initial_backoff: Seconds to wait after the first failure.
        sleeper: Injection point for sleeping, so tests run instantly.

    Returns:
        The transcript text.

    Raises:
        Exception: The last error, if every attempt failed or the error was
            not transient.
    """
    backoff = initial_backoff
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.automatic_speech_recognition(str(clip.path), model=model)
            return _extract_text(response)
        except Exception as error:  # noqa: BLE001 - surfaced after the retries
            if attempt >= max_attempts or not _is_retryable(error):
                raise
            LOGGER.warning(
                "Attempt %d/%d failed for %s (%s); retrying in %.0fs",
                attempt,
                max_attempts,
                clip.key,
                error,
                backoff,
            )
            sleeper(backoff)
            backoff *= 2
    raise RuntimeError("unreachable: retry loop exited without returning")


def resolve_transcripts_path(out_dir: Path, transcripts: Path | None = None) -> Path:
    """Decide where the transcript cache lives.

    Args:
        out_dir: Results folder.
        transcripts: Explicit path, or ``None`` for ``out_dir/transcripts.jsonl``.

    Returns:
        The cache path.
    """
    if transcripts is not None:
        return Path(transcripts)
    return Path(out_dir) / "transcripts.jsonl"


def transcribe_all(
    audio_dir: Path,
    out_dir: Path,
    client: ASRClient | None = None,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
    call_interval: float = CALL_INTERVAL_SECONDS,
    sleeper: Any = time.sleep,
    backend: str = DEFAULT_BACKEND,
    transcripts: Path | None = None,
) -> list[dict[str, Any]]:
    """Transcribe every clip under ``audio_dir``, caching as it goes.

    Args:
        audio_dir: Root folder of ``<accent>/<clip>`` audio.
        out_dir: Results folder; the default cache is ``out_dir/transcripts.jsonl``.
        client: Client to use; built from ``backend`` when ``None``.
        model: Model id.
        limit: Transcribe at most this many *new* clips (cheap test runs).
        call_interval: Seconds to sleep between successive calls.
        sleeper: Injection point for sleeping, so tests run instantly.
        backend: ``hf_api`` or ``local``; selects the client when none is given.
        transcripts: Explicit cache path, so a local-backend run can write to a
            separate file rather than mixing backends in one cache.

    Returns:
        All records for the discovered clips, cached and freshly fetched alike.
    """
    audio_dir, out_dir = Path(audio_dir), Path(out_dir)
    cache_path = resolve_transcripts_path(out_dir, transcripts)

    clips = discover_clips(audio_dir)
    cache = load_cache(cache_path)
    _warn_on_mixed_backends(cache, backend, cache_path)

    pending = [clip for clip in clips if clip.key not in cache]
    skipped = len(clips) - len(pending)
    if skipped:
        LOGGER.info("Skipping %d clips already present in the cache", skipped)
    if limit is not None and len(pending) > limit:
        LOGGER.info("--limit %d: transcribing %d of %d new clips", limit, limit, len(pending))
        pending = pending[:limit]

    if pending and client is None:
        client = get_backend_client(backend, model)

    failures: list[str] = []
    for position, clip in enumerate(pending, start=1):
        LOGGER.info("[%d/%d] %s", position, len(pending), clip.key)
        try:
            text = transcribe_clip(client, clip, model=model, sleeper=sleeper)
        except Exception as error:  # noqa: BLE001 - one bad clip must not stop the run
            LOGGER.error("Failed on %s: %s", clip.key, error)
            failures.append(clip.key)
            if _fatal_status(error):
                # An exhausted quota or a rejected token fails identically for
                # every remaining clip. Stop now rather than logging the same
                # error a hundred more times and burying it.
                LOGGER.error(
                    "Aborting after %d of %d clips: this error affects every "
                    "remaining call, so continuing cannot help. %d clips are "
                    "cached and will be skipped when you rerun.",
                    position,
                    len(pending),
                    len(cache),
                )
                break
            continue

        if not text:
            LOGGER.warning("Empty transcript returned for %s", clip.key)
        record = {
            "key": clip.key,
            "accent": clip.accent,
            "file": clip.path.as_posix(),
            "text": text,
            "model": model,
            "backend": backend,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _append_record(cache_path, record)
        cache[clip.key] = record

        if position < len(pending):
            sleeper(call_interval)

    if failures:
        LOGGER.warning(
            "%d clips failed and were not cached (rerun to retry): %s",
            len(failures),
            ", ".join(failures),
        )

    records = [cache[clip.key] for clip in clips if clip.key in cache]
    LOGGER.info(
        "Transcripts available for %d of %d clips (%d new this run)",
        len(records),
        len(clips),
        len(pending) - len(failures),
    )
    return records


def describe_backends(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count how many records came from each backend.

    Args:
        records: Cached transcription records.

    Returns:
        Mapping of backend name -> record count. Records written before the
        backend field existed count as ``hf_api``, which is what produced them.
    """
    counts: dict[str, int] = {}
    for record in records:
        name = str(record.get("backend") or "hf_api")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _warn_on_mixed_backends(
    cache: dict[str, dict[str, Any]],
    backend: str,
    cache_path: Path,
) -> None:
    """Warn loudly when a cache is about to hold more than one backend.

    Two backends are two different systems: a hosted provider's build of
    Whisper and a local one can differ in version, precision and decoding. WERs
    from a mix of the two are not comparable, and the difference would be
    silently attributed to accent.

    Args:
        cache: Records already in the cache.
        backend: The backend about to be used.
        cache_path: Cache location, named in the warning.
    """
    existing = describe_backends(cache.values())
    others = {name: count for name, count in existing.items() if name != backend}
    if others:
        summary = ", ".join(f"{count} from {name}" for name, count in sorted(others.items()))
        LOGGER.warning(
            "Cache %s already holds transcripts from another backend (%s) and "
            "you are now running %r. Mixing backends in one analysis is not "
            "valid: differences between them would be indistinguishable from "
            "accent effects. Use --transcripts to keep each backend in its own "
            "file, e.g. results/transcripts_local.jsonl.",
            cache_path,
            summary,
            backend,
        )


def records_for_clips(
    audio_dir: Path,
    out_dir: Path,
    transcripts: Path | None = None,
) -> list[dict[str, Any]]:
    """Return cached records for the clips currently under ``audio_dir``.

    Args:
        audio_dir: Root folder of ``<accent>/<clip>`` audio.
        out_dir: Results folder holding the default cache.
        transcripts: Explicit cache path, overriding the default.

    Returns:
        One record per clip that has a cached transcript.
    """
    clips = discover_clips(Path(audio_dir))
    cache = load_cache(resolve_transcripts_path(out_dir, transcripts))
    return [cache[clip.key] for clip in clips if clip.key in cache]


def iter_missing(clips: Iterable[Clip], cache: dict[str, dict[str, Any]]) -> list[Clip]:
    """Return the clips of ``clips`` that are absent from ``cache``.

    Args:
        clips: Discovered clips.
        cache: Cache mapping from :func:`load_cache`.

    Returns:
        Clips still needing an API call.
    """
    return [clip for clip in clips if clip.key not in cache]
