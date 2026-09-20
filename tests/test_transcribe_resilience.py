"""Tests for clip ordering and for how the client classifies API errors.

These cover the two failure modes seen on a real run: credits running out
partway through, and ordering that made the resulting partial sample useless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.transcribe import (
    Clip,
    _fatal_status,
    _is_retryable,
    discover_clips,
    transcribe_all,
    transcribe_clip,
)


@pytest.fixture()
def uneven_tree(tmp_path: Path) -> Path:
    """Three accents of very different sizes, like the real sample."""
    audio_dir = tmp_path / "audio"
    for accent, count in (("arabic", 4), ("english_us", 3), ("pakistani", 2)):
        (audio_dir / accent).mkdir(parents=True)
        for index in range(count):
            (audio_dir / accent / f"{accent}{index}.mp3").write_bytes(b"ID3 dummy")
    return audio_dir


def _noop_sleep(_seconds: float) -> None:
    """Sleep replacement that returns immediately."""


def test_clips_are_interleaved_across_accents(uneven_tree: Path) -> None:
    """A truncated run must still touch every accent."""
    clips = discover_clips(uneven_tree)
    assert [c.accent for c in clips[:3]] == ["arabic", "english_us", "pakistani"]
    # The first six clips cover all three groups evenly, not one group entirely.
    first_six = [c.accent for c in clips[:6]]
    assert first_six.count("arabic") == 2
    assert first_six.count("english_us") == 2
    assert first_six.count("pakistani") == 2


def test_interleaving_can_be_disabled(uneven_tree: Path) -> None:
    clips = discover_clips(uneven_tree, interleave=False)
    assert [c.accent for c in clips[:4]] == ["arabic"] * 4


def test_interleaving_keeps_every_clip_exactly_once(uneven_tree: Path) -> None:
    interleaved = discover_clips(uneven_tree)
    grouped = discover_clips(uneven_tree, interleave=False)
    assert sorted(c.key for c in interleaved) == sorted(c.key for c in grouped)
    assert len(interleaved) == 9


def test_interleaved_order_is_deterministic(uneven_tree: Path) -> None:
    assert [c.key for c in discover_clips(uneven_tree)] == [
        c.key for c in discover_clips(uneven_tree)
    ]


@pytest.mark.parametrize(
    ("message", "retryable"),
    [
        ("Client error '429 Too Many Requests'", True),
        ("Server error '503 Service Unavailable'", True),
        ("Read timed out", True),
        ("Client error '402 Payment Required' for url", False),
        ("Client error '401 Unauthorized'", False),
        ("Client error '404 Not Found'", False),
    ],
)
def test_error_classification(message: str, retryable: bool) -> None:
    assert _is_retryable(RuntimeError(message)) is retryable


def test_request_id_digits_are_not_read_as_status_codes() -> None:
    """A request id embedding '503' must not turn a fatal 402 into a retry."""
    error = RuntimeError(
        "Client error '402 Payment Required' for url "
        "(Request ID: Root=1-6ab01117-503a355e230c11fd76c31269)"
    )
    assert _is_retryable(error) is False
    assert _fatal_status(error) is True


def test_fatal_error_is_not_retried(tmp_path: Path) -> None:
    """A 402 must fail on the first attempt, not burn five."""
    calls: list[str] = []

    class BrokeClient:
        def automatic_speech_recognition(self, audio: str, *, model: str):
            calls.append(audio)
            raise RuntimeError("Client error '402 Payment Required' for url")

    clip = Clip(accent="a", path=tmp_path / "a" / "x.mp3", key="a/x.mp3")
    with pytest.raises(RuntimeError, match="402"):
        transcribe_clip(BrokeClient(), clip, sleeper=_noop_sleep, initial_backoff=0.0)
    assert len(calls) == 1


def test_run_aborts_on_exhausted_credits(uneven_tree: Path, tmp_path: Path, caplog) -> None:
    """One 402 must stop the run, not repeat for every remaining clip."""
    calls: list[str] = []

    class OutOfCreditClient:
        def automatic_speech_recognition(self, audio: str, *, model: str):
            calls.append(audio)
            raise RuntimeError(
                "Client error '402 Payment Required': "
                "You have depleted your monthly included credits."
            )

    with caplog.at_level("ERROR"):
        records = transcribe_all(
            uneven_tree, tmp_path / "results", client=OutOfCreditClient(), sleeper=_noop_sleep
        )

    assert records == []
    assert len(calls) == 1, "should abort after the first fatal error, not try all 9"
    assert any("Aborting after" in record.getMessage() for record in caplog.records)
