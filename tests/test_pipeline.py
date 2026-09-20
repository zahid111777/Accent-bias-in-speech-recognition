"""Caching, retry and end-to-end tests, with the Inference API mocked out.

Nothing here touches the network: a stub client stands in for
:class:`huggingface_hub.InferenceClient`, and sleeping is injected so retries
and the 1s call interval cost no wall-clock time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_reference
from src.run import run_pipeline
from src.transcribe import (
    Clip,
    TranscriptionError,
    discover_clips,
    get_client,
    load_cache,
    transcribe_all,
    transcribe_clip,
)

REFERENCE = load_reference()

#: Three toy clips: one near-perfect, one degraded, one returned in Urdu script.
FAKE_TRANSCRIPTS = {
    "english_us/us1.mp3": REFERENCE,
    "pakistani/pk1.mp3": (
        "Please call Stella asked her to bring this thinks with her from the store "
        "six spoon of fresh snow peas five thick slap of blue cheese and maybe a "
        "snake for her brother Bob"
    ),
    "pakistani/pk2.mp3": "اسٹیلا کو فون کریں اور اس سے کہیں کہ وہ یہ چیزیں دکان سے لے آئے",
}


class StubClient:
    """Stand-in for ``InferenceClient`` that replays canned transcripts.

    Attributes:
        calls: Paths passed to the client, in order, so tests can assert on
            how many API calls the pipeline would have made.
    """

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[str] = []
        self._fail_times = fail_times

    def automatic_speech_recognition(self, audio: str, *, model: str) -> dict[str, str]:
        """Return the canned transcript for ``audio``."""
        self.calls.append(audio)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("429 Too Many Requests: rate limit exceeded")
        key = "/".join(Path(audio).parts[-2:])
        return {"text": FAKE_TRANSCRIPTS.get(key, "")}


@pytest.fixture()
def audio_tree(tmp_path: Path) -> Path:
    """Create ``audio/<accent>/<clip>.mp3`` dummy files for the three clips."""
    audio_dir = tmp_path / "audio"
    for key in FAKE_TRANSCRIPTS:
        path = audio_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ID3 dummy audio")
    return audio_dir


def _noop_sleep(_seconds: float) -> None:
    """Sleep replacement that returns immediately."""


def test_discover_clips_sorted_by_accent_then_name(audio_tree: Path) -> None:
    clips = discover_clips(audio_tree)
    assert [clip.key for clip in clips] == [
        "english_us/us1.mp3",
        "pakistani/pk1.mp3",
        "pakistani/pk2.mp3",
    ]
    assert {clip.accent for clip in clips} == {"english_us", "pakistani"}


def test_discover_clips_rejects_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="Audio folder not found"):
        discover_clips(tmp_path / "nope")


def test_get_client_without_token_explains_both_options(monkeypatch, tmp_path: Path) -> None:
    """With no token anywhere, the error names .env and setx."""
    # Point the loader at a non-existent file so a real .env in the repository
    # root cannot satisfy the lookup and mask the error being tested.
    monkeypatch.setattr("src.config.ENV_PATH", tmp_path / "absent.env")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(TranscriptionError) as excinfo:
        get_client()
    message = str(excinfo.value)
    assert ".env" in message and "setx HF_TOKEN" in message


def test_get_client_reads_token_from_dotenv(monkeypatch, tmp_path: Path) -> None:
    """A token present only in .env is enough to build a client."""
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_dotenv_token\n", encoding="utf-8")
    monkeypatch.setattr("src.config.ENV_PATH", env_file)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    captured: dict[str, object] = {}

    class FakeInferenceClient:
        def __init__(self, *, provider: str, api_key: str) -> None:
            captured["provider"] = provider
            captured["api_key"] = api_key

    monkeypatch.setattr("huggingface_hub.InferenceClient", FakeInferenceClient)
    get_client()
    assert captured == {"provider": "auto", "api_key": "hf_dotenv_token"}


def test_transcribe_clip_retries_transient_errors(tmp_path: Path) -> None:
    client = StubClient(fail_times=2)
    clip = Clip(accent="pakistani", path=tmp_path / "pakistani" / "pk1.mp3", key="pakistani/pk1.mp3")
    text = transcribe_clip(client, clip, sleeper=_noop_sleep, initial_backoff=0.0)
    assert len(client.calls) == 3
    assert text.startswith("Please call Stella")


def test_transcribe_clip_gives_up_after_max_attempts(tmp_path: Path) -> None:
    client = StubClient(fail_times=99)
    clip = Clip(accent="a", path=tmp_path / "a" / "x.mp3", key="a/x.mp3")
    with pytest.raises(RuntimeError, match="429"):
        transcribe_clip(client, clip, max_attempts=3, sleeper=_noop_sleep, initial_backoff=0.0)
    assert len(client.calls) == 3


def test_cache_is_written_and_reused(audio_tree: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "results"

    first_client = StubClient()
    records = transcribe_all(audio_tree, out_dir, client=first_client, sleeper=_noop_sleep)
    assert len(records) == 3
    assert len(first_client.calls) == 3

    cache_path = out_dir / "transcripts.jsonl"
    lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first_record = json.loads(lines[0])
    assert set(first_record) >= {"key", "accent", "file", "text", "timestamp"}

    # A rerun must spend no API calls at all.
    second_client = StubClient()
    again = transcribe_all(audio_tree, out_dir, client=second_client, sleeper=_noop_sleep)
    assert len(again) == 3
    assert second_client.calls == []


def test_limit_restricts_new_calls(audio_tree: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "results"
    client = StubClient()
    records = transcribe_all(audio_tree, out_dir, client=client, limit=2, sleeper=_noop_sleep)
    assert len(client.calls) == 2
    assert len(records) == 2


def test_failed_clips_are_not_cached(audio_tree: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "results"
    client = StubClient(fail_times=99)
    records = transcribe_all(audio_tree, out_dir, client=client, limit=1, sleeper=_noop_sleep)
    assert records == []
    assert not (out_dir / "transcripts.jsonl").exists()


def test_load_cache_skips_malformed_lines(tmp_path: Path) -> None:
    cache_path = tmp_path / "transcripts.jsonl"
    cache_path.write_text(
        '{"key": "a/1.mp3", "text": "ok"}\nnot json at all\n\n', encoding="utf-8"
    )
    cache = load_cache(cache_path)
    assert list(cache) == ["a/1.mp3"]


def test_end_to_end_with_mocked_api(audio_tree: Path, tmp_path: Path) -> None:
    """Full pipeline on three dummy files: every table and figure appears."""
    out_dir = tmp_path / "results"
    tables = run_pipeline(
        audio_dir=audio_tree,
        out_dir=out_dir,
        client=StubClient(),
        sleeper=_noop_sleep,
        focus="pakistani",
    )

    for name in (
        "results.csv",
        "summary_by_accent.csv",
        "pairwise_tests.csv",
        "error_types_by_accent.csv",
        "word_error_rates.csv",
        "hard_words.csv",
        "wrong_language_counts.csv",
        "overall_summary.csv",
    ):
        assert (out_dir / name).is_file(), f"missing {name}"

    for figure in ("wer_by_accent.png", "error_types_by_accent.png", "hardest_words.png"):
        assert (out_dir / "figures" / figure).is_file(), f"missing {figure}"

    results = tables["results"]
    assert len(results) == 3
    # The clip fed the exact reference must score a perfect zero.
    assert results.loc[results["key"] == "english_us/us1.mp3", "wer"].iloc[0] == 0.0
    # The Urdu-script clip must be flagged, and only that one.
    flagged = results.loc[results["wrong_language_output"], "key"].tolist()
    assert flagged == ["pakistani/pk2.mp3"]

    summary = tables["summary_by_accent"]
    assert set(summary["accent"]) == {"english_us", "pakistani"}
    pakistani = summary.loc[summary["accent"] == "pakistani"].iloc[0]
    assert pakistani["n"] == 2 and pakistani["n_clean"] == 1
    assert pakistani["ci95_low"] <= pakistani["mean_wer"] <= pakistani["ci95_high"]

    pairwise = tables["pairwise_tests"]
    assert len(pairwise) == 1
    assert pairwise.iloc[0]["group_b"] == "english_us"
    assert -1.0 <= pairwise.iloc[0]["cliffs_delta"] <= 1.0

    assert not tables["hard_words"].empty
    assert tables["wrong_language_counts"].set_index("accent").loc[
        "pakistani", "n_wrong_language"
    ] == 1


def test_rerun_with_skip_transcribe_makes_no_calls(audio_tree: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "results"
    run_pipeline(audio_dir=audio_tree, out_dir=out_dir, client=StubClient(), sleeper=_noop_sleep)

    client = StubClient()
    tables = run_pipeline(
        audio_dir=audio_tree, out_dir=out_dir, client=client, skip_transcribe=True
    )
    assert client.calls == []
    assert len(tables["results"]) == 3
