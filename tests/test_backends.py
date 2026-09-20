"""Tests for the two transcription backends and the transcript cache path.

``torch`` and ``transformers`` are mocked, so these run on a laptop with
neither installed — which is the point of the lazy imports in
:mod:`src.transcribe`.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from src.run import run_pipeline
from src.transcribe import (
    BACKENDS,
    DEFAULT_BACKEND,
    TranscriptionError,
    describe_backends,
    get_backend_client,
    resolve_transcripts_path,
    transcribe_all,
)

REFERENCE_TEXT = "Please call Stella. Ask her to bring these things with her from the store."


class FakePipeline:
    """Stands in for ``transformers.pipeline``'s return value.

    Attributes:
        calls: ``(audio, generate_kwargs)`` per invocation.
        built_with: The kwargs the pipeline factory was called with.
    """

    def __init__(self, **kwargs: object) -> None:
        self.built_with = kwargs
        self.calls: list[tuple[str, object]] = []

    def __call__(self, audio: str, generate_kwargs: object = None) -> dict[str, str]:
        self.calls.append((audio, generate_kwargs))
        return {"text": REFERENCE_TEXT}


@pytest.fixture()
def fake_transformers(monkeypatch):
    """Install fake torch and transformers modules reporting a CUDA device.

    Yields:
        A dict with the fake pipeline factory's recorded state.
    """
    state: dict[str, object] = {"pipelines": [], "cuda": True}

    torch_module = types.ModuleType("torch")
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    # scipy probes torch.Tensor whenever torch is in sys.modules; a fake without
    # it breaks scipy's array dispatch depending on import order.
    torch_module.Tensor = type("Tensor", (), {})
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: bool(state["cuda"]),
        get_device_name=lambda index: "Tesla T4",
    )
    torch_module.cuda = cuda_ns

    def fake_pipeline(task: str, **kwargs: object) -> FakePipeline:
        built = FakePipeline(task=task, **kwargs)
        state["pipelines"].append(built)
        return built

    transformers_module = types.ModuleType("transformers")
    transformers_module.pipeline = fake_pipeline

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    yield state


@pytest.fixture()
def audio_tree(tmp_path: Path) -> Path:
    """Two accents, two clips each."""
    audio_dir = tmp_path / "audio"
    for accent in ("english_us", "pakistani"):
        (audio_dir / accent).mkdir(parents=True)
        for index in range(2):
            (audio_dir / accent / f"{accent}{index}.mp3").write_bytes(b"ID3 dummy")
    return audio_dir


def _noop_sleep(_seconds: float) -> None:
    """Sleep replacement that returns immediately."""


# --- backend selection -----------------------------------------------------


def test_default_backend_is_the_hosted_api() -> None:
    assert DEFAULT_BACKEND == "hf_api"
    assert BACKENDS == ("hf_api", "local")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(TranscriptionError, match="Unknown backend"):
        get_backend_client("gpu_please")


def test_local_backend_loads_the_model_once(fake_transformers, audio_tree, tmp_path) -> None:
    """The pipeline must be built once, not per clip."""
    transcribe_all(
        audio_tree,
        tmp_path / "results",
        backend="local",
        sleeper=_noop_sleep,
    )
    assert len(fake_transformers["pipelines"]) == 1
    assert len(fake_transformers["pipelines"][0].calls) == 4


def test_local_backend_uses_gpu_and_float16_when_cuda_present(fake_transformers) -> None:
    client = get_backend_client("local")
    built = fake_transformers["pipelines"][0].built_with
    assert built["task"] == "automatic-speech-recognition"
    assert built["model"] == "openai/whisper-large-v3"
    assert built["device"] == 0
    assert built["torch_dtype"] == "float16"
    assert built["chunk_length_s"] == 30
    assert client.device == 0


def test_local_backend_falls_back_to_cpu_float32(fake_transformers, caplog) -> None:
    fake_transformers["cuda"] = False
    with caplog.at_level("WARNING"):
        client = get_backend_client("local")
    built = fake_transformers["pipelines"][0].built_with
    assert built["device"] == -1
    assert built["torch_dtype"] == "float32"
    assert client.device == -1
    assert any("No CUDA device" in r.getMessage() for r in caplog.records)


def test_local_backend_does_not_force_a_language(fake_transformers) -> None:
    """Auto-detection must be preserved, or wrong-language output is hidden."""
    client = get_backend_client("local")
    client.automatic_speech_recognition("clip.mp3", model="openai/whisper-large-v3")
    _audio, generate_kwargs = fake_transformers["pipelines"][0].calls[0]
    assert generate_kwargs == {"task": "transcribe"}
    assert "language" not in generate_kwargs


def test_local_backend_reports_missing_dependencies(monkeypatch) -> None:
    """A laptop without torch gets an actionable message, not an ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(TranscriptionError, match="pip install torch transformers"):
        get_backend_client("local")


def test_torch_and_transformers_are_imported_lazily() -> None:
    """Neither may appear at module scope, or the hf_api backend breaks.

    A laptop with no torch installed must still import src.transcribe, run the
    hosted backend and run this suite. Guarding that with a source check is
    blunt but it fails loudly if someone hoists the import to the top.
    """
    source = Path("src/transcribe.py").read_text(encoding="utf-8")
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    ]
    offenders = [line for line in module_level if "torch" in line or "transformers" in line]
    assert offenders == [], f"must be imported inside a function: {offenders}"


def test_hosted_backend_works_without_torch_installed(monkeypatch, audio_tree, tmp_path):
    """With torch unimportable, the hf_api path must still run end to end."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    class Stub:
        def automatic_speech_recognition(self, audio: str, *, model: str):
            return {"text": REFERENCE_TEXT}

    records = transcribe_all(
        audio_tree, tmp_path / "results", client=Stub(),
        backend="hf_api", sleeper=_noop_sleep,
    )
    assert len(records) == 4
    assert all(row["backend"] == "hf_api" for row in records)


# --- cache path and provenance ---------------------------------------------


def test_resolve_transcripts_path_default_and_override(tmp_path: Path) -> None:
    assert resolve_transcripts_path(tmp_path) == tmp_path / "transcripts.jsonl"
    explicit = tmp_path / "transcripts_local.jsonl"
    assert resolve_transcripts_path(tmp_path, explicit) == explicit


def test_records_carry_backend_and_model(fake_transformers, audio_tree, tmp_path) -> None:
    out_dir = tmp_path / "results"
    cache = out_dir / "transcripts_local.jsonl"
    transcribe_all(
        audio_tree, out_dir, backend="local", transcripts=cache, sleeper=_noop_sleep
    )

    assert cache.is_file()
    assert not (out_dir / "transcripts.jsonl").exists(), "must not write the default cache"
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    for row in rows:
        assert row["backend"] == "local"
        assert row["model"] == "openai/whisper-large-v3"
        # Format stays compatible with the hf_api cache.
        assert {"key", "accent", "file", "text", "timestamp"} <= set(row)


def test_cache_format_is_shared_between_backends(audio_tree, tmp_path) -> None:
    """A local-backend cache must be readable by the ordinary loader."""
    class Stub:
        def automatic_speech_recognition(self, audio: str, *, model: str):
            return {"text": REFERENCE_TEXT}

    out_dir = tmp_path / "results"
    api_rows = transcribe_all(
        audio_tree, out_dir, client=Stub(), backend="hf_api",
        transcripts=out_dir / "a.jsonl", sleeper=_noop_sleep,
    )
    local_rows = transcribe_all(
        audio_tree, out_dir, client=Stub(), backend="local",
        transcripts=out_dir / "b.jsonl", sleeper=_noop_sleep,
    )
    assert set(api_rows[0]) == set(local_rows[0])
    assert api_rows[0]["backend"] == "hf_api"
    assert local_rows[0]["backend"] == "local"


def test_describe_backends_treats_legacy_rows_as_hf_api() -> None:
    records = [{"backend": "local"}, {"backend": "local"}, {}]
    assert describe_backends(records) == {"local": 2, "hf_api": 1}


def test_mixing_backends_in_one_cache_warns(audio_tree, tmp_path, caplog) -> None:
    class Stub:
        def automatic_speech_recognition(self, audio: str, *, model: str):
            return {"text": REFERENCE_TEXT}

    out_dir = tmp_path / "results"
    shared = out_dir / "transcripts.jsonl"
    transcribe_all(
        audio_tree, out_dir, client=Stub(), backend="hf_api",
        transcripts=shared, sleeper=_noop_sleep,
    )
    # Remove one clip from the cache so there is something left to transcribe.
    lines = shared.read_text(encoding="utf-8").splitlines()
    shared.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        transcribe_all(
            audio_tree, out_dir, client=Stub(), backend="local",
            transcripts=shared, sleeper=_noop_sleep,
        )
    assert any("Mixing backends" in r.getMessage() for r in caplog.records)


# --- the analysis reads whichever cache it is given ------------------------


def test_analysis_reads_the_named_transcripts_file(fake_transformers, audio_tree, tmp_path):
    """Tables must be built from --transcripts, not the default path."""
    out_dir = tmp_path / "results"
    cache = out_dir / "transcripts_local.jsonl"

    tables = run_pipeline(
        audio_dir=audio_tree,
        out_dir=out_dir,
        backend="local",
        transcripts=cache,
        sleeper=_noop_sleep,
        focus="pakistani",
    )
    assert cache.is_file()
    assert len(tables["results"]) == 4
    provenance = __import__("pandas").read_csv(out_dir / "run_provenance.csv")
    assert provenance.set_index("backend").loc["local", "n_clips"] == 4


def test_skip_transcribe_honours_the_named_transcripts_file(audio_tree, tmp_path) -> None:
    class Stub:
        def automatic_speech_recognition(self, audio: str, *, model: str):
            return {"text": REFERENCE_TEXT}

    out_dir = tmp_path / "results"
    cache = out_dir / "transcripts_local.jsonl"
    run_pipeline(
        audio_dir=audio_tree, out_dir=out_dir, client=Stub(),
        transcripts=cache, backend="local", sleeper=_noop_sleep,
    )
    tables = run_pipeline(
        audio_dir=audio_tree, out_dir=out_dir,
        transcripts=cache, skip_transcribe=True,
    )
    assert len(tables["results"]) == 4
