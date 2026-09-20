"""Unit tests for the .env loader in :mod:`src.config`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import load_dotenv, load_reference


def test_load_dotenv_sets_missing_variables(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_from_file\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    applied = load_dotenv(env_file)
    assert applied == {"HF_TOKEN": "hf_from_file"}
    assert os.environ["HF_TOKEN"] == "hf_from_file"


def test_exported_variable_wins_over_the_file(tmp_path: Path, monkeypatch) -> None:
    """A token exported in the shell must not be silently replaced."""
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_from_file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_from_shell")

    assert load_dotenv(env_file) == {}
    assert os.environ["HF_TOKEN"] == "hf_from_shell"


def test_override_replaces_existing_value(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_from_file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_from_shell")

    load_dotenv(env_file, override=True)
    assert os.environ["HF_TOKEN"] == "hf_from_file"


def test_load_dotenv_handles_comments_quotes_and_export(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                'HF_TOKEN="hf_quoted"',
                "export OTHER_KEY='single'",
                "malformed line without equals",
            ]
        ),
        encoding="utf-8",
    )
    for key in ("HF_TOKEN", "OTHER_KEY"):
        monkeypatch.delenv(key, raising=False)

    applied = load_dotenv(env_file)
    assert applied == {"HF_TOKEN": "hf_quoted", "OTHER_KEY": "single"}


def test_load_dotenv_without_a_file_is_a_no_op(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_reference_matches_the_dataset_passage() -> None:
    """reference.txt must stay the Speech Accent Archive elicitation paragraph."""
    reference = load_reference()
    assert reference.startswith("Please call Stella.")
    assert reference.rstrip().endswith("at the train station.")
    assert len(reference.split()) == 69
