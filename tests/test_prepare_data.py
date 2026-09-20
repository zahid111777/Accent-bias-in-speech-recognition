"""Unit tests for :mod:`src.prepare_data` using a toy CSV and dummy audio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.prepare_data import (
    PrepareDataError,
    assign_groups,
    find_audio,
    index_audio_files,
    parse_rule,
    prepare_data,
    resolve_columns,
)

#: A toy stand-in for speakers_all.csv, including the quirks of the real file:
#: mixed casing, a filename column without extensions, and a missing-file flag.
TOY_ROWS = [
    ("pakistani1", "urdu", "pakistan", "male", "31", "false"),
    ("pakistani2", "urdu", "Pakistan", "female", "24", "false"),
    ("pakistani3", "punjabi", "pakistan", "male", "45", "false"),
    ("pakistani4", "urdu", "pakistan", "female", "29", "true"),  # flagged missing
    ("hindi1", "hindi", "india", "female", "22", "false"),
    ("hindi2", "hindi", "india", "male", "37", "false"),
    ("english1", "english", "usa", "male", "40", "false"),
    ("english2", "english", "usa", "female", "33", "false"),
    ("english3", "english", "uk", "male", "51", "false"),
    ("ghost1", "urdu", "pakistan", "male", "28", "false"),  # no audio on disk
]


@pytest.fixture()
def toy_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a toy CSV, a recordings folder and an output folder.

    Returns:
        ``(csv_path, audio_src, out_dir)``.
    """
    csv_path = tmp_path / "speakers_all.csv"
    frame = pd.DataFrame(
        TOY_ROWS,
        columns=["filename", "native_language", "country", "sex", "age", "file_missing?"],
    )
    frame.to_csv(csv_path, index=False)

    audio_src = tmp_path / "recordings"
    audio_src.mkdir()
    for row in TOY_ROWS:
        if row[0] == "ghost1":  # deliberately absent from disk
            continue
        (audio_src / f"{row[0]}.mp3").write_bytes(b"ID3 dummy audio")

    return csv_path, audio_src, tmp_path / "audio"


def test_parse_rule_single_condition() -> None:
    rule = parse_rule("pakistani:country=pakistan")
    assert rule.name == "pakistani"
    assert rule.conditions == {"country": "pakistan"}


def test_parse_rule_multiple_conditions_and_case_folding() -> None:
    rule = parse_rule("english_us:country=USA&native_language=English")
    assert rule.conditions == {"country": "usa", "native_language": "english"}
    assert rule.describe() == "english_us:country=usa&native_language=english"


@pytest.mark.parametrize(
    "bad",
    [
        "no_colon_here",
        "empty:",
        ":country=pakistan",
        "spaces in name:country=pakistan",
        "broken:country",
        "broken:=pakistan",
    ],
)
def test_parse_rule_rejects_malformed_input(bad: str) -> None:
    with pytest.raises(PrepareDataError):
        parse_rule(bad)


def test_resolve_columns_is_insensitive_to_spelling() -> None:
    resolved = resolve_columns(["Filename", "Native Language", "COUNTRY", "file_missing?"])
    assert resolved["filename"] == "Filename"
    assert resolved["native_language"] == "Native Language"
    assert resolved["country"] == "COUNTRY"
    assert resolved["file_missing"] == "file_missing?"


def test_index_and_find_audio_accept_stems_names_and_paths(toy_dataset) -> None:
    _, audio_src, _ = toy_dataset
    index = index_audio_files(audio_src)
    assert find_audio("english1", index).name == "english1.mp3"
    assert find_audio("english1.mp3", index).name == "english1.mp3"
    assert find_audio("recordings/english1.mp3", index).name == "english1.mp3"
    assert find_audio("ghost1", index) is None
    assert find_audio("", index) is None


def test_index_audio_files_rejects_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(PrepareDataError):
        index_audio_files(tmp_path / "does_not_exist")


def test_assign_groups_is_disjoint_and_first_rule_wins() -> None:
    frame = pd.DataFrame(
        TOY_ROWS,
        columns=["filename", "native_language", "country", "sex", "age", "file_missing?"],
    )
    resolved = resolve_columns(list(frame.columns))
    rules = [
        parse_rule("pakistani:country=pakistan"),
        parse_rule("urdu_speakers:native_language=urdu"),  # overlaps with the first
    ]
    groups = assign_groups(frame, rules, resolved)
    assert len(set(groups["pakistani"]) & set(groups["urdu_speakers"])) == 0
    assert len(groups["pakistani"]) == 5  # every Pakistan row, incl. flagged/ghost
    assert groups["urdu_speakers"] == []


def test_country_synonyms_match(toy_dataset) -> None:
    frame = pd.DataFrame(
        TOY_ROWS,
        columns=["filename", "native_language", "country", "sex", "age", "file_missing?"],
    )
    resolved = resolve_columns(list(frame.columns))
    groups = assign_groups(frame, [parse_rule("us:country=united states")], resolved)
    assert len(groups["us"]) == 2


def test_prepare_data_copies_files_and_writes_manifest(toy_dataset, caplog) -> None:
    csv_path, audio_src, out_dir = toy_dataset
    rules = [
        parse_rule("pakistani:country=pakistan"),
        parse_rule("hindi:native_language=hindi"),
        parse_rule("english_us:country=usa&native_language=english"),
    ]
    manifest = prepare_data(csv_path, audio_src, out_dir, rules)

    counts = manifest["accent"].value_counts().to_dict()
    assert counts == {"pakistani": 3, "hindi": 2, "english_us": 2}

    # The flagged row and the row with no audio must not be copied.
    copied = {path.name for path in (out_dir / "pakistani").iterdir()}
    assert copied == {"pakistani1.mp3", "pakistani2.mp3", "pakistani3.mp3"}

    assert (out_dir / "manifest.csv").is_file()
    assert set(manifest.columns) >= {"file", "accent", "country", "native_language", "sex", "age"}


def test_prepare_data_warns_about_small_groups(toy_dataset, caplog) -> None:
    csv_path, audio_src, out_dir = toy_dataset
    with caplog.at_level("WARNING"):
        prepare_data(csv_path, audio_src, out_dir, [parse_rule("pakistani:country=pakistan")])
    assert any("below the 10-clip minimum" in record.getMessage() for record in caplog.records)


def test_max_per_group_is_capped_and_deterministic(toy_dataset) -> None:
    csv_path, audio_src, out_dir = toy_dataset
    rules = [parse_rule("pakistani:country=pakistan")]
    first = prepare_data(csv_path, audio_src, out_dir / "a", rules, max_per_group=2, seed=7)
    second = prepare_data(csv_path, audio_src, out_dir / "b", rules, max_per_group=2, seed=7)
    assert len(first) == 2
    assert [Path(p).name for p in first["file"]] == [Path(p).name for p in second["file"]]


def test_prepare_data_rejects_unknown_rule_column(toy_dataset) -> None:
    csv_path, audio_src, out_dir = toy_dataset
    with pytest.raises(PrepareDataError, match="not in the CSV"):
        prepare_data(csv_path, audio_src, out_dir, [parse_rule("x:favourite_colour=blue")])


def test_prepare_data_rejects_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(PrepareDataError, match="Metadata CSV not found"):
        prepare_data(
            tmp_path / "nope.csv",
            tmp_path,
            tmp_path / "out",
            [parse_rule("pakistani:country=pakistan")],
        )


def test_prepare_data_errors_when_no_group_matches(toy_dataset) -> None:
    csv_path, audio_src, out_dir = toy_dataset
    with pytest.raises(PrepareDataError, match="No files were selected"):
        prepare_data(csv_path, audio_src, out_dir, [parse_rule("nowhere:country=atlantis")])
