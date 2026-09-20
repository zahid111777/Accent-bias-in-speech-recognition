"""Select Speech Accent Archive recordings into per-accent folders.

The Kaggle mirror of the Speech Accent Archive (``rtatman/speech-accent-archive``)
ships a metadata CSV plus a folder of recordings. This module turns that into a
small, balanced working set: ``audio/<accent>/*.mp3`` plus ``audio/manifest.csv``.

Accent groups are defined on the command line with ``--rule`` so that the
grouping is explicit and auditable rather than hidden in the code, e.g.::

    --rule "pakistani:country=pakistan"
    --rule "english_us:country=usa&native_language=english"

Column names are resolved case-insensitively through a small alias table, so the
script adapts to the header spellings actually present in the CSV.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .config import AUDIO_EXTENSIONS, MIN_GROUP_SIZE, SEED, setup_logging

LOGGER = logging.getLogger(__name__)

#: Canonical metadata field -> header spellings seen in the wild.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "filename": ("filename", "file", "filenames", "audio", "recording", "path"),
    "native_language": ("nativelanguage", "language", "nativelang", "l1"),
    "country": ("country", "birthcountry", "countryofbirth"),
    "sex": ("sex", "gender"),
    "age": ("age",),
    "speakerid": ("speakerid", "id", "speaker"),
    "birthplace": ("birthplace", "placeofbirth"),
    "file_missing": ("filemissing", "missing", "filemissingq"),
}

#: Alternative spellings accepted on the right-hand side of a country rule.
COUNTRY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "usa": (
        "usa",
        "us",
        "u.s.a.",
        "u.s.",
        "united states",
        "united states of america",
        "america",
    ),
    "uk": (
        "uk",
        "u.k.",
        "united kingdom",
        "great britain",
        "britain",
        "england",
        "scotland",
        "wales",
    ),
}

#: Metadata columns copied into the manifest when present.
MANIFEST_FIELDS: tuple[str, ...] = (
    "country",
    "native_language",
    "sex",
    "age",
    "speakerid",
    "birthplace",
)


class PrepareDataError(RuntimeError):
    """Raised for user-fixable problems such as missing folders or bad rules."""


@dataclass(frozen=True)
class AccentRule:
    """One accent group definition parsed from a ``--rule`` argument.

    Attributes:
        name: Folder-safe group name, e.g. ``pakistani``.
        conditions: Metadata field -> required value (lowercased).
    """

    name: str
    conditions: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        """Return a human-readable rendering of the rule."""
        body = "&".join(f"{k}={v}" for k, v in self.conditions.items())
        return f"{self.name}:{body}"


def _canonical_key(raw: str) -> str:
    """Reduce a column name or field name to a comparison key."""
    return re.sub(r"[^a-z0-9]", "", str(raw).strip().lower())


def parse_rule(rule: str) -> AccentRule:
    """Parse a ``name:field=value[&field=value...]`` rule string.

    Args:
        rule: Raw ``--rule`` value.

    Returns:
        The parsed :class:`AccentRule`.

    Raises:
        PrepareDataError: If the rule is malformed or has no conditions.
    """
    if ":" not in rule:
        raise PrepareDataError(
            f"Bad --rule {rule!r}: expected 'name:field=value[&field=value]', "
            "for example 'pakistani:country=pakistan'."
        )
    name, _, body = rule.partition(":")
    name = name.strip()
    if not name:
        raise PrepareDataError(f"Bad --rule {rule!r}: the group name is empty.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise PrepareDataError(
            f"Bad --rule {rule!r}: group name {name!r} must contain only "
            "letters, digits, underscores or hyphens (it becomes a folder name)."
        )

    conditions: dict[str, str] = {}
    for clause in body.split("&"):
        clause = clause.strip()
        if not clause:
            continue
        if "=" not in clause:
            raise PrepareDataError(
                f"Bad --rule {rule!r}: condition {clause!r} is missing '='."
            )
        key, _, value = clause.partition("=")
        key, value = key.strip(), value.strip().lower()
        if not key or not value:
            raise PrepareDataError(
                f"Bad --rule {rule!r}: condition {clause!r} has an empty side."
            )
        conditions[key] = value

    if not conditions:
        raise PrepareDataError(f"Bad --rule {rule!r}: no conditions given.")
    return AccentRule(name=name, conditions=conditions)


def resolve_columns(columns: Sequence[str]) -> dict[str, str]:
    """Map canonical field names to the actual CSV column names.

    Matching uses a letters-and-digits-only lowercase key, so ``"Native
    Language"``, ``native_language`` and ``nativelanguage`` all resolve to the
    same field. Columns outside the alias table stay addressable by their own
    canonicalised name, so a rule can reference any column in the file.

    Args:
        columns: The CSV header as read by pandas.

    Returns:
        Mapping of canonical field name -> real column name.
    """
    by_key = {_canonical_key(col): col for col in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for candidate in (canonical, *aliases):
            key = _canonical_key(candidate)
            if key in by_key:
                resolved[canonical] = by_key[key]
                break
    for key, col in by_key.items():
        resolved.setdefault(key, col)
    return resolved


def _lookup_column(
    field_name: str, resolved: dict[str, str], columns: Sequence[str]
) -> str:
    """Return the real column backing ``field_name`` or raise a helpful error."""
    if field_name in resolved:
        return resolved[field_name]
    key = _canonical_key(field_name)
    if key in resolved:
        return resolved[key]
    raise PrepareDataError(
        f"Rule refers to column {field_name!r}, which is not in the CSV. "
        f"Available columns: {', '.join(map(str, columns))}"
    )


def _values_match(actual: object, wanted: str, field_name: str) -> bool:
    """Compare a cell value with a rule value, allowing country synonyms."""
    actual_norm = str(actual).strip().lower()
    if actual_norm == wanted:
        return True
    if _canonical_key(field_name) == "country":
        for canonical, spellings in COUNTRY_SYNONYMS.items():
            group = {canonical, *spellings}
            if wanted in group and actual_norm in group:
                return True
    return False


def index_audio_files(audio_src: Path) -> dict[str, Path]:
    """Index every audio file under ``audio_src`` by lowercase stem and name.

    Args:
        audio_src: Folder holding the recordings (searched recursively).

    Returns:
        Mapping of lookup key -> path. Both ``english1`` and ``english1.mp3``
        are registered for each file.

    Raises:
        PrepareDataError: If the folder does not exist or holds no audio.
    """
    if not audio_src.is_dir():
        raise PrepareDataError(
            f"Audio source folder not found: {audio_src}. Point --audio_src at "
            "the unzipped 'recordings' folder inside raw_data/."
        )

    index: dict[str, Path] = {}
    count = 0
    for path in sorted(audio_src.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        count += 1
        for key in (path.stem.lower(), path.name.lower()):
            # First writer wins; sorted traversal keeps that deterministic when
            # duplicate stems exist in nested folders.
            index.setdefault(key, path)
    if not count:
        raise PrepareDataError(
            f"No audio files ({', '.join(AUDIO_EXTENSIONS)}) found under {audio_src}."
        )
    LOGGER.info("Indexed %d audio files under %s", count, audio_src)
    return index


def find_audio(filename: object, index: dict[str, Path]) -> Path | None:
    """Resolve a metadata filename to a real audio path.

    Handles metadata that stores the stem only (``english1``), the full name
    (``english1.mp3``) or a relative path (``recordings/english1.mp3``).

    Args:
        filename: Value of the metadata filename column.
        index: Index produced by :func:`index_audio_files`.

    Returns:
        The matching path, or ``None`` if no recording exists.
    """
    if filename is None:
        return None
    raw = str(filename).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return None

    base = Path(raw.replace("\\", "/")).name.lower()
    stem = Path(base).stem
    candidates = [base, stem, *(f"{stem}{ext}" for ext in AUDIO_EXTENSIONS)]
    for candidate in candidates:
        hit = index.get(candidate)
        if hit is not None:
            return hit
    return None


def _is_missing_flag(value: object) -> bool:
    """Interpret the dataset's ``file_missing?`` column as a boolean."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "yes", "1", "1.0"}


def assign_groups(
    frame: pd.DataFrame,
    rules: Sequence[AccentRule],
    resolved: dict[str, str],
) -> dict[str, list[int]]:
    """Assign each metadata row to the first rule it satisfies.

    Groups are kept disjoint on purpose: a speaker counted in two groups would
    break the independence assumption of the later significance tests.

    Args:
        frame: The metadata table.
        rules: Accent rules in command-line order.
        resolved: Column resolution map from :func:`resolve_columns`.

    Returns:
        Mapping of accent name -> list of row labels.
    """
    columns = list(frame.columns)
    compiled: list[tuple[AccentRule, list[tuple[str, str, str]]]] = []
    for rule in rules:
        clauses = [
            (field_name, _lookup_column(field_name, resolved, columns), wanted)
            for field_name, wanted in rule.conditions.items()
        ]
        compiled.append((rule, clauses))

    groups: dict[str, list[int]] = {rule.name: [] for rule in rules}
    for idx, row in frame.iterrows():
        for rule, clauses in compiled:
            if all(
                _values_match(row.get(column, ""), wanted, field_name)
                for field_name, column, wanted in clauses
            ):
                groups[rule.name].append(idx)
                break
    return groups


def _sample(indices: Sequence[int], max_per_group: int | None, seed: int) -> list[int]:
    """Take a deterministic sample of at most ``max_per_group`` indices."""
    ordered = sorted(indices)
    if max_per_group is None or len(ordered) <= max_per_group:
        return ordered
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, max_per_group))


def prepare_data(
    csv_path: Path,
    audio_src: Path,
    out_dir: Path,
    rules: Sequence[AccentRule],
    max_per_group: int | None = None,
    seed: int = SEED,
    copy_files: bool = True,
) -> pd.DataFrame:
    """Build ``out_dir/<accent>/`` folders and ``out_dir/manifest.csv``.

    Args:
        csv_path: Path to ``speakers_all.csv``.
        audio_src: Folder of recordings (searched recursively).
        out_dir: Destination folder, typically ``audio``.
        rules: Accent group definitions, in priority order.
        max_per_group: Cap on clips per accent, or ``None`` for no cap.
        seed: Seed for the per-group subsample.
        copy_files: If False, build the manifest without copying audio.

    Returns:
        The manifest as a DataFrame.

    Raises:
        PrepareDataError: For missing inputs, unknown columns or empty groups.
    """
    csv_path, audio_src, out_dir = Path(csv_path), Path(audio_src), Path(out_dir)
    if not csv_path.is_file():
        raise PrepareDataError(
            f"Metadata CSV not found: {csv_path}. Expected the Kaggle file "
            "'speakers_all.csv' unzipped into raw_data/."
        )

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # The Kaggle CSV carries a few unnamed trailing columns; drop them quietly.
    frame = frame.loc[:, [c for c in frame.columns if not str(c).startswith("Unnamed:")]]
    LOGGER.info("Read %d metadata rows from %s", len(frame), csv_path)
    LOGGER.info("CSV columns: %s", ", ".join(map(str, frame.columns)))

    resolved = resolve_columns(list(frame.columns))
    if "filename" not in resolved:
        raise PrepareDataError(
            "Could not find a filename column in the CSV. Columns present: "
            f"{', '.join(map(str, frame.columns))}"
        )
    filename_col = resolved["filename"]

    index = index_audio_files(audio_src)
    groups = assign_groups(frame, rules, resolved)

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    small_groups: list[str] = []

    for rule in rules:
        indices = groups[rule.name]
        LOGGER.info("Rule %-30s matched %d metadata rows", rule.describe(), len(indices))

        resolvable: list[tuple[int, Path]] = []
        unmatched = 0
        flagged_missing = 0
        for idx in indices:
            row = frame.loc[idx]
            missing_col = resolved.get("file_missing")
            if missing_col and _is_missing_flag(row.get(missing_col)):
                flagged_missing += 1
                continue
            audio_path = find_audio(row.get(filename_col), index)
            if audio_path is None:
                unmatched += 1
                LOGGER.debug("No audio for %s (%s)", row.get(filename_col), rule.name)
                continue
            resolvable.append((idx, audio_path))

        if flagged_missing:
            LOGGER.info(
                "%s: skipped %d rows the CSV flags as missing",
                rule.name,
                flagged_missing,
            )
        if unmatched:
            LOGGER.warning(
                "%s: %d matching rows had no audio file on disk and were skipped",
                rule.name,
                unmatched,
            )

        keep = set(_sample([i for i, _ in resolvable], max_per_group, seed))
        chosen = [(i, p) for i, p in resolvable if i in keep]

        group_dir = out_dir / rule.name
        if copy_files and chosen:
            group_dir.mkdir(parents=True, exist_ok=True)

        for idx, src_path in chosen:
            row = frame.loc[idx]
            dest = group_dir / src_path.name
            if copy_files and not dest.exists():
                shutil.copy2(src_path, dest)
            record: dict[str, object] = {
                "file": dest.as_posix(),
                "accent": rule.name,
                "source_path": src_path.as_posix(),
            }
            for canonical in MANIFEST_FIELDS:
                column = resolved.get(canonical)
                record[canonical] = row.get(column, "") if column else ""
            records.append(record)

        LOGGER.info("Accent %-12s -> %d files in %s", rule.name, len(chosen), group_dir)
        if not chosen:
            LOGGER.error(
                "Accent %r ended up with 0 files. Check the rule against the "
                "actual CSV values (columns are logged above).",
                rule.name,
            )
        elif len(chosen) < MIN_GROUP_SIZE:
            small_groups.append(f"{rule.name} (n={len(chosen)})")

    if small_groups:
        LOGGER.warning(
            "Groups below the %d-clip minimum: %s. Continuing, but treat their "
            "confidence intervals and p-values as indicative only.",
            MIN_GROUP_SIZE,
            "; ".join(small_groups),
        )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise PrepareDataError(
            "No files were selected for any accent group. Check the --rule "
            "values against the CSV (the log above lists the real columns)."
        )

    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    LOGGER.info("Wrote manifest with %d rows to %s", len(manifest), manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m src.prepare_data`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.prepare_data",
        description="Copy Speech Accent Archive recordings into per-accent folders.",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to speakers_all.csv")
    parser.add_argument(
        "--audio_src", type=Path, required=True, help="Folder holding the recordings"
    )
    parser.add_argument("--out", type=Path, default=Path("audio"), help="Output folder")
    parser.add_argument(
        "--rule",
        action="append",
        dest="rules",
        required=True,
        metavar="NAME:FIELD=VALUE[&FIELD=VALUE]",
        help="Accent group definition; repeatable. E.g. pakistani:country=pakistan",
    )
    parser.add_argument(
        "--max_per_group", type=int, default=None, help="Cap on clips per accent group"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for subsampling")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for ``python -m src.prepare_data``.

    Returns:
        Process exit code: 0 on success, 2 on a user-fixable error.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    setup_logging(args.verbose)
    try:
        rules = [parse_rule(raw) for raw in args.rules]
        prepare_data(
            csv_path=args.csv,
            audio_src=args.audio_src,
            out_dir=args.out,
            rules=rules,
            max_per_group=args.max_per_group,
            seed=args.seed,
        )
    except PrepareDataError as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
