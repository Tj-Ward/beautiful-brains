"""Lookup-table parsing for integer masks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from beautiful_brains.exceptions import ConfigError


@dataclass(frozen=True)
class LookupTableEntry:
    label: int
    name: str
    rgba: tuple[int, int, int, int]


def builtin_lookup_table_path(name: str) -> Path:
    normalized = name.lower()
    if normalized in {"freesurfer", "fs", "freesurfer-basic"}:
        return Path(str(files("beautiful_brains.assets.luts").joinpath("freesurfer_basic.tsv")))
    raise ConfigError(f"Unknown built-in LUT: {name}")


def resolve_lookup_table_path(path_or_name: Path | str) -> Path:
    path = Path(path_or_name).expanduser()
    if path.exists():
        return path
    if isinstance(path_or_name, str):
        return builtin_lookup_table_path(path_or_name)
    raise ConfigError(f"LUT path does not exist: {path}")


def load_lookup_table(path_or_name: Path | str) -> dict[int, LookupTableEntry]:
    path = resolve_lookup_table_path(path_or_name)
    if not path.exists():
        raise ConfigError(f"LUT path does not exist: {path}")

    entries: dict[int, LookupTableEntry] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else None
        if delimiter:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                entry = _entry_from_mapping(row)
                entries[entry.label] = entry
        else:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                label = int(parts[0])
                name = parts[1]
                r, g, b, a = [int(float(value)) for value in parts[2:6]]
                entries[label] = LookupTableEntry(label=label, name=name, rgba=(r, g, b, a))

    if 0 not in entries:
        entries[0] = LookupTableEntry(label=0, name="Unknown", rgba=(0, 0, 0, 0))
    return entries


def _entry_from_mapping(row: dict[str, str]) -> LookupTableEntry:
    normalized = {key.lower().strip(): value for key, value in row.items()}
    label = int(float(normalized.get("index") or normalized.get("label") or normalized["id"]))
    name = normalized.get("name", str(label))
    r = int(float(normalized.get("r") or normalized.get("red") or 0))
    g = int(float(normalized.get("g") or normalized.get("green") or 0))
    b = int(float(normalized.get("b") or normalized.get("blue") or 0))
    a = int(float(normalized.get("a") or normalized.get("alpha") or 255))
    return LookupTableEntry(label=label, name=name, rgba=(r, g, b, a))
