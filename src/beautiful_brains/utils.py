"""Small parsing and validation helpers."""

from __future__ import annotations

from pathlib import Path

from beautiful_brains.exceptions import ConfigError


def ensure_path(path: str | Path, *, must_exist: bool = True) -> Path:
    resolved = Path(path).expanduser()
    resolved = resolve_existing_nifti(resolved)
    if must_exist and not resolved.exists():
        raise ConfigError(f"Path does not exist: {resolved}")
    return resolved


def resolve_existing_nifti(path: Path) -> Path:
    if path.exists():
        return path
    for candidate in nifti_variants(path):
        if candidate.exists():
            return candidate
    return path


def nifti_variants(path: Path) -> tuple[Path, ...]:
    name = path.name
    if name.endswith(".nii.gz"):
        return (path, path.with_name(name[: -len(".gz")]))
    if name.endswith(".nii"):
        return (path, path.with_name(name + ".gz"))
    return (path,)


def parse_labels(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    labels: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start, stop = [int(part.strip()) for part in chunk.split("-", maxsplit=1)]
            if stop < start:
                raise ConfigError("Label ranges must be ascending.")
            labels.extend(range(start, stop + 1))
        else:
            labels.append(int(chunk))
    return tuple(labels)
