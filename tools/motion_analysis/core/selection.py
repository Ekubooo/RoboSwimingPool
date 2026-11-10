"""Normalize user-facing component choices into one reconstruction contract."""

import operator
import re
from typing import Sequence


def parse_components(text: str) -> tuple[int, ...]:
    """Parse positive 1-based indices and inclusive ranges, e.g. ``1,3,5-8``."""
    if not text.strip():
        return ()
    tokens = re.split(r"[,\s]+", text.strip())
    values: set[int] = set()
    maximum_entries = 10_000
    for token in tokens:
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", token)
        if match is None:
            raise ValueError("Use positive integers or inclusive ranges, e.g. 1,3,5-8.")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start < 1 or end < start:
            raise ValueError("Component numbers start at 1 and ranges must be ascending.")
        if end - start + 1 > maximum_entries:
            raise ValueError("A component range may contain at most 10000 entries.")
        values.update(range(start, end + 1))
        if len(values) > maximum_entries:
            raise ValueError("A component selection may contain at most 10000 entries.")
    return tuple(sorted(values))


def normalize_indices(indices: Sequence[int], max_components: int) -> tuple[int, ...]:
    try:
        if any(isinstance(value, bool) for value in indices):
            raise TypeError("Boolean component index")
        selected = tuple(sorted(set(operator.index(value) for value in indices)))
    except TypeError as exc:
        raise ValueError("Component indices must be integers.") from exc
    if any(value < 0 or value >= max_components for value in selected):
        raise ValueError(f"Component index outside the available range 0..{max_components - 1}.")
    return selected


def resolve_selection(mode: str, k: int | None = None,
                      components: Sequence[int] | None = None,
                      max_components: int = 0) -> tuple[int, ...]:
    if mode == "subset":
        if components is None:
            components = ()
        try:
            if any(isinstance(value, bool) for value in components):
                raise TypeError("Boolean component number")
            values = tuple(operator.index(value) - 1 for value in components)
        except TypeError as exc:
            raise ValueError("Component numbers must be integers.") from exc
        return normalize_indices(values, max_components)
    if mode not in ("cumulative", "single"):
        raise ValueError("Selection mode must be cumulative, single or subset.")
    try:
        if isinstance(k, bool):
            raise TypeError("Boolean K")
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError("K must be an integer.") from exc
    if mode == "cumulative":
        if not 0 <= k <= max_components:
            raise ValueError(f"Cumulative K must be in 0..{max_components}.")
        return tuple(range(k))
    if max_components == 0 and k == 0:
        return ()
    if not 1 <= k <= max_components:
        raise ValueError(f"Single-component K must be in 1..{max_components}.")
    return (k - 1,)
