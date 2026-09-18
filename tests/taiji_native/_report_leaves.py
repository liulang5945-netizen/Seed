"""Leaf-path flattening of two JSON artifacts, shared by the CAP-0 contract tests.

Why this exists: several CAP-0 claims are of the form "these two artifacts of the same instrument
differ only in *these* fields" (a fresh re-sample vs a sealed one; a relaxed guard vs none). Stated
as prose that claim rots silently the moment a field is added; stated as an exact leaf comparison it
becomes falsifiable. Flattening rules and the volatile-field judgement are the bug-prone part, so
they live here once instead of per test file.

Leaf paths are dotted for object keys and bracketed for list indices
(``dimensions.C.items[3].turns[0].seconds``); ``tail()`` returns the trailing field name.
"""

from __future__ import annotations

import re
from typing import Any


def leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON to ``{path: scalar}``. Empty containers yield no leaves."""

    if isinstance(value, dict):
        return {
            child_key: leaf
            for key, child in value.items()
            for child_key, leaf in leaves(child, f"{prefix}.{key}" if prefix else str(key)).items()
        }
    if isinstance(value, list):
        return {
            child_key: leaf
            for index, child in enumerate(value)
            for child_key, leaf in leaves(child, f"{prefix}[{index}]").items()
        }
    return {prefix: value}


def tail(path: str) -> str:
    """The trailing field name of a leaf path, ignoring its list index."""

    return re.split(r"[.\[]", path)[-1].rstrip("]")


def normalized(path: str) -> str:
    """Wildcard every list index so a pattern can name a set of sibling leaves."""

    return re.sub(r"\[\d+\]", "[*]", path)
