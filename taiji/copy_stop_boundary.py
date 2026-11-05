"""Structural copy stop boundary for the two-hop answer (package 19.2u).

19.2t showed that the copy operator has no stopping edge: after the object name
ends, the carried memory keeps copying the record terminator into the template
slot.  This module supplies that edge as an *algorithmic primitive*: the record
template itself declares where the name field starts and ends, so the copy
operator can be told to stop before the terminator.

The boundary is derived from the record bytes only:

* ``<description>是<name>。``  - name follows ``是`` and ends at ``。``
* ``<name>又叫<description>。`` - name precedes ``又叫``
* ``<description>对应<name>。`` - name follows ``对应`` and ends at ``。``

Nothing here is sample-specific: there is no hard-coded "byte 15" or "12 bytes".
The only constants are the three template delimiters and the terminator, which
are the contract of the frozen 18.0 data.  The returned offsets are **byte**
offsets, matching ``RecordSourceView.record_offsets``.

Registered in ``candidate_identity.GRAPH_MODULES`` as of package 19.2u, so
``candidate_sha_v2`` changes by design; the module has no learnable parameters.
"""

from __future__ import annotations

TERMINATOR = "。"
SPLIT_STATED = "是"
SPLIT_AKA = "又叫"
SPLIT_MAPPED = "对应"


class CopyBoundaryError(ValueError):
    """Raised when a record does not expose a name field next to its template."""


def style_name(raw: str) -> str:
    """Template style of a record line, read from its delimiters."""
    body = raw[: -len(TERMINATOR)] if raw.endswith(TERMINATOR) else raw
    for split, style in ((SPLIT_AKA, "aka"), (SPLIT_MAPPED, "mapped"), (SPLIT_STATED, "stated")):
        if split in body:
            return style
    raise CopyBoundaryError(f"record has no known template delimiter: {raw!r}")


def field_bounds(raw: str, description: str) -> tuple[int, int]:
    """Byte offsets ``(start, end)`` of the name field adjacent to ``description``.

    ``end`` is the first byte after the name, i.e. the stop position the copy
    operator must honour.  The terminator itself is never inside the field.
    """
    if not raw.endswith(TERMINATOR):
        raise CopyBoundaryError(f"record does not end with {TERMINATOR!r}: {raw!r}")
    body = raw[: -len(TERMINATOR)]
    at = body.find(description)
    if at < 0:
        raise CopyBoundaryError(f"description not present in record: {raw!r}")
    tail = body[at + len(description) :]
    if tail.startswith(SPLIT_STATED):
        prefix = body[: at + len(description) + len(SPLIT_STATED)]
        name = tail[len(SPLIT_STATED) :]
    elif tail.startswith(SPLIT_MAPPED):
        prefix = body[: at + len(description) + len(SPLIT_MAPPED)]
        name = tail[len(SPLIT_MAPPED) :]
    else:
        head = body[:at]
        if not head.endswith(SPLIT_AKA):
            raise CopyBoundaryError(
                f"description is neither suffixed nor prefixed by a split: {raw!r}"
            )
        prefix = ""
        name = head[: -len(SPLIT_AKA)]
    if not name:
        raise CopyBoundaryError(f"empty name field in record: {raw!r}")
    start = len(prefix.encode("utf-8"))
    end = start + len(name.encode("utf-8"))
    return start, end


def copy_limits_for(
    raw: str,
    description: str,
    route: int,
    record_count: int,
) -> tuple[int, ...]:
    """Per-record copy limits; ``-1`` means "this line carries no stop edge"."""
    limits = [-1] * record_count
    if 0 <= route < record_count:
        _start, end = field_bounds(raw, description)
        limits[route] = end
    return tuple(limits)
