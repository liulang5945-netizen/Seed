"""Taiji-native memory journal API.

Endpoints:
- POST /api/memory/record  → append one entry (idempotent per content)
- GET  /api/memory/recall  → keyword/importance/recency ranked read
- GET  /api/memory/status  → what the journal holds

This surface is always available because it is the runtime's own memory.  The
legacy ``/api/agent/memory/*`` face is a different thing: no-op stubs behind
``SEED_ENABLE_LEGACY``, never mounted on a Seed-native runtime.
"""

from fastapi import APIRouter, Query

from api.models import MemoryRecordRequest
from seed_platform import memory_store

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.post("/record")
def memory_record(request: MemoryRecordRequest):
    """Append one entry; a repeated report of the same content is not appended twice.

    The outcome is returned rather than raised, so a reporter retrying a write
    can tell "recorded" from "duplicate" without treating either as an error.
    """

    return memory_store.record(
        request.kind,
        request.text,
        session_id=request.session_id,
        turn=request.turn,
        tags=request.tags,
        importance=request.importance,
        source=request.source,
        metadata=request.metadata,
    )


@router.get("/recall")
def memory_recall(
    query: str = "",
    kind: str = "",
    session_id: str = "",
    tag: str = "",
    limit: int = Query(8, ge=0, le=200),
):
    """Return ranked entries; an empty query ranks by importance and recency."""

    return {
        "query": query,
        "entries": memory_store.recall(
            query, kind=kind, session_id=session_id, tag=tag, limit=limit
        ),
    }


@router.get("/status")
def memory_status():
    """Return journal counts, for operators and the Life surface."""

    return memory_store.status()
