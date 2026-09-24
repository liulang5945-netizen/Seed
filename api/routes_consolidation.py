"""Native sleep pass API.

Endpoints:
- GET  /api/consolidation/status  → what the last pass did and what the rings hold
- GET  /api/consolidation/spec    → the data ring's current product
- POST /api/consolidate           → run one pass (analyse, project, specify, optionally sleep the substrate)

This surface is Taiji-native and always available.  It is also the only entry the
native line has into its own consolidation: the legacy ``/api/life/sleep`` drives
``neuroplex.life.sleep_engine``, which the native runtime never mounts.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.models import ConsolidationRequest
from api.seed_runtime import get_seed_runtime
from seed_platform import sleep_pass

logger = logging.getLogger("ApiServer.Consolidation")
router = APIRouter(prefix="/api", tags=["consolidation"])


@router.get("/consolidation/status")
def consolidation_status():
    """Return the last pass's report plus what the native rings hold now."""

    return sleep_pass.status()


@router.get("/consolidation/spec")
def consolidation_spec():
    """Return the data ring's current product, or null when none was ever written."""

    return {"spec": sleep_pass.status().get("spec")}


@router.post("/consolidate")
async def consolidate(request: ConsolidationRequest):
    """Run one native sleep pass.

    The pass reads the native rings, writes the rehearsal corpus and the
    data-ring spec, and — only when asked — hands the night's texts to the
    substrate's own sleep scheduler.  ``learn`` additionally allows that sleep to
    change weights, so it stays an explicit request rather than a default.
    """

    try:
        return await asyncio.to_thread(
            sleep_pass.run,
            reason=request.reason,
            runtime=get_seed_runtime(),
            organs=request.organs,
            learn=request.learn,
            cycles_per_text=request.cycles_per_text,
            max_symbols=request.max_symbols,
            max_texts=request.max_texts,
            max_records=request.max_records,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive, the pass logs its own steps
        logger.error("睡眠巩固失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"睡眠巩固失败: {exc}") from exc
