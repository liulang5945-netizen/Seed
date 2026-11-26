"""Taiji N36/N37/N38 Operational Telemetry & Live Training Dashboard API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from taiji.bounded_delivery_runtime import BOUNDARIES, SCOPE
from taiji.operational_state_machine import (
    ALLOWED_TRANSITIONS,
    STATES,
    OutOfContractChangeError,
    classify_change,
)

ROOT = Path(__file__).resolve().parents[1]
N38_STATUS_PATH = ROOT / "data" / "n38_open_zh" / "training_status.json"
N38_METRICS_PATH = ROOT / "data" / "n38_open_zh" / "training_metrics.jsonl"
N38_PREFLIGHT_PATH = ROOT / "reports" / "n38_0_01_stage0_preflight.json"

router = APIRouter(tags=["taiji-ops"])


class ClassifyChangeRequest(BaseModel):
    kind: str


def _read_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_recent_metrics(path: Path, limit: int = 40) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [
        ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()
    ]
    out: list[dict[str, Any]] = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


@router.get("/api/taiji/ops/overview")
def taiji_ops_overview() -> dict[str, Any]:
    """Return consolidated N31–N37 closeout status, boundaries, and N38 training state."""
    n38_status = _read_json_safe(N38_STATUS_PATH)
    return {
        "schema": "taiji-ops-overview-v1",
        "scope": SCOPE,
        "boundaries": dict(BOUNDARIES),
        "mainline_closed": ["<=N30", "N31", "N32", "N33", "N34", "N35", "N36", "N37"],
        "fsm_states": list(STATES),
        "fsm_transitions": {k: sorted(v) for k, v in ALLOWED_TRANSITIONS.items()},
        "n38_pretraining": n38_status or {"status": "not_started"},
    }


@router.get("/api/taiji/ops/n38-training")
def taiji_n38_training() -> dict[str, Any]:
    """Return live N38 mid_18m pretraining telemetry, recent metrics, and Stage 0 baseline."""
    return {
        "schema": "taiji-n38-training-telemetry-v1",
        "status": _read_json_safe(N38_STATUS_PATH) or {"status": "not_started"},
        "recent_metrics": _read_recent_metrics(N38_METRICS_PATH, limit=40),
        "stage0_preflight": _read_json_safe(N38_PREFLIGHT_PATH),
    }


@router.post("/api/taiji/ops/classify-change")
def taiji_classify_change(req: ClassifyChangeRequest) -> dict[str, Any]:
    """Route a proposed change through the §15 3-tier change classifier."""
    try:
        return classify_change(req.kind)
    except OutOfContractChangeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Taiji 态极运营与 N38 预训练实时看板</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --border:#30363d; --fg:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; }
  body { margin:0; font-family:system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--fg); padding:24px; }
  h1 { margin:0 0 8px; font-size:22px; }
  .sub { color:var(--muted); margin-bottom:20px; font-size:13px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .card h2 { margin:0 0 12px; font-size:15px; color:var(--accent); }
  .kv { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #21262d; font-size:13px; }
  .kv:last-child { border-bottom:none; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; background:#1f6feb33; color:var(--accent); }
  .ok { background:#23863633; color:var(--ok); }
  pre { background:#0b0f14; padding:10px; border-radius:6px; overflow-x:auto; font-size:12px; white-space:pre-wrap; }
  input, button { background:#0d1117; color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 10px; font-size:13px; }
  button { background:#1f6feb; border-color:#1f6feb; cursor:pointer; }
</style>
</head>
<body>
  <h1>态极（Taiji）N36/N37 运营与 N38 预训练实时看板</h1>
  <div class="sub">有界主线 ≤N30..N37 已收口（0 check / 0 final） · 每 5 秒自动刷新</div>
  <div class="grid">
    <div class="card">
      <h2>N38.1.01 mid_18m 24h CPU 预训练</h2>
      <div id="n38-stats">加载中...</div>
      <h2 style="margin-top:14px;">实时中文续写采样</h2>
      <pre id="n38-sample">等待首个评测步...</pre>
    </div>
    <div class="card">
      <h2>§15 八态运营状态机与边界</h2>
      <div id="ops-stats">加载中...</div>
      <h2 style="margin-top:14px;">三层变更分流测试</h2>
      <div style="display:flex;gap:8px;">
        <input id="chg-kind" value="ecl_calibration" style="flex:1" />
        <button onclick="testClassify()">分流检测</button>
      </div>
      <pre id="chg-out" style="margin-top:8px;">点击分流检测查看结果</pre>
    </div>
  </div>
<script>
async function refresh() {
  try {
    const [ov, tr] = await Promise.all([
      fetch('/api/taiji/ops/overview').then(r => r.json()),
      fetch('/api/taiji/ops/n38-training').then(r => r.json())
    ]);
    const st = tr.status || {};
    document.getElementById('n38-stats').innerHTML = [
      ['状态', `<span class="badge ok">${st.status || 'unknown'}</span>`],
      ['模型预设 / 参数量', `${st.preset || '-'} / ${(st.params || 0).toLocaleString()}`],
      ['优化步 / 已见 Token', `${st.opt_step || 0} / ${(st.tokens_seen || 0).toLocaleString()}`],
      ['吞吐 / 进度', `${st.tokens_per_sec || 0} tok/s (${st.progress_pct || 0}%)`],
      ['Train Loss / Val PPL', `${st.train_loss ?? '-'} / ${st.val_ppl ?? '-'}`]
    ].map(([k,v]) => `<div class="kv"><span>${k}</span><strong>${v}</strong></div>`).join('');
    if (st.sample_generation) {
      document.getElementById('n38-sample').textContent = st.sample_generation;
    }
    document.getElementById('ops-stats').innerHTML = [
      ['声明范围', ov.scope],
      ['已收口阶段', (ov.mainline_closed || []).join(' → ')],
      ['check / final 读取', `${ov.boundaries?.check_reads ?? 0} / ${ov.boundaries?.final_reads ?? 0}`],
      ['8 态 FSM', (ov.fsm_states || []).join(' · ')]
    ].map(([k,v]) => `<div class="kv"><span>${k}</span><strong>${v}</strong></div>`).join('');
  } catch (e) { console.error(e); }
}
async function testClassify() {
  const kind = document.getElementById('chg-kind').value.trim();
  const res = await fetch('/api/taiji/ops/classify-change', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({kind})
  });
  document.getElementById('chg-out').textContent = JSON.stringify(await res.json(), null, 2);
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


@router.get("/taiji-ops", response_class=HTMLResponse, include_in_schema=False)
def taiji_ops_dashboard() -> HTMLResponse:
    """Serve the standalone Taiji N36/N37/N38 operations & training dashboard."""
    return HTMLResponse(content=_DASHBOARD_HTML)
