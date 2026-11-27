"""Taiji N36/N37/N38 Operational Telemetry & Live Training Dashboard API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from taiji.ops_contract import (
    ALLOWED_TRANSITIONS,
    BOUNDARIES,
    SCOPE,
    STATES,
    OutOfContractChangeError,
    classify_change,
)

ROOT = Path(__file__).resolve().parents[1]
N38_STATUS_PATH = ROOT / "data" / "n38_open_zh" / "training_status.json"
N38_METRICS_PATH = ROOT / "data" / "n38_open_zh" / "training_metrics.jsonl"
N38_PREFLIGHT_PATH = ROOT / "reports" / "n38_0_01_stage0_preflight.json"
NATIVE_STATUS_PATH = ROOT / "data" / "n38_native_zh" / "native_backbone_status.json"
NATIVE_PREFLIGHT_PATH = ROOT / "reports" / "n38_2_01_native_backbone_preflight.json"
GROWTH_POLICY_PATH = ROOT / "data" / "taiji_ops" / "growth_policy.json"

DEFAULT_GROWTH_POLICY: dict[str, Any] = {
    "auto_grow_enabled": True,
    "max_params": 40_000_000,
    "max_layers": 16,
    "max_group_units": 2048,
    "max_fast_heads": 8,
    "max_retention_drop": 0.02,
    "min_gain": 0.001,
}

router = APIRouter(tags=["taiji-ops"])


class ClassifyChangeRequest(BaseModel):
    kind: str


class GrowthPolicyUpdateRequest(BaseModel):
    auto_grow_enabled: bool | None = None
    max_params: int | None = None
    max_layers: int | None = None
    max_group_units: int | None = None
    max_fast_heads: int | None = None
    max_retention_drop: float | None = None
    min_gain: float | None = None


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


def _read_growth_policy() -> dict[str, Any]:
    policy = dict(DEFAULT_GROWTH_POLICY)
    saved = _read_json_safe(GROWTH_POLICY_PATH)
    if saved:
        for k in DEFAULT_GROWTH_POLICY:
            if k in saved:
                policy[k] = saved[k]
    return policy


def _write_growth_policy(policy: dict[str, Any]) -> None:
    GROWTH_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = GROWTH_POLICY_PATH.with_name(f".{GROWTH_POLICY_PATH.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, GROWTH_POLICY_PATH)


@router.get("/api/taiji/ops/overview")
def taiji_ops_overview() -> dict[str, Any]:
    """Return consolidated N31–N37 closeout status, boundaries, and N38 training state."""
    n38_status = _read_json_safe(N38_STATUS_PATH) or {"status": "not_started"}
    n38_status.setdefault("role", "baseline_transformer")
    return {
        "schema": "taiji-ops-overview-v1",
        "scope": SCOPE,
        "boundaries": dict(BOUNDARIES),
        "mainline_closed": ["<=N30", "N31", "N32", "N33", "N34", "N35", "N36", "N37"],
        "fsm_states": list(STATES),
        "fsm_transitions": {k: sorted(v) for k, v in ALLOWED_TRANSITIONS.items()},
        "n38_pretraining": n38_status,
        "native_backbone": taiji_native_backbone(),
    }


@router.get("/api/taiji/ops/n38-training")
def taiji_n38_training() -> dict[str, Any]:
    """Return live N38 mid_18m pretraining telemetry, recent metrics, and Stage 0 baseline."""
    status = _read_json_safe(N38_STATUS_PATH) or {"status": "not_started"}
    status.setdefault("role", "baseline_transformer")
    return {
        "schema": "taiji-n38-training-telemetry-v1",
        "status": status,
        "recent_metrics": _read_recent_metrics(N38_METRICS_PATH, limit=40),
        "stage0_preflight": _read_json_safe(N38_PREFLIGHT_PATH),
    }


@router.get("/api/taiji/ops/native-backbone")
def taiji_native_backbone() -> dict[str, Any]:
    """Return Taiji native non-Transformer backbone status, preflight, and growth policy."""
    baseline = _read_json_safe(N38_STATUS_PATH) or {"status": "not_started"}
    baseline.setdefault("role", "baseline_transformer")
    return {
        "schema": "taiji-native-backbone-ops-v1",
        "architecture": "taiji-native-byte-group-v2",
        "input_encoding": "raw_utf8_bytes_259",
        "non_transformer": True,
        "growth_trigger_scope": "engineering_prototype",
        "status": _read_json_safe(NATIVE_STATUS_PATH)
        or {"status": "ready_in_trunk", "preset": "native_byte_mid"},
        "preflight": _read_json_safe(NATIVE_PREFLIGHT_PATH),
        "growth_policy": _read_growth_policy(),
        "baseline_transformer": baseline,
    }


@router.post("/api/taiji/ops/growth-policy")
def taiji_update_growth_policy(req: GrowthPolicyUpdateRequest) -> dict[str, Any]:
    """Update autonomous shadow growth budget & safety gates."""
    policy = _read_growth_policy()
    if req.auto_grow_enabled is not None:
        policy["auto_grow_enabled"] = bool(req.auto_grow_enabled)
    if req.max_params is not None:
        if req.max_params < 1_000_000:
            raise HTTPException(status_code=400, detail="max_params must be >= 1,000,000")
        policy["max_params"] = int(req.max_params)
    if req.max_layers is not None:
        if req.max_layers < 1:
            raise HTTPException(status_code=400, detail="max_layers must be >= 1")
        policy["max_layers"] = int(req.max_layers)
    if req.max_group_units is not None:
        if req.max_group_units < 32:
            raise HTTPException(status_code=400, detail="max_group_units must be >= 32")
        policy["max_group_units"] = int(req.max_group_units)
    if req.max_fast_heads is not None:
        if req.max_fast_heads < 1:
            raise HTTPException(status_code=400, detail="max_fast_heads must be >= 1")
        policy["max_fast_heads"] = int(req.max_fast_heads)
    if req.max_retention_drop is not None:
        if not (0.0 <= req.max_retention_drop <= 0.2):
            raise HTTPException(status_code=400, detail="max_retention_drop must be in [0, 0.2]")
        policy["max_retention_drop"] = float(req.max_retention_drop)
    if req.min_gain is not None:
        if req.min_gain < 0.0:
            raise HTTPException(status_code=400, detail="min_gain must be >= 0")
        policy["min_gain"] = float(req.min_gain)
    _write_growth_policy(policy)
    return {"status": "updated", "growth_policy": policy}


@router.get("/api/taiji/ops/architecture-readiness")
def taiji_architecture_readiness() -> dict[str, Any]:
    """Return authoritative-architecture module readiness: implemented vs waiting on base weights."""
    return {
        "schema": "taiji-architecture-readiness-v1",
        "native_pretraining": _read_json_safe(NATIVE_STATUS_PATH)
        or {"status": "not_started", "preset": "native_byte_mid"},
        "baseline_transformer": _read_json_safe(N38_STATUS_PATH)
        or {"status": "not_started", "preset": "mid_18m"},
        "parallel_modules": {
            "dynamic_byte_chunker": {
                "module": "taiji.dynamic_byte_chunker",
                "status": "implemented_and_tested",
                "spec_ref": "TAIJI_NATIVE_ARCHITECTURE_V1 §2.2/§5.2/§5.10",
            },
            "conditional_value": {
                "module": "taiji.conditional_value",
                "status": "implemented_and_tested",
                "spec_ref": "VISION §3.2/VX2",
            },
            "action_lifecycle": {
                "module": "taiji.action_lifecycle",
                "status": "implemented_and_tested",
                "spec_ref": "TAIJI_NATIVE_ARCHITECTURE_V1 §4.7/GOV-ACT",
            },
            "development_learning": {
                "module": "taiji.development_learning",
                "status": "implemented_and_tested",
                "spec_ref": "VISION §3.4/VX4/GOV-LEARN",
            },
        },
        "waiting_for_trained_native_base": [
            {
                "id": "open_zh_curriculum_stage2",
                "reason": "Requires converged native_byte_mid byte representation before source/correction copy curriculum",
            },
            {
                "id": "dynamic_chunk_bpb_ablation",
                "reason": "Must compare dynamic macro-chunk growth against fixed 8-byte patch BPB curve from task-5203",
            },
            {
                "id": "autonomous_vx5_policy",
                "reason": "Learned autonomous scheduling must be trained atop real representation & outcome feedback (VX2/VX4)",
            },
            {
                "id": "a1_a9_joint_gates",
                "reason": "Full semantic/world/planning/language joint capability gates require non-random native substrate",
            },
        ],
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
      <h2>Baseline-Transformer（标准对照基线）N38.1.01 mid_18m</h2>
      <div id="n38-stats">加载中...</div>
      <h2 style="margin-top:14px;">实时中文续写采样</h2>
      <pre id="n38-sample">等待首个评测步...</pre>
    </div>
    <div class="card">
      <h2>态极原生纯字节有状态神经群底座（零 BPE / 零 Transformer / FP1 KKT / SVD 中性出生）</h2>
      <div id="native-stats">加载中...</div>
      <div style="display:flex;gap:8px;margin-top:10px;align-items:center;">
        <label style="font-size:12px;"><input type="checkbox" id="auto-grow" checked /> 自动影子生长</label>
        <input id="max-params" type="number" value="40000000" style="width:110px" />
        <button onclick="savePolicy()">保存预算</button>
      </div>
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
    const [ov, tr, nb] = await Promise.all([
      fetch('/api/taiji/ops/overview').then(r => r.json()),
      fetch('/api/taiji/ops/n38-training').then(r => r.json()),
      fetch('/api/taiji/ops/native-backbone').then(r => r.json())
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
    const nst = nb.status || {};
    const gp = nb.growth_policy || {};
    document.getElementById('native-stats').innerHTML = [
      ['架构', nb.architecture],
      ['状态 / 预设', `${nst.status || '-'} / ${nst.preset || 'native_mid_18m'}`],
      ['自动生长 / 参数上限', `${gp.auto_grow_enabled ? '开启' : '关闭'} / ${(gp.max_params || 0).toLocaleString()}`],
      ['保持门禁 / 净增益门禁', `≤ ${gp.max_retention_drop ?? 0.02} / ≥ ${gp.min_gain ?? 0.001}`]
    ].map(([k,v]) => `<div class="kv"><span>${k}</span><strong>${v}</strong></div>`).join('');
    document.getElementById('auto-grow').checked = !!gp.auto_grow_enabled;
    if (gp.max_params) document.getElementById('max-params').value = gp.max_params;
    document.getElementById('ops-stats').innerHTML = [
      ['声明范围', ov.scope],
      ['已收口阶段', (ov.mainline_closed || []).join(' → ')],
      ['check / final 读取', `${ov.boundaries?.check_reads ?? 0} / ${ov.boundaries?.final_reads ?? 0}`],
      ['8 态 FSM', (ov.fsm_states || []).join(' · ')]
    ].map(([k,v]) => `<div class="kv"><span>${k}</span><strong>${v}</strong></div>`).join('');
  } catch (e) { console.error(e); }
}
async function savePolicy() {
  const auto_grow_enabled = document.getElementById('auto-grow').checked;
  const max_params = parseInt(document.getElementById('max-params').value, 10);
  await fetch('/api/taiji/ops/growth-policy', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({auto_grow_enabled, max_params})
  });
  refresh();
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
