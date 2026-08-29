"""recovery portfolio 客户端审计回放 Gate（plans/active/roadmap/04_EXECUTION_PLAN.md §2）。

退出条件（§2.3）：审计组件仅有 capability/event/portfolio GET 访问；前后端测试证明
不输出敏感 parameters；正确处理 stale state；从客户端实际可追溯到同一 checkpoint
revision。本文件覆盖 S0（投影/状态矩阵）与 S1（checkpoint 回放）两层证据。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app
from api.seed_runtime import SeedRuntime
from seed import Seed
from seed_platform.workbench import WorkbenchEnvironment
from taiji import ActionIntent


def _boot(tmp_path, monkeypatch, *, episode_id="recovery-audit"):
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    checkpoint = tmp_path / f"{episode_id}.pt"
    runtime = SeedRuntime(Seed(episode_id=episode_id), checkpoint_path=checkpoint)
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    runtime.model.architecture.observe(65, learn=False)
    return runtime


def _snapshot_id(runtime) -> str:
    return runtime.workbench_environment.capability_snapshot.snapshot_id


def _branch(
    *,
    branch_id: str,
    status: str,
    created_tick: int = 10,
    expires_at_tick: int = 210,
    terminal_reason: str = "",
    **extra,
) -> dict:
    """构造一条内部 branch 记录；extra 用于藏入必须被投影脱敏的执行细节。"""
    return {
        "branch_id": branch_id,
        "loop_id": f"loop:{branch_id}",
        "parent_loop_id": "parent-a",
        "capability_id": "workspace.read",
        "source_evidence_id": f"evidence:{branch_id}",
        "source_after_state_digest": "digest:" + branch_id,
        "budget_limit": 1.0,
        "budget_units": 0.25,
        "completed_steps": 2,
        "frontier_affordance_ids": ["affordance:next"],
        "created_tick": created_tick,
        "last_touched_tick": created_tick + 5,
        "expires_at_tick": expires_at_tick,
        "status": status,
        "terminal_reason": terminal_reason,
        **extra,
    }


def _inject_portfolio(
    runtime,
    *,
    parent_loop_id="parent-a",
    branches,
    evicted=(),
    revision=3,
    max_branches=5,
    ttl_ticks=200,
    last_maintenance_tick=1,
) -> None:
    runtime._workbench_loop_state["recovery_portfolio"] = {
        "format": "taiji-recovery-portfolio-v1",
        "version": 1,
        "parent_loop_id": parent_loop_id,
        "snapshot_id": _snapshot_id(runtime),
        "branches": branches,
        "evicted_branches": evicted,
        "revision": revision,
        "max_branches": max_branches,
        "branch_ttl_ticks": ttl_ticks,
        "last_maintenance_tick": last_maintenance_tick,
    }


##############################################################################
# S0-a：投影状态矩阵 + 脱敏
##############################################################################


def test_snapshot_projection_locks_lifecycle_matrix_and_redacts_payloads(
    tmp_path, monkeypatch
) -> None:
    runtime = _boot(tmp_path, monkeypatch, episode_id="audit-matrix")
    _inject_portfolio(
        runtime,
        branches=[
            _branch(branch_id="done", status="completed", created_tick=1),
            _branch(
                branch_id="fail",
                status="failed",
                created_tick=2,
                terminal_reason="recovery_needed",
            ),
            _branch(branch_id="sel", status="selected", created_tick=3),
            _branch(branch_id="act", status="active", created_tick=4),
            # 超过 expires_at_tick 的 active 分支应按 liveness 投影为 expired
            _branch(
                branch_id="stale-active",
                status="active",
                created_tick=1,
                expires_at_tick=0,
            ),
            # 内部记录藏有可执行细节，投影必须全部剔除
            _branch(
                branch_id="secret",
                status="active",
                created_tick=5,
                parameters={"path": "/sensitive/executable.json"},
                evidence={"after_state": "cannot-leak"},
                committed_request_ids=["req:1"],
                consumed_affordance_ids=["affordance:consumed"],
            ),
        ],
        evicted=[
            {
                "branch_id": "evicted-1",
                "loop_id": "loop:evicted-1",
                "source_evidence_id": "evidence:evicted-1",
                "source_after_state_digest": "digest:evicted-1",
                "evicted_tick": 40,
                "reason": "capacity_exhausted",
            }
        ],
        revision=7,
    )
    runtime._workbench_loop_state["successor_graph"] = {
        "parent_loop_id": "parent-a",
        "recovery_branch_id": "sel",
    }
    observed = runtime.taiji_workbench_recovery_portfolio_snapshot(
        parent_loop_id="parent-a",
        snapshot_id=_snapshot_id(runtime),
        expected_revision=7,
    )

    assert observed["revision"] == 7
    assert observed["status"] == "portfolio_snapshot"
    # 元数据与新鲜度
    assert observed["max_branches"] == 5
    assert observed["branch_ttl_ticks"] == 200
    assert observed["last_maintenance_tick"] == 1
    assert observed["current_tick"] > 0
    assert observed["selected_branch_id"] == "sel"

    # 生命周期状态机：五种状态一个不漏，且 liveness 过期被投影为 expired
    statuses = [item["status"] for item in observed["branches"]]
    assert set(statuses) == {"completed", "failed", "selected", "active", "expired"}
    assert observed["counts"] == {
        "active": 2,
        "selected": 1,
        "completed": 1,
        "failed": 1,
        "expired": 1,
        "evicted": 1,
    }
    assert observed["liveness_due_branch_ids"] == ["stale-active"]
    assert [item["branch_id"] for item in observed["branches"] if item["status"] == "expired"] == [
        "stale-active"
    ]

    # lineage 字段必须投影（source evidence / after-state digest / 预算 / frontier）
    secret = next(item for item in observed["branches"] if item["branch_id"] == "secret")
    assert secret["source_evidence_id"] == "evidence:secret"
    assert secret["source_after_state_digest"] == "digest:secret"
    assert secret["budget_units"] == 0.25
    assert secret["frontier_affordance_ids"] == ["affordance:next"]

    # 脱敏：任何分支/墓碑都不得携带可执行字段
    for item in [*observed["branches"], *observed["evicted_branches"]]:
        for forbidden in (
            "parameters",
            "evidence",
            "committed_request_ids",
            "consumed_affordance_ids",
        ):
            assert forbidden not in item, f"投影泄露 {forbidden}：{item.get('branch_id')}"

    # 墓碑：原因与关联 revision 可审计，但无可执行细节
    tombstone = observed["evicted_branches"][0]
    assert tombstone["branch_id"] == "evicted-1"
    assert tombstone["status"] == "evicted"
    assert tombstone["evicted_tick"] == 40
    assert tombstone["reason"] == "capacity_exhausted"


def test_snapshot_projection_handles_capacity_pressure_and_empty_state(
    tmp_path, monkeypatch
) -> None:
    runtime = _boot(tmp_path, monkeypatch, episode_id="audit-capacity")
    # 容量压力：max_branches=2 却已经有 3 条（含两条过期待回收）
    _inject_portfolio(
        runtime,
        branches=[
            _branch(branch_id="b1", status="active", created_tick=1, expires_at_tick=0),
            _branch(branch_id="b2", status="active", created_tick=2, expires_at_tick=0),
            _branch(branch_id="b3", status="selected", created_tick=3),
        ],
        max_branches=2,
        revision=2,
    )
    observed = runtime.taiji_workbench_recovery_portfolio_snapshot(
        parent_loop_id="parent-a",
        snapshot_id=_snapshot_id(runtime),
    )
    assert observed["counts"]["expired"] == 2
    assert observed["counts"]["selected"] == 1
    assert observed["max_branches"] == 2
    assert set(observed["liveness_due_branch_ids"]) == {"b1", "b2"}

    # 空组合：结构化空态，不炸、含元数据
    _inject_portfolio(runtime, branches=[], evicted=[], revision=0)
    empty = runtime.taiji_workbench_recovery_portfolio_snapshot(
        parent_loop_id="parent-a",
        snapshot_id=_snapshot_id(runtime),
        expected_revision=0,
    )
    assert empty["branches"] == []
    assert empty["evicted_branches"] == []
    assert empty["counts"] == {
        "active": 0,
        "selected": 0,
        "completed": 0,
        "failed": 0,
        "expired": 0,
        "evicted": 0,
    }
    assert empty["revision"] == 0


##############################################################################
# S0-b：只读 lineage 绑定（新的 context 投影）+ 结构化错误码
##############################################################################


def test_context_projection_supplies_readonly_binding_key(tmp_path, monkeypatch) -> None:
    runtime = _boot(tmp_path, monkeypatch, episode_id="audit-context")
    # 无 portfolio：结构化空绑定
    no_portfolio = runtime.taiji_workbench_recovery_portfolio_context()
    assert no_portfolio["has_portfolio"] is False

    _inject_portfolio(
        runtime,
        branches=[_branch(branch_id="b1", status="selected", created_tick=1)],
        revision=4,
    )
    runtime._workbench_loop_state["successor_graph"] = {
        "parent_loop_id": "parent-a",
        "recovery_branch_id": "b1",
    }
    context = runtime.taiji_workbench_recovery_portfolio_context()
    assert context["has_portfolio"] is True
    assert context["parent_loop_id"] == "parent-a"
    assert context["snapshot_id"] == _snapshot_id(runtime)
    assert context["revision"] == 4
    assert context["selected_branch_id"] == "b1"


def test_snapshot_and_context_routes_emit_structured_error_codes(tmp_path, monkeypatch) -> None:
    runtime = _boot(tmp_path, monkeypatch, episode_id="audit-errors")

    import api.seed_runtime as seed_runtime_module

    monkeypatch.setattr(seed_runtime_module, "_runtime", runtime)
    snapshot_id = _snapshot_id(runtime)
    with TestClient(create_app(startup_tasks=False)) as client:
        # 未持久化
        response = client.get(
            "/api/workbench/taiji/recovery-branch/portfolio",
            params={"parent_loop_id": "parent-a", "snapshot_id": snapshot_id},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "portfolio_not_persisted"

        # 只读 context 路由：无 portfolio 时返回结构化空绑定而非错误
        context = client.get("/api/workbench/taiji/recovery-branch/context")
        assert context.status_code == 200
        assert context.json()["has_portfolio"] is False
        assert "parent_loop_id" not in context.json()

        # 注入后：snapshot 不匹配、父循环不匹配、revision 过期
        _inject_portfolio(runtime, branches=[], evicted=[], revision=3)
        response = client.get(
            "/api/workbench/taiji/recovery-branch/portfolio",
            params={"parent_loop_id": "parent-a", "snapshot_id": "wrong-snapshot"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "portfolio_snapshot_not_current"

        response = client.get(
            "/api/workbench/taiji/recovery-branch/portfolio",
            params={"parent_loop_id": "other-parent", "snapshot_id": snapshot_id},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "portfolio_parent_mismatch"

        response = client.get(
            "/api/workbench/taiji/recovery-branch/portfolio",
            params={
                "parent_loop_id": "parent-a",
                "snapshot_id": snapshot_id,
                "expected_revision": 2,
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["error"] == "portfolio_revision_stale"
        assert detail["observed_revision"] == 3

        # 注入后 context 返回真实绑定键
        context = client.get("/api/workbench/taiji/recovery-branch/context")
        assert context.status_code == 200
        assert context.json()["has_portfolio"] is True
        assert context.json()["parent_loop_id"] == "parent-a"
        assert context.json()["revision"] == 3


##############################################################################
# S1：真实 portfolio 经 checkpoint 往返后，绑定键 / revision / 顺序 / 新鲜度一致
##############################################################################


def test_real_portfolio_checkpoint_replay_preserves_binding_and_audit_facts(
    tmp_path, monkeypatch
) -> None:
    runtime = _boot(tmp_path, monkeypatch, episode_id="audit-s1")
    snapshot_id = _snapshot_id(runtime)

    # 真实的 reproject → 失败 → handoff 链路，先造出 portfolio 的父失败记录
    runtime.project_workbench_affordances(
        snapshot_id=snapshot_id,
        parameter_bindings={"workspace.read": {"path": "missing.txt"}},
    )
    runtime.execute_taiji_workbench_successor_loop(
        snapshot_id=snapshot_id,
        loop_id="parent-loop",
        max_steps=1,
    )
    (tmp_path / "missing.txt").write_bytes(b"portfolio evidence\n")
    runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="s1-fresh-read",
            kind="workspace.read",
            parameters={"path": "missing.txt"},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
    )
    runtime.handoff_taiji_workbench_recovery(
        parent_loop_id="parent-loop",
        recovery_loop_id="child-loop",
        snapshot_id=snapshot_id,
        max_steps=1,
    )
    before = runtime.taiji_workbench_recovery_portfolio_snapshot(
        parent_loop_id="parent-loop",
        snapshot_id=snapshot_id,
    )
    assert before["revision"] >= 1
    runtime.save()

    restored = SeedRuntime.load(runtime.checkpoint_path)
    restored._workbench_environment = WorkbenchEnvironment(tmp_path)
    context = restored.taiji_workbench_recovery_portfolio_context()
    assert context["parent_loop_id"] == "parent-loop"
    assert context["revision"] == before["revision"]
    assert context["snapshot_id"] == snapshot_id

    after = restored.taiji_workbench_recovery_portfolio_snapshot(
        parent_loop_id="parent-loop",
        snapshot_id=snapshot_id,
        expected_revision=before["revision"],
    )
    assert after["revision"] == before["revision"]
    assert after["counts"] == before["counts"]
    assert [item["branch_id"] for item in after["branches"]] == [
        item["branch_id"] for item in before["branches"]
    ]
    # 新鲜度：恢复后的运行时不得把旧 tick 当成当前时刻
    assert after["current_tick"] == int(restored.model.architecture.cognitive_snapshot().world.tick)
