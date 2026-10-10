"""PLAN-N5-06 实施格的产品侧契约测（J-N6a-1／2／3／5 与 G-N6a-1／2／3）。

判据先冻在 `plans/reference/PLAN-N5-06_world_model_interface_prereg_20261009.md` §2/§3，
本册只做一件事：**让每一条都能为假**。三处口径必须在文字上写死，否则冻结式会被读成两种结论：

* J-N6a-1 的 `signature.index("budget") == 2`：普查器出版的是 `ast.unparse` 的**字符串**，
  字符下标没有语义 ⇒ 本册把它解回形参名并**显式排除 `self`**（不排除则是 3）。
  这是解释，不是放松——`self` 的取舍写在断言旁边。
* J-N6a-3 在冻结形状（连续 8 步 vs 4 步＋快照＋恢复＋4 步）之间**插入 2 步污染**：
  没有这一步，`restore` 哪怕是个 no-op 也能过（[[guard-must-be-able-to-fail]]）。
* G-N6a-1 用字面量钉普查器的别名集：补齐合同必须靠**加方法**，不许改宽别名。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
CENSUS = REPO / "scripts" / "training" / "audit_taiji_n5_world_model_contract.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji import (  # noqa: E402
    Outcome,
    WorldAction,
    WorldDynamicsLearner,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldSchema,
    WorldState,
    WorldTransition,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.world_learning import (  # noqa: E402
    WORLD_LEARNER_SNAPSHOT_FORMAT,
    WORLD_LEARNER_SNAPSHOT_VERSION,
)

#: G-N6a-1 的字面量钉子（㊵-641 入库时就是这个别名集，实施本件时一字未改）。
PINNED_CONTRACT_VERBS: dict[str, tuple[str, ...]] = {
    "observe": ("observe", "online_update"),
    "propose": ("propose",),
    "feedback": ("feedback", "record_schema_feedback"),
    "snapshot_restore": ("snapshot", "restore"),
}

OBJECT_IDS = ("agent", "red", "blue")
COUNTER_KEYS = (
    "online_updates",
    "transition_acceptances",
    "transition_rejections",
    "schema_evolution_count",
)
UPDATE_SEQUENCE = (("red", 1.0), ("blue", -1.0), ("red", 2.0), ("blue", 1.0))


def _load_census() -> Any:
    spec = importlib.util.spec_from_file_location("n5_19_census", CENSUS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(tick: int, positions: dict[str, float]) -> WorldState:
    return WorldState(
        tick=tick,
        objects=tuple(
            WorldObject(object_id, attributes={"position": positions[object_id]})
            for object_id in OBJECT_IDS
        ),
    )


def _home() -> WorldState:
    return _state(0, {"agent": 0.0, "red": 0.0, "blue": 0.0})


def _move(target: str, step: float, index: int) -> WorldInterventionCase:
    initial = _home()
    positions = {"agent": 0.0, "red": 0.0, "blue": 0.0}
    positions[target] = positions[target] + step
    action = WorldAction(
        action_id=f"move-{index}",
        kind="move",
        tick=0,
        actor_id="agent",
        target_id=target,
        parameters={"step": step},
    )
    return WorldInterventionCase(
        case_id=f"case-{index}",
        initial=initial,
        action=action,
        expected_state=_state(1, positions),
        expected_outcome=Outcome(
            intent_id=action.action_id, reward=step, success=step > 0.0, tick=1
        ),
    )


def _transitions() -> tuple[WorldTransition, ...]:
    return tuple(
        WorldTransition(
            before=case.initial,
            action=case.action,
            after=case.expected_state,
            outcome=case.expected_outcome,
        )
        for index, case in enumerate(
            _move(target, step, index) for index, (target, step) in enumerate(UPDATE_SEQUENCE)
        )
    )


def _learner() -> WorldDynamicsLearner:
    #: 两 split 必须**不相交**（`contracts.py:2117` 对 case_id 交集是响亮拒绝），所以留检那两例
    #: 用另一段编号，而不是从 train 里切前一条。
    train = tuple(
        _move(target, step, index)
        for index, (target, step) in enumerate(UPDATE_SEQUENCE + (("red", -2.0), ("blue", 2.0)))
    )
    holdout = tuple(
        _move(target, step, 100 + index)
        for index, (target, step) in enumerate((("red", 0.5), ("blue", -0.5)))
    )
    schema = WorldSchema.from_corpus(WorldInterventionCorpus(train=train, holdout=holdout))
    return WorldDynamicsLearner(schema, hidden_dim=16, seed=7)


def _counters(learner: WorldDynamicsLearner) -> dict[str, int]:
    return {key: int(getattr(learner, key)) for key in COUNTER_KEYS}


def _param_face(learner: WorldDynamicsLearner) -> tuple[int, str]:
    state = learner.state_dict()
    return (
        sum(int(tensor.numel()) for tensor in state.values()),
        content_digest({name: tensor.tolist() for name, tensor in state.items()}),
    )


def _predict_gap(
    left: WorldDynamicsLearner, right: WorldDynamicsLearner, state: WorldState, action: WorldAction
) -> float:
    a = left.predict(state, action)
    b = right.predict(state, action)
    values_a = left.schema.state_values(a.state).tolist()
    values_b = right.schema.state_values(b.state).tolist()
    assert len(values_a) == len(values_b), (values_a, values_b)
    gaps = [abs(x - y) for x, y in zip(values_a, values_b, strict=True)]
    gaps += [abs(a.reward - b.reward), abs(a.success_probability - b.success_probability)]
    return max(gaps)


def test_the_census_alias_set_is_unchanged() -> None:
    """G-N6a-1：本件没把普查器改宽来换绿。"""

    census = _load_census()
    assert census.CONTRACT_VERBS == PINNED_CONTRACT_VERBS, census.CONTRACT_VERBS


def test_j_n6a_1_propose_is_matched_with_budget_third() -> None:
    census = _load_census()
    row = census.audit(REPO)["contract"]["propose"]
    assert row["matched_methods"] == ["propose"], row
    signature = row["signatures"]["propose"]
    function = ast.parse(f"def f({signature}): pass").body[0]
    declared = [arg.arg for arg in function.args.args]
    assert declared[0] == "self", declared
    assert declared[1:].index("budget") == 2, declared
    assert "**" not in signature, signature
    #: `self` 计入则是 3——把这条口径写死，防止下一个人以为冻结式在说另一件事。
    assert declared.index("budget") == 3, declared


def test_j_n6a_2_snapshot_and_restore_are_public_and_paired() -> None:
    payload = _load_census().audit(REPO)
    assert payload["missing_verbs"] == [], payload["missing_verbs"]
    assert payload["contract_verdict"] == "contract_complete", payload["contract_verdict"]
    matched = payload["contract"]["snapshot_restore"]["matched_methods"]
    assert matched == ["snapshot", "restore"], matched
    #: 反向：不是靠把私有件改名来"在场"的。
    assert "_snapshot_state_dict" not in payload["public_methods"], payload["public_methods"]


def test_j_n6a_2_restore_refuses_envelopes_snapshot_never_published() -> None:
    learner = _learner()
    good = learner.snapshot()
    assert good["format"] == WORLD_LEARNER_SNAPSHOT_FORMAT
    assert good["version"] == WORLD_LEARNER_SNAPSHOT_VERSION

    with pytest.raises(ValueError, match="format"):
        learner.restore(dict(good, format="taiji-world-learner-snapshot-v0"))
    with pytest.raises(ValueError, match="version"):
        learner.restore(dict(good, version=WORLD_LEARNER_SNAPSHOT_VERSION + 1))
    #: 动一枚权重却留着原摘要 ⇒ 必须被自摘要抓到。
    tampered = dict(good)
    tampered["model_state"] = {name: tensor + 1.0 for name, tensor in good["model_state"].items()}
    with pytest.raises(ValueError, match="digest mismatch"):
        learner.restore(tampered)
    #: 计数块键集是闭合的：多一枚 ⇒ 重新摘要后仍要当场红。
    forged = dict(good)
    forged["counters"] = dict(good["counters"], hacked=1)
    forged["snapshot_digest"] = content_digest(
        {key: value for key, value in forged.items() if key != "snapshot_digest"}
    )
    with pytest.raises(ValueError, match="counters"):
        learner.restore(forged)
    with pytest.raises(TypeError):
        learner.restore([("not", "a mapping")])  # type: ignore[arg-type]


def test_g_n6a_3_the_budget_is_a_real_knob_and_propose_is_read_only() -> None:
    learner = _learner()
    for transition in _transitions():
        learner.online_update(transition)
    action_count = len(learner._proposal_actions())
    assert action_count >= 2, action_count

    goal = _state(0, {"agent": 0.0, "red": 4.0, "blue": 0.0})
    state = _home()
    before = _param_face(learner)
    counters_before = _counters(learner)

    zero = learner.propose(goal, state, 0)
    one = learner.propose(goal, state, 1)
    four = learner.propose(goal, state, 4)

    assert zero == (), zero
    assert len(one) == 1, one
    assert len(four) == min(4, action_count), (len(four), action_count)
    assert len(four) > len(one), (len(one), len(four))
    distances = [candidate.goal_distance for candidate in four]
    assert distances == sorted(distances), distances
    #: 只读性：参数量、权重摘要、四枚计数器在预算 0／1／4 三档之后一字不动。
    assert _param_face(learner) == before, _param_face(learner)
    assert _counters(learner) == counters_before, _counters(learner)
    with pytest.raises(ValueError, match="budget"):
        learner.propose(goal, state, -1)


def test_j_n6a_3_a_restored_learner_continues_bit_for_bit() -> None:
    transitions = _transitions()
    assert len(transitions) == 4, transitions

    path_a = _learner()
    path_b = _learner()
    for transition in transitions:
        path_a.online_update(transition)
        path_b.online_update(transition)
    #: 先证两条路的第 4 步末态本来就相同，否则后面的"逐位相同"是在比两个不同的东西。
    assert _param_face(path_a) == _param_face(path_b), _param_face(path_a)

    envelope = path_b.snapshot()
    for transition in transitions[:2]:
        path_b.online_update(transition)
    assert _param_face(path_b) != _param_face(path_a), "污染步没挪动权重 ⇒ 这条守卫是空的"

    report = path_b.restore(envelope)
    assert report["snapshot_digest"] == envelope["snapshot_digest"], report
    assert report["counters"] == _counters(path_a), (report["counters"], _counters(path_a))
    assert _param_face(path_b) == _param_face(path_a), "restore 没把权重带回来"

    for transition in transitions:
        path_a.online_update(transition)
        path_b.online_update(transition)
    assert _param_face(path_a) == _param_face(path_b), (
        _param_face(path_a)[1],
        _param_face(path_b)[1],
    )

    probe_state = _state(0, {"agent": 0.0, "red": 0.5, "blue": -0.25})
    probe_action = WorldAction(
        action_id="probe",
        kind="move",
        tick=0,
        actor_id="agent",
        target_id="red",
        parameters={"step": 1.0},
    )
    assert _predict_gap(path_a, path_b, probe_state, probe_action) == 0.0


def test_the_census_still_runs_as_a_process_and_reports_complete(tmp_path: Path) -> None:
    """冒烟：判读器不只是 import 得到，**命令行面**也要走得通（㊵-640 那族教训）。"""

    out = tmp_path / "census.json"
    proc = subprocess.run(
        [sys.executable, str(CENSUS), "--out-report", str(out)],
        capture_output=True,
        check=False,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")[:400]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "ok", payload
    assert payload["contract_verdict"] == "contract_complete", payload
    assert payload["missing_verbs"] == [], payload
    assert payload["public_method_count"] == 11, payload["public_method_count"]
