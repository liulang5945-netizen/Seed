"""N5 §6 世界模型合同普查器（`audit_taiji_n5_world_model_contract.py`）的契约测。

普查器的全部主张是"四法合同的齐/缺与生产链引用点是**数出来的**，不是说出来的"，所以
**判级两支、缺位支与拒绝支都必须实走**：

* 合成一份四法齐的类 ⇒ `contract_complete`、`missing_verbs` 为空；
* 合成一份缺 `propose` 的类 ⇒ `contract_incomplete` 且**点名缺哪个动词**（不能只给一个总数）；
* 真树读数按本行钉值：现读 `contract_incomplete`／缺 `propose` 与 `snapshot_restore`／
  语料训练链对它的引用计数为 **0**——但同一件里 `taiji/foundation_training.py` 的引用计数
  不为 0 ⇒ 09 那句"零真实调用"**必须带范围限定**，这条测就是防止它被读成"全仓没人用它"；
* 目标文件不在 ⇒ rc=2 且出件写 `census_failed`，那是"没算"，不许折叠成"没缺"。

夹具全部落在 pytest 的 `tmp_path`，不碰 `reports/` 与 `output/`。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_n5_world_model_contract.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_13_world_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

#: 四法齐的类：合同的四个动词都给出公开方法名（名字齐 ⇒ 判齐，语义另说）。
COMPLETE = """
class WorldDynamicsLearner:
    def observe(self, observation, state):
        return state

    def propose(self, goal, state, budget):
        return []

    def feedback(self, real_outcome, state):
        return {}

    def snapshot(self):
        return {}

    def restore(self, payload):
        return None
"""

#: 只缺 `propose`：用来证"缺位点名"不是把所有缺席合并成一个总数。
MISSING_PROPOSE = """
class WorldDynamicsLearner:
    def observe(self, observation, state):
        return state

    def feedback(self, real_outcome, state):
        return {}

    def snapshot(self):
        return {}

    def restore(self, payload):
        return None
"""


def _root(tmp_path: Path, body: str) -> Path:
    learner = tmp_path / "taiji"
    learner.mkdir(parents=True, exist_ok=True)
    (learner / "world_learning.py").write_text(body, encoding="utf-8", newline="\n")
    return tmp_path


def test_a_complete_learner_publishes_contract_complete(tmp_path: Path) -> None:
    payload = MODULE.audit(_root(tmp_path, COMPLETE))
    assert payload["contract_verdict"] == "contract_complete", payload["contract"]
    assert payload["missing_verbs"] == []
    assert payload["public_method_count"] == 5
    assert payload["reading_limit"]  # 取法声明必须在件里，不许只在注释里


def test_a_missing_verb_is_named_not_folded_into_a_count(tmp_path: Path) -> None:
    payload = MODULE.audit(_root(tmp_path, MISSING_PROPOSE))
    assert payload["contract_verdict"] == "contract_incomplete"
    #: 钉"缺的是哪一个"：只报"缺 1 个"就允许把 propose 换成别的缺席还照样绿。
    assert payload["missing_verbs"] == ["propose"], payload["missing_verbs"]
    assert payload["contract"]["propose"]["matched_methods"] == []
    assert payload["contract"]["feedback"]["matched_methods"] == ["feedback"]


def test_the_real_tree_reads_incomplete_and_the_zero_is_scope_bounded() -> None:
    """现读钉值：语料训练链零引用（09 那句话成立的那一半），但生产面并非零引用（过宽的那一半要收）。"""

    payload = MODULE.audit(REPO)
    assert payload["contract_verdict"] == "contract_incomplete"
    assert payload["missing_verbs"] == ["propose", "snapshot_restore"], payload["missing_verbs"]
    assert payload["corpus_trainer_references_learner"] == 0
    #: 反面对照：同一个类在 `foundation_training.py` 里有引用 ⇒ "零真实调用"只能限定到语料链。
    assert payload["reference_sites"]["taiji/foundation_training.py"] > 0
    assert payload["reference_site_total"] > 0
    assert MODULE.LEARNER_FILE in payload["learner_file"]


def test_a_missing_source_file_is_census_failed_not_no_absence(tmp_path: Path) -> None:
    report = tmp_path / "out.json"
    #: 空目录 ⇒ 目标文件不在场，必须 rc=2 并写明"没算"，不许出 `missing_verbs == []` 那种假绿。
    rc = MODULE.main(["--root", str(tmp_path), "--out-report", str(report)])
    assert rc == 2, rc
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "census_failed"
    assert "contract_verdict" not in payload


def test_help_runs_and_lists_both_flags(monkeypatch) -> None:
    """`--help` 必须真的跑得起来并列出两个旗标（argparse 会对 help 做 % 插值，本仓踩过）。"""

    captured: list[str] = []
    monkeypatch.setattr(sys.stdout, "write", lambda text: captured.append(text) or len(text))
    try:
        MODULE.main(["--help"])
    except SystemExit as exit_info:
        assert exit_info.code == 0, exit_info.code
    else:
        raise AssertionError("--help 没有按惯例退出 0")
    joined = "".join(captured)
    assert "--out-report" in joined, joined[:200]
    assert "--root" in joined, joined[:200]
