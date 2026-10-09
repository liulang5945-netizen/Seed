"""DEBT-G61 修法②：`decision_consecutive_pressure_steps` 这一列的语义必须由真产品复现，不由散文规定。

面行里那一列直接取自产品 `AdaptiveResidualGrowthDecision.consecutive_pressure_steps`
（`scripts/training/train_seed_corpus.py` 的 `_record_pressure`，零重算），而㊵-534 实测到两件事：
提议行上它**恒为 0**、全场最大只有 **2**（而 `required_pressure_steps=3`）。当时只能给候选解释。
本测驱动真 trigger 复现该形态，把解释升级为实测语义：
**这一列是"提议后清零"之后才出版的读数**（`taiji/adaptive_residual_growth.py` 里
`:445` 自增 → `:448` 比持久项 → `:457` 提议即清零 → `:482` 才写进 decision 载荷），
所以它**永远取不到 `required_pressure_steps` 本身**，最大值是 `required-1`。

三支都能为假：
* 结构正例：提议行的出版值必须全为 0、提议前一段必须是 `1..required-1` 的连击、
  且整段里**从不出现 `required` 这个值**（若产品改成"先出版后清零"，这条必红）；
* 负对照：断链（不过阈）同样出版 0，所以"0"不能反推"刚提议"（若清零只发生在别处，这条必红）；
* 面头自述与产品行为同源：训练脚本必须出版 `consecutive_steps_column_semantics` 且值为
  `post_increment_reset_on_proposal`（缺键或改口径都要红）。
"""

from __future__ import annotations

from pathlib import Path

from taiji.adaptive_residual_growth import (
    AdaptiveResidualGrowthPolicy,
    AdaptiveResidualGrowthPressure,
    AdaptiveResidualGrowthTrigger,
)

REPO = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO / "scripts" / "training" / "train_seed_corpus.py"

#: 阈值全部读自产品自己的 policy，不抄字面量；`pressure` 的权重和为 1.0
#: （`:79-87`：0.30/0.25/0.20/0.25），所以均匀输入 `HOT` 时压强项也等于 `HOT`。
POLICY = AdaptiveResidualGrowthPolicy()
HOT = 0.9
COLD = 0.0
#: 足够长以跨过六条 EMA 阈并容纳多次提议；具体"第几步过阈"不由本测规定（那要重抄 EMA 生成链）。
STEPS = 24


def _pressure(tick: int, *, factor: float) -> AdaptiveResidualGrowthPressure:
    return AdaptiveResidualGrowthPressure.create(
        bridge_id="bridge-g61-semantics",
        tick=tick,
        residual_error=factor * HOT,
        fast_slow_conflict=factor * HOT,
        activity_saturation=factor * HOT,
        utility_gap=factor * HOT,
        resource_state=factor * HOT,
        evidence_id=f"evidence-{tick:04d}",
        parent_checkpoint_digest="digest-g61",
    )


def _run(factors: list[float]) -> list[tuple[int, bool]]:
    trigger = AdaptiveResidualGrowthTrigger(
        bridge_id="bridge-g61-semantics",
        policy=POLICY,
        parent_checkpoint_digest="digest-g61",
    )
    rows: list[tuple[int, bool]] = []
    for tick, factor in enumerate(factors, start=1):
        decision = trigger.observe(_pressure(tick, factor=factor), structural_budget=10**6)
        rows.append((int(decision.consecutive_pressure_steps), bool(decision.should_propose)))
    return rows


def test_published_column_is_the_post_reset_reading() -> None:
    required = int(POLICY.required_pressure_steps)
    rows = _run([HOT] * STEPS)
    published = [value for value, _ in rows]
    proposals = [index for index, (_, proposed) in enumerate(rows) if proposed]

    #: 复现㊵-534 的面形态：确有提议发生，而这一列的最大值仍到不了 `required`。
    assert proposals, "no proposal published on a fully hot run"
    assert max(published) == required - 1
    #: 提议行上的出版值恒为 0（这就是"这一列不能用来主张连续 N 步"的机器依据）。
    assert {published[index] for index in proposals} == {0}
    #: 每次提议之前的连击必须是 `1..required-1` 连续递增，之后从 1 重新起算。
    for index in proposals:
        assert published[max(0, index - required + 1) : index + 1] == (
            list(range(1, required)) + [0]
        )
        if index + 1 < len(published):
            assert published[index + 1] == 1
    #: `required` 这个值整段不出现——若产品改为"先出版后清零"，本条即红。
    assert required not in published


def test_zero_is_not_only_a_proposal_artifact() -> None:
    required = int(POLICY.required_pressure_steps)
    hot = [HOT] * STEPS
    rows = _run(hot)
    first = next(index for index, (_, proposed) in enumerate(rows) if proposed)
    #: 在"提议后重新起算的第一行"之后再插一步不过阈：这一列同样出版 0，但没有提议。
    broken = _run(hot[: first + 2] + [COLD])
    assert broken[-1] == (0, False)
    assert broken[-2] == (1, False)
    assert broken[first] == (0, True)
    #: 而纯热跑在同一个位置是 (1, False)——两支差异只在被插入的那一步。
    assert rows[first + 1] == (1, False)
    assert required >= 2


def test_face_header_publishes_the_same_semantics_as_the_product() -> None:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert '"consecutive_steps_column_semantics": "post_increment_reset_on_proposal"' in source
