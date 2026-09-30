"""A30 出厂装配守卫：带回路信封成为默认加载源＋两条污染门槛（owner 裁定 §7-1）。

钉四件事：
1. **默认档翻转**：`DEFAULT_CHECKPOINT`＝带回路信封（sha 与 A-29 §7 守卫-B 落盘件逐位同），
   厂档 `seed_beta.pt` 原位不动（逃生口/对照面）。
2. **两枚面的载入行为**：默认档 ⇒ 电路挂载＋证据门开（A-25/A-28 的 restore 路）＋门槛 armed；
   厂档 ⇒ 无回路（§2t 的出厂面）。
3. **判据逐位性**：`seed.surface_gate` 的 `well_formed`/`mean_nll` 与仪器
   `diag_taiji_r2_surface_decode` 在真中文/退化样本上判定与数值逐位相同（漂移即红）。
4. **门槛接线**：武装装配上回写被门槛拦截/放行（monkeypatch 判定，真链观测 learn_bytes），
   出厂面回写不受门槛。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: A-29 §7 守卫-B 落盘件（output/a28_product_face/seed_beta_with_circuit_0.pt）的 sha——
#: 出厂信封是它的逐位拷贝。*.pt 不进 git（DEBT-I7），所以这条钉的是**本机磁盘态**；
#: 重建用 build_circuit_carried_envelope（torch.save 元数据不逐位，重建件 sha 会不同——已登记）。
#: 2026-10-01 重出（owner 裁定 §7c，PLAN-A-30 §2bh）：默认换成自写档候选 `a31_chunked_self`
#: 烤成的信封（三线全过＋F0 floor_pass）。旧件 `seed_beta_with_circuit.pt`（sha f9343433…）
#: 仍留在盘上＝回滚点，一并钉死以便回滚路径可验证。
SEALED_ENVELOPE_SHA256 = "d6169a358eaee6d194d4795e3167a7bcbb42dfde57b92aa1199bcebbed89699b"
ROLLBACK_ENVELOPE_SHA256 = "f9343433d519084659926d0bdd689efef96c3bee02e79bc33f441fd177d8baa8"
CANDIDATE_BASE_SHA256 = "ca2628077b21bc4c9a6f2fc06ff410d70311b1218a30b8794ecfd7e5a027d8bb"
FACTORY_BASE_SHA256 = "ad2a06465e0ef78c75aff7302a6f7a2ae11825d2f19361d9d65926006f3bf793"
NGRAM_ARTIFACT_SHA256 = "4a40da9cdd7cc938af326d3e3b4d1055a220b37809a8650ef045e33a96300f17"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_checkpoint_points_at_circuit_envelope_and_files_are_pinned() -> None:
    from api.seed_runtime import DEFAULT_CHECKPOINT, FACTORY_CHECKPOINT

    assert DEFAULT_CHECKPOINT.name == "seed_a31self_with_circuit.pt"
    assert DEFAULT_CHECKPOINT.is_file()
    assert _sha256(DEFAULT_CHECKPOINT) == SEALED_ENVELOPE_SHA256
    assert FACTORY_CHECKPOINT.name == "seed_beta.pt"
    assert _sha256(FACTORY_CHECKPOINT) == FACTORY_BASE_SHA256

    artifact = DEFAULT_CHECKPOINT.parent / "seed_surface_ngram.lzma"
    assert artifact.is_file()
    assert _sha256(artifact) == NGRAM_ARTIFACT_SHA256


def test_rollback_point_and_candidate_base_are_pinned() -> None:
    """重出不是单程票：旧默认件与候选基底都要在盘且逐位在册（回滚/重建可核）。"""

    checkpoints = Path(__file__).resolve().parents[2] / "checkpoints"
    rollback = checkpoints / "seed_beta_with_circuit.pt"
    assert rollback.is_file()
    assert _sha256(rollback) == ROLLBACK_ENVELOPE_SHA256

    candidate = Path(__file__).resolve().parents[2] / "output" / "a31_chunked_self" / "checkpoint.pt"
    assert candidate.is_file()
    assert _sha256(candidate) == CANDIDATE_BASE_SHA256


def test_default_load_mounts_circuit_and_arms_surface_gate() -> None:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load()
    try:
        assert runtime.model.substrate.copy_circuit is not None
        assert runtime.model.substrate._copy_evidence_utf8_gate_override is True
        assert runtime.surface_gate_state == "armed"
    finally:
        del runtime


def test_factory_face_stays_circuit_free() -> None:
    from api.seed_runtime import FACTORY_CHECKPOINT, SeedRuntime

    runtime = SeedRuntime.load(FACTORY_CHECKPOINT)
    try:
        assert runtime.model.substrate.copy_circuit is None
    finally:
        del runtime


def test_surface_gate_criterion_matches_instrument_bitwise() -> None:
    import json
    import random

    import diag_taiji_r2_surface_decode as instrument

    from seed import surface_gate

    model = surface_gate.load_surface_ngram(
        PROJECT_ROOT / "checkpoints" / "seed_surface_ngram.lzma"
    )
    assert model[2] == 8899 and model[3] == 117379360

    samples: list[str] = []
    with (PROJECT_ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl").open(
        "r", encoding="utf-8", errors="replace"
    ) as handle:
        for index, raw in enumerate(handle):
            if index >= 500:
                break
            try:
                text = "".join((json.loads(raw).get("text") or "").split())
            except ValueError:
                continue
            if len(text) >= 12:
                samples.append(text[:32])
    rng = random.Random(19)
    pool = rng.sample(samples, 20)
    #: 真中文＋两类退化样本都要逐位同判（NLL 数值也逐位同）。
    pool += ["君" * 40, "是是是是是是是是是是是是", "hello world 123", "", "ab"]
    for text in pool:
        assert surface_gate.well_formed(text, model) == instrument.well_formed(text, model), text
        if len("".join(text.split())) >= 2:
            assert surface_gate.mean_nll(text, model) == instrument.mean_nll(text, model), text


def test_write_back_gate_units() -> None:
    from seed import surface_gate

    model = surface_gate.load_surface_ngram(
        PROJECT_ROOT / "checkpoints" / "seed_surface_ngram.lzma"
    )
    markers = ("\n问：", "问：")
    budget = 256

    #: 真中文答复＋自然收口（发出轮界）⇒ 放行。
    ok, reason = surface_gate.write_back_allowed(
        "今天天气很好，我们一起去公园散步。",
        "今天天气很好，我们一起去公园散步。\n问：".encode(),
        model,
        turn_markers=markers,
        budget=budget,
    )
    assert (ok, reason) == (True, "passed")

    #: 同字拖写（§2v 的污染形态；60 字＝180 字节，预算内自然收口）⇒ well_formed 拦下。
    ok, reason = surface_gate.write_back_allowed(
        "君" * 60, ("君" * 60).encode("utf-8"), model, turn_markers=markers, budget=budget
    )
    assert (ok, reason) == (False, "not_well_formed")

    #: 吃满预算且无轮界＝未收口 ⇒ 长度上限拦下。
    ok, reason = surface_gate.write_back_allowed(
        "今天天气很好，我们一起去公园散步。" * 20,
        ("今天天气很好，我们一起去公园散步。" * 20).encode("utf-8"),
        model,
        turn_markers=markers,
        budget=budget,
    )
    assert (ok, reason) == (False, "not_ended_naturally")


def test_history_filter_drops_bad_replies() -> None:
    from seed import surface_gate

    model = surface_gate.load_surface_ngram(
        PROJECT_ROOT / "checkpoints" / "seed_surface_ngram.lzma"
    )
    good = ("今天天气如何", "今天天气很好，适合出门散步。")
    bad = ("你是谁", "君" * 100)
    empty = ("", "没有问题的一轮")

    filtered = surface_gate.filter_history([good, bad, empty], model)
    assert filtered == [good]


@pytest.mark.parametrize("gate_verdict", [True, False])
def test_armed_write_back_consults_the_gate(
    gate_verdict: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.seed_runtime import SeedRuntime
    from seed import surface_gate as gate_module

    monkeypatch.setattr(
        gate_module,
        "write_back_allowed",
        lambda *a, **k: (gate_verdict, "patched"),
    )
    runtime = SeedRuntime.load()
    calls: list[bytes] = []
    monkeypatch.setattr(
        runtime.model,
        "learn_bytes",
        lambda data, **k: calls.append(data) or {"observations": float(len(data))},
    )
    runtime.chat("你好", max_length=64, learn=True)
    assert runtime.last_write_back_gate == (gate_verdict, "patched")
    assert (len(calls) == 1) is gate_verdict


def test_factory_face_write_back_ignores_the_gate() -> None:
    from api.seed_runtime import FACTORY_CHECKPOINT, SeedRuntime

    runtime = SeedRuntime.load(FACTORY_CHECKPOINT)
    calls: list[bytes] = []

    def _spy(data: bytes, **k: object) -> dict[str, float]:
        calls.append(data)
        return {"observations": float(len(data))}

    runtime.model.learn_bytes = _spy  # type: ignore[method-assign]
    runtime.chat("你好", max_length=64, learn=True)
    assert runtime.last_write_back_gate == (True, "factory_face_or_gate_disarmed")
    assert len(calls) == 1
