"""B-4 覆盖率第四刀：Cortex 神经元生命周期（add / remove / isolate / revive）与装配 setter。

这是 `cortex.py` 里最大一块**可确定性单测**的表面（约 500 语句），也是审计 M12
"上帝对象把生命周期与推理揉在一起"的回归位。夹具走生产构造路径
`neuroplex.loader.create_cortex()`（fallback 单神经元，秒级），`neurons_dir` 一律指到 tmp
⇒ ckpt 写盘不落在仓库里。

断言钉各处 docstring 声明的契约，不是"调用没炸"：
* add：`{domain}_{n}` 递增命名、同一对象同时进 cortex 与 ensemble、ckpt 落盘、eval 态、
  未知 domain 与不存在的 `from_split` 必须 ValueError；
* remove：**最后一个神经元拒绝移除**（ensemble 不能空）、`delete_ckpt` 两条分支、
  其他神经元的 excite/inhibit 通道引用要跟着清；
* isolate：先把**运行时最新权重**写回 ckpt 再摘除 ⇒ 改一个权重后 isolate→revive
  必须拿回改后的值（"隔离保留权重、复活继续运行"契约；旧实现只恢复 add 时的初始副本）；
* revive：不在隔离池 / ckpt 不在盘上都要 False，成功时从隔离池移出。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from neuroplex.brain.cortex import Cortex

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module():
    """复用 C-1 的夹具构造代码（两处各写一遍必漂）。"""

    path = PROJECT_ROOT / "scripts" / "training" / "capture_taiji_cortex_rolling_nll_golden.py"
    spec = importlib.util.spec_from_file_location("c1_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()


@pytest.fixture()
def cortex(tmp_path):
    built, _hub, _general_sp, _zh_sp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    os.makedirs(built.neurons_dir, exist_ok=True)
    return built


def _param_snapshot(neuron) -> dict:
    return {k: v.detach().clone() for k, v in neuron.state_dict().items() if v.is_floating_point()}


def _max_diff(a: dict, b: dict) -> float:
    shared = set(a) & set(b)
    assert shared, "两份 state_dict 没有共同参数 ⇒ 对比是空的"
    return max(float((a[k] - b[k]).abs().max()) for k in shared)


def _disk_params(path: str) -> dict:
    """读盘上的 ckpt，取同样的浮点参数子集（与 _param_snapshot 同口径）。"""

    from neuroplex.legacy_checkpoint import load_legacy_checkpoint

    return _param_snapshot(
        type(
            "_Disk",
            (),
            {
                "state_dict": lambda self: load_legacy_checkpoint(path, map_location="cpu")[
                    "state_dict"
                ]
            },
        )()
    )


# ── add_neuron ────────────────────────────────────────────────────────────────


def test_add_neuron_registers_everywhere_and_increments_id(cortex: Cortex) -> None:
    before = set(cortex.neurons)
    nid = cortex.add_neuron("zh")

    assert nid == "zh_1", f"命名应为 {{domain}}_{{n}}，实得 {nid}"
    assert set(cortex.neurons) == before | {nid}
    # cortex.neurons 与 ensemble.neurons 必须是同一份（不是拷贝）——否则热插拔后前向看不见新神经元
    assert cortex.neurons is cortex.ensemble.neurons or set(cortex.ensemble.neurons) == set(
        cortex.neurons
    )
    assert cortex.ensemble.neurons[nid] is cortex.neurons[nid]
    assert (
        cortex.neurons[nid].training is False
    ), "新神经元必须置于 eval（否则推理期会更新 BN/dropout 态）"
    assert os.path.exists(os.path.join(cortex.neurons_dir, f"neuron_{nid}.pt")), "ckpt 必须落盘"

    # 同域再加一个 ⇒ 序号递增且不覆盖
    nid2 = cortex.add_neuron("zh")
    assert nid2 == "zh_2" and nid not in (nid2,)
    assert len(cortex.neurons) == len(before) + 2


def test_add_neuron_rejects_unknown_domain_and_missing_parent(cortex: Cortex) -> None:
    with pytest.raises(ValueError):
        cortex.add_neuron("nope")
    with pytest.raises(ValueError):
        cortex.add_neuron("en", from_split="ghost_1")
    assert not any(n.startswith("nope") for n in cortex.neurons), "被拒的添加不能留下半成品"


def test_add_neuron_from_split_inherits_parent_spec(cortex: Cortex) -> None:
    parent = cortex.add_neuron("zh")
    child = cortex.add_neuron("zh", from_split=parent)
    assert child not in (None, "") and child != parent
    assert (
        cortex.neurons[child].config.spec == cortex.neurons[parent].config.spec
    ), "分裂子神经元必须继承父规格（同域同规格分化）"


# ── remove_neuron ─────────────────────────────────────────────────────────────


def test_remove_neuron_refuses_last_and_honours_delete_ckpt(cortex: Cortex) -> None:
    nid = cortex.add_neuron("zh")
    ckpt = os.path.join(cortex.neurons_dir, f"neuron_{nid}.pt")
    assert os.path.exists(ckpt)

    assert cortex.remove_neuron(nid, delete_ckpt=False) is True
    assert nid not in cortex.neurons and nid not in cortex.ensemble.neurons
    assert os.path.exists(ckpt), "delete_ckpt=False 必须保留 ckpt"

    # 现在只剩 fallback 的那一个 ⇒ 拒绝清空 ensemble
    remaining = list(cortex.neurons)
    assert len(remaining) == 1
    assert cortex.remove_neuron(remaining[0]) is False, "最后一个神经元必须拒绝移除"
    assert list(cortex.neurons) == remaining

    assert cortex.remove_neuron("ghost") is False, "不存在的 id 必须返回 False 而不是抛"


def test_remove_neuron_clears_side_channel_references(cortex: Cortex) -> None:
    victim = cortex.add_neuron("zh")
    other = cortex.add_neuron("en")
    target = cortex.neurons[other]
    if not hasattr(target, "excite_channels"):
        pytest.skip("该神经元实现没有 side channel 容器，本条无从适用")
    # 造一条指向 victim 的通道，移除后必须一起清掉（悬空引用会在前向时炸或静默错配）
    target.excite_channels[victim] = torch.nn.Linear(2, 2)
    assert victim in target.excite_channels
    assert cortex.remove_neuron(victim) is True
    assert victim not in target.excite_channels, "excite_channels 残留悬空引用"


# ── isolate / revive ──────────────────────────────────────────────────────────


def test_isolate_then_revive_preserves_runtime_weights(cortex: Cortex) -> None:
    """契约核心：隔离前把运行时最新权重写回 ckpt ⇒ 复活拿回的必须是改后的值。"""

    nid = cortex.add_neuron("zh")
    neuron = cortex.neurons[nid]
    before = _param_snapshot(neuron)

    # 模拟睡眠/整合在运行期写入权重
    with torch.no_grad():
        for tensor in neuron.state_dict().values():
            if tensor.is_floating_point():
                tensor.add_(0.25)
    mutated = _param_snapshot(cortex.neurons[nid])
    assert _max_diff(before, mutated) > 0.1, "先确认改权重真的生效，否则本条是自证"

    assert cortex.isolate_neuron(nid) is True
    assert nid not in cortex.neurons
    assert cortex.get_isolated_neurons() == [nid]

    assert cortex.revive_neuron(nid) is True
    revived = cortex.neurons[nid]
    assert (
        _max_diff(_param_snapshot(revived), mutated) < 1e-6
    ), "复活必须带回归一化前的运行时权重（拿回 add 时的初始副本＝契约破）"
    assert _max_diff(_param_snapshot(revived), before) > 0.1
    assert cortex.get_isolated_neurons() == [], "复活后必须移出隔离池"


def test_isolate_refuses_last_and_revive_rejects_bad_state(cortex: Cortex) -> None:
    assert cortex.isolate_neuron("ghost") is False
    only = list(cortex.neurons)
    assert len(only) == 1
    assert cortex.isolate_neuron(only[0]) is False, "最后一个神经元拒绝隔离（与 remove 同规则）"

    # 不在隔离池 ⇒ False；在池里但 ckpt 丢了 ⇒ False（不能静默造一个空壳神经元）
    nid = cortex.add_neuron("zh")
    assert cortex.revive_neuron("zh_ghost") is False
    assert cortex.isolate_neuron(nid) is True
    os.remove(os.path.join(cortex.neurons_dir, f"neuron_{nid}.pt"))
    assert cortex.revive_neuron(nid) is False
    assert nid not in cortex.neurons, "ckpt 缺失时复活失败不得留下半装配的神经元"


def test_isolated_neuron_keeps_domain_and_ckpt_in_recovery_record(cortex: Cortex) -> None:
    nid = cortex.add_neuron("code")
    assert cortex.isolate_neuron(nid) is True
    record = cortex._isolated[nid]
    assert record["domain"] == "code"
    assert record["ckpt"].endswith(f"neuron_{nid}.pt") and os.path.exists(record["ckpt"])
    assert nid not in cortex.neurons and nid in cortex.get_isolated_neurons()


# ── 装配 setter ───────────────────────────────────────────────────────────────


def test_general_tokenizer_and_alignment_cache_setters_take_effect(cortex: Cortex) -> None:
    from neuroplex.resonance.translator import TokenizerHub

    sp = capture.build_cortex()[3]  # zh SentencePiece，用于换一个不同的 general tokenizer
    cortex.set_general_tokenizer(sp)
    assert cortex._general_sp is sp

    hub = TokenizerHub(general_tokenizer=sp)
    hub.register_domain("general", sp)
    cortex.set_tokenizer_hub(hub)
    assert cortex._tokenizer_hub is hub
    assert cortex._tokenizer_hub.get_tokenizer("general") is sp

    cortex.set_alignment_rules({"zh": {0: [1, 2]}})
    cortex._domain_to_general_cache = {"zh": {"sentinel": 1}}
    cortex.invalidate_alignment_cache("zh")
    assert "zh" not in getattr(cortex, "_domain_to_general_cache", {}), "按域失效必须真删该域"

    cortex._domain_to_general_cache = {"zh": {"s": 1}, "en": {"s": 2}}
    cortex.invalidate_alignment_cache()
    assert cortex._domain_to_general_cache == {}, "不带 domain 时清空全缓存"


def test_add_neuron_never_reuses_an_isolated_id(cortex: Cortex) -> None:
    """命名必须避开隔离池（缺陷修复守卫，实测 2026-09-26）。

    旧实现只扫 `self.neurons` ⇒ 被隔离的神经元（不在字典里、ckpt 还在盘上）的 id 会被
    新神经元复用并覆盖其 ckpt；随后 `revive_neuron` 命中"已在运行中"的早退分支，
    **报成功却恢复成别人的权重**，隔离池里还留下永久清不掉的条目。
    """

    from neuroplex.legacy_checkpoint import load_legacy_checkpoint  # noqa: F401  (helper 内已用)

    cortex.add_neuron("math")  # math_1
    victim = cortex.add_neuron("math")  # math_2
    neuron = cortex.neurons[victim]
    with torch.no_grad():
        for tensor in neuron.state_dict().values():
            if tensor.is_floating_point():
                tensor.add_(0.5)
                break
    runtime = _param_snapshot(neuron)

    assert cortex.isolate_neuron(victim) is True
    ckpt = os.path.join(cortex.neurons_dir, f"neuron_{victim}.pt")
    on_disk = _disk_params(ckpt)
    assert _max_diff(on_disk, runtime) < 1e-5, "isolate 应已把运行时权重写回 ckpt"

    fresh = cortex.add_neuron("math")
    assert fresh != victim, f"新神经元复用了隔离池里的 id {victim} ⇒ 会覆盖它的 ckpt"
    assert _max_diff(on_disk, _disk_params(ckpt)) < 1e-5, "被隔离神经元的 ckpt 被改写了"

    assert cortex.revive_neuron(victim) is True
    assert (
        _max_diff(_param_snapshot(cortex.neurons[victim]), runtime) < 1e-5
    ), "复活必须拿回被隔离者自己的权重"
    assert cortex.get_isolated_neurons() == [], "成功复活后隔离池必须清空"
