"""checkpoint 往返等价性门禁（plans/active/roadmap/04_EXECUTION_PLAN.md §3 line 61）。

准入原文：「创建 checkpoint → 关闭运行时 → 恢复 → 继续一步 → 对 lineage、预算、
结构、provider artifact 和可见指标做等价性断言」。在本门禁转绿之前，数据集与续训
工作只算「数据集可发现性与 API 契约修复」，不构成训练能力宣称。

同时钉住 02_GATES_AND_CI.md §14.3「checkpoint 往返对称不变量」：
`checkpoint()` 写出的每一个键都必须能被 `restore()` 消费，否则长训练会在最后一步
失败并丢弃整个 checkpoint。
"""

from __future__ import annotations

import queue
from pathlib import Path

import torch

from api.training import resume
from seed import Seed, SeedConfig
from seed.language_provider import attach_native_language_provider
from taiji import (
    AdaptiveNeuronRegion,
    EpisodicMemoryStore,
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageProviderArtifact,
    SemanticMemoryLearner,
)


class _StubDecoder:
    """外部成熟解码器的最小替身：只实现 TextDecoder 协议。"""

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        return f"decoded:{prompt}"


def _wire_runtime(episode_id: str) -> Seed:
    """构造一个真实带器官的运行时，避免门禁在空组件上空转。

    裸 Seed + learn_bytes 会让约 25 个组件保持 None，
    等价性断言在这种情况下会退化为「都是空的所以相等」。
    """
    model = Seed(SeedConfig(), episode_id=episode_id)
    architecture = model.architecture
    architecture.ensure_native_executive()
    architecture.ensure_homeostatic_controller()
    attach_native_language_provider(architecture)
    architecture.attach_episodic_memory(EpisodicMemoryStore(capacity=8))
    architecture.attach_semantic_memory(SemanticMemoryLearner(architecture.perception.feature_dim))
    architecture.attach_adaptive_neuron_region(
        AdaptiveNeuronRegion(
            region_id="adaptive.cortex",
            input_dim=5,
            unit_ids=("u0", "u1"),
            fan_in=2,
            generator=torch.Generator().manual_seed(7),
        )
    )
    return model


def _commit_real_history(model: Seed) -> None:
    """写入一条真实情节记忆与一次真实结构生长，让等价性有内容可比。"""
    architecture = model.architecture
    architecture.observe(97, learn=False)
    architecture.act((97, 98), sample=False)
    architecture.settle_action(1.0, learn=False)

    proposal = architecture.propose_neuron_add(
        region_id="adaptive.cortex",
        unit_id="u2",
        evidence_ids=("runtime:checkpoint-gate",),
    )
    assert architecture.commit_neuron_add(proposal) is True, "结构生长必须真实落账"


def _attach_guarded_provider(architecture) -> LanguageProviderArtifact:
    """按产品的 guarded 接入方式挂载外接解码器（backend spec → artifact → organ）。"""
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id="mature-decoder-v1",
            family="external-causal-decoder-guarded",
            training_contract="expression-to-text-v1",
        )
    )
    architecture.attach_language_backend_registry(registry)

    artifact = LanguageProviderArtifact(
        artifact_id="qwen-lora-v1",
        backend_id="mature-decoder-v1",
        mode="guarded",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path="providers/qwen-lora-v1",
    )
    architecture.attach_language_provider_artifact(artifact)
    architecture.attach_language_organ(
        ExternalTextDecoderLanguageOrgan(
            _StubDecoder(),
            prompt_builder=lambda expression: expression.content_id,
            backend_id="mature-decoder-v1",
        )
    )
    return artifact


def _fingerprint(model: Seed) -> dict[str, object]:
    """采集五个准入维度的可观测指纹。"""
    architecture = model.architecture
    snapshot = architecture.cognitive_snapshot()
    episodic = architecture._episodic_memory
    artifact = architecture.language_provider_artifact
    organ = architecture.language_organ
    proposals = architecture.topology_proposals
    regions = architecture.neuron_regions
    return {
        "tick": model.tick,
        "structural_budget": snapshot.development.structural_budget,
        "unit_ids": tuple(region.unit_ids for region in regions),
        "topology_count": len(proposals),
        "topology_status": tuple(proposal.status for proposal in proposals),
        "episodic_count": None if episodic is None else episodic.count,
        "homeostasis": architecture.homeostatic_state(),
        "artifact_id": None if artifact is None else artifact.artifact_id,
        "organ_backend": None if organ is None else organ.backend_id,
    }


def test_native_checkpoint_roundtrip_preserves_all_admission_dimensions(
    tmp_path: Path, monkeypatch
) -> None:
    """通过真实落盘路径往返后，lineage/预算/结构/provider/可见指标必须逐项等价。"""
    monkeypatch.setattr(resume, "PROGRESS_EVERY", 100)
    monkeypatch.setattr(resume, "CHECKPOINT_EVERY", 10_000_000)

    model = _wire_runtime("ckpt-gate-native")
    _commit_real_history(model)

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "' + "ab" * 300 + '"}\n', encoding="utf-8")
    save_path = tmp_path / "ckpt.pt"

    resume._train_worker(
        model,
        [corpus],
        400,
        save_path,
        queue.Queue(maxsize=4096),
        200,
    )
    assert save_path.exists(), "_train_worker 必须在收尾时落盘"

    envelope = torch.load(save_path, map_location="cpu", weights_only=False)
    restored = Seed.from_checkpoint(envelope)

    before = _fingerprint(model)
    after = _fingerprint(restored)
    assert before["topology_count"] == 1, "前置条件：必须存在一条真实结构提案"
    assert before["unit_ids"] == (("u0", "u1", "u2"),), "前置条件：结构生长必须已落账"
    for key, expected in before.items():
        assert after[key] == expected, (
            f"往返后 {key} 不等价：期望 {expected!r}，实测 {after[key]!r}"
        )

    left = model.observe(97, learn=True)
    right = restored.observe(97, learn=True)
    assert left.predicted_symbol == right.predicted_symbol, "恢复后继续一步的预测必须与原运行时一致"
    assert torch.equal(left.probabilities, right.probabilities), (
        "恢复后继续一步的概率分布必须逐元素一致"
    )
    assert model.tick == restored.tick, "继续一步后 lineage tick 必须仍然对齐"


def test_guarded_provider_checkpoint_can_be_restored(tmp_path: Path) -> None:
    """外接成熟解码器运行时的 checkpoint 必须可被恢复。

    `checkpoint()` 无条件序列化任意已挂载的语言器官，若 `restore()` 只认
    native-readable/structured-stub，则接入 guarded provider 的产品运行时
    永远无法从自己的存档启动——这正是 §14.3 要拦截的往返不对称。
    """
    model = _wire_runtime("ckpt-gate-guarded")
    architecture = model.architecture

    architecture.attach_language_organ(None)
    _attach_guarded_provider(architecture)
    _commit_real_history(model)

    envelope = model.checkpoint()
    assert envelope["taiji"]["components"]["language_organ"] is not None, (
        "前置条件：checkpoint 确实写出了外部器官载荷"
    )

    restored = Seed.from_checkpoint(envelope)

    restored_artifact = restored.architecture.language_provider_artifact
    assert restored_artifact is not None, "provider artifact 必须随 checkpoint 保留"
    assert restored_artifact.artifact_id == "qwen-lora-v1"
    assert restored_artifact.mode == "guarded"
    assert restored.architecture.neuron_regions[0].unit_ids == ("u0", "u1", "u2"), (
        "guarded 运行时的结构生长同样必须往返保真"
    )
    assert restored.architecture.detached_language_organ_backend == "mature-decoder-v1", (
        "外接器官脱挂必须可观测，否则运行时无法区分「降级」与「本来就是原生」"
    )
    assert restored.architecture.language_organ is None, (
        "外接解码器权重不在存档内，恢复后不得伪造一个终端器官"
    )


def test_rebinding_runtime_organ_clears_detached_marker() -> None:
    """运行时重新挂载外接器官后，脱挂标记必须清除，避免持续误报降级。"""
    model = _wire_runtime("ckpt-gate-rebind")
    architecture = model.architecture
    architecture.attach_language_organ(None)
    _attach_guarded_provider(architecture)

    restored = Seed.from_checkpoint(model.checkpoint())
    assert restored.architecture.detached_language_organ_backend == "mature-decoder-v1"

    restored.architecture.attach_language_organ(
        ExternalTextDecoderLanguageOrgan(
            _StubDecoder(),
            prompt_builder=lambda expression: expression.content_id,
            backend_id="mature-decoder-v1",
        )
    )
    assert restored.architecture.detached_language_organ_backend is None, (
        "重新绑定运行时后不得继续报告脱挂"
    )
    assert restored.architecture.language_organ is not None
