"""Resonance Field Architecture — 共振场架构模块.

三层架构：
- Layer 1 (共享感官): shared_embedding(256000, 512)，所有 neuron 共用
- Layer 2 (认知空间): per-neuron embed_adapter + Transformer body
- Layer 3 (神经语言): 4096-dim 共振场

P7 对齐机制：build_position_alignment 通过字符 span 重叠对齐 general/domain
token，batch_align_and_embed 查共享嵌入表。通用词表可热插拔，不影响 neuron 内部。

Core components:
- ResonanceField: shared 4096-dim field with L2-normalized writes
- ResonanceNeuron: wraps Transformer backbone with field_write/field_read
- ResonanceEnsemble: multi-round resonance loop
- TokenizerHub: domain-specific tokenizer hot-swap
"""

from .config import (
    COMPACT,
    DEFAULT_NEURON_SPEC,
    DOMAIN_VOCAB_SIZES,
    EXPERT,
    FOUNDATION,
    GENERAL_TOKENIZER_DOMAIN,
    MICRO,
    STANDARD,
    TINY_TEST,
    NeuronConfig,
    get_default_neuron_config,
    get_domain_neuron_config,
)
from .ensemble import ResonanceEnsemble
from .field import ResonanceField
from .gamma_oscillator import GammaOscillator, apply_gamma_gate
from .geometry import NeuronGeometry
from .lifecycle import ApoptosisTracker, LifecycleManager, MaturityTracker, NeurogenesisTrigger
from .neuro_modulation import NeuromodulatorState, SleepConsolidator
from .neuron import ResonanceNeuron
from .phasor import PhasorDynamics
from .stdp import FiringRecord, STDPRule, STDPTracker
from .topology import (
    build_topology,
    establish_topology_channels,
    infer_topology_from_state,
    topology_detail,
    topology_summary,
)
from .translator import (
    TokenizerHub,
    batch_align_and_embed,
    build_position_alignment,
)

__all__ = [
    # Core
    "ResonanceField",
    "ResonanceNeuron",
    "ResonanceEnsemble",
    "NeuronConfig",
    "MICRO",
    "COMPACT",
    "STANDARD",
    "FOUNDATION",
    "EXPERT",
    "TINY_TEST",
    "DEFAULT_NEURON_SPEC",
    "get_default_neuron_config",
    "DOMAIN_VOCAB_SIZES",
    "GENERAL_TOKENIZER_DOMAIN",
    "get_domain_neuron_config",
    # Translator
    "TokenizerHub",
    "build_position_alignment",
    "batch_align_and_embed",
    # 生命周期
    "LifecycleManager",
    "ApoptosisTracker",
    "MaturityTracker",
    "NeurogenesisTrigger",
    # STDP
    "STDPTracker",
    "STDPRule",
    "FiringRecord",
    # 神经调质
    "NeuromodulatorState",
    "SleepConsolidator",
    # Gamma 同步
    "GammaOscillator",
    "apply_gamma_gate",
    "PhasorDynamics",
    # RSGN 几何
    "NeuronGeometry",
    # S7 拓扑
    "build_topology",
    "establish_topology_channels",
    "infer_topology_from_state",
    "topology_summary",
    "topology_detail",
]
