"""Native raw-byte sensory and motor organs for Taiji."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .config import TaijiConfig
from .developmental_synapse import DevelopmentalSynapseBank
from .sparse import SparseSynapses, bound_norm
from .utf8_state import UTF8_POSITION_DIM


class ByteSensor:
    """Map each raw byte and the boundary marker to one receptor population."""

    def __init__(self, config: TaijiConfig, *, device: torch.device | str = "cpu"):
        if config.alphabet_size < 257:
            raise ValueError("ByteSensor requires 256 bytes plus one boundary symbol")
        self.config = config
        self.device = torch.device(device)

    def encode(self, symbol: int) -> torch.Tensor:
        if not 0 <= int(symbol) < self.config.alphabet_size:
            raise ValueError(f"symbol {symbol} is outside the sensor alphabet")
        value = torch.zeros(self.config.alphabet_size, device=self.device)
        value[int(symbol)] = 1.0
        return value

    def symbols(
        self,
        data: bytes,
        *,
        include_boundary: bool = True,
        include_start_boundary: bool | None = None,
        include_end_boundary: bool | None = None,
    ) -> tuple[int, ...]:
        """Return a byte stream with independently controlled edge markers.

        ``include_boundary`` remains the compatibility switch for callers
        that want both markers.  The split controls let a resumable learner
        feed adjacent chunks without inserting a synthetic boundary between
        them: only the first chunk receives the opening marker and only the
        last chunk receives the closing marker.
        """

        if not isinstance(include_boundary, bool):
            raise TypeError("include_boundary must be a bool")
        if include_start_boundary is None:
            include_start_boundary = include_boundary
        if include_end_boundary is None:
            include_end_boundary = include_boundary
        if not isinstance(include_start_boundary, bool):
            raise TypeError("include_start_boundary must be a bool or None")
        if not isinstance(include_end_boundary, bool):
            raise TypeError("include_end_boundary must be a bool or None")
        body = tuple(int(value) for value in data)
        boundary = self.config.boundary_symbol
        prefix = (boundary,) if include_start_boundary else ()
        suffix = (boundary,) if include_end_boundary else ()
        return (*prefix, *body, *suffix)


#: R2 组合绑定实验：分块受力面用**独立** generator 的固定种子。
#: 独立是为了不动主 RNG 流 —— 开/关两臂的其它器官拓扑因此逐位相同（见 BytePredictiveContext.__init__）。
RECEPTOR_FACTOR_SEED = 20260923


class SparseReceptorBank:
    """Fold every cortical signal into a shared, bounded motor evidence space.

    Each input coordinate has exactly one fixed excitatory or inhibitory edge.
    Inputs are assigned evenly across receptor channels, so sparse compression
    never makes a cortical coordinate invisible to the action population.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: torch.Generator,
        context_norm: float,
        device: torch.device | str = "cpu",
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("receptor dimensions must be positive")
        if out_features > in_features:
            raise ValueError("receptor count cannot exceed cortical inputs")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.context_norm = float(context_norm)
        self.device = torch.device(device)

        order = torch.randperm(self.in_features, generator=generator)
        channel = torch.empty(self.in_features, dtype=torch.long)
        channel[order] = torch.arange(self.in_features) % self.out_features
        polarity = (torch.randint(0, 2, (self.in_features,), generator=generator) * 2 - 1).to(
            torch.float32
        )
        counts = torch.bincount(channel, minlength=self.out_features).to(torch.float32)

        self.channel = channel.to(self.device)
        self.polarity = polarity.to(self.device)
        self.channel_scale = counts.rsqrt().to(self.device)

    def forward(self, cortical_state: torch.Tensor) -> torch.Tensor:
        if cortical_state.shape != (self.in_features,):
            raise ValueError(
                f"cortical state shape must be ({self.in_features},), "
                f"got {tuple(cortical_state.shape)}"
            )
        context = torch.zeros(self.out_features, device=self.device)
        context.scatter_add_(
            0,
            self.channel,
            cortical_state.to(self.device) * self.polarity,
        )
        context.mul_(self.channel_scale)
        norm = context.norm()
        if float(norm.item()) < 1e-8:
            return context
        scaled: torch.Tensor = context * (self.context_norm / norm)
        return scaled

    def to_payload(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "context_norm": self.context_norm,
            "channel": self.channel.detach().cpu().clone(),
            "polarity": self.polarity.detach().cpu().clone(),
        }

    def _payload_tensors(
        self, payload: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.in_features, self.out_features, self.context_norm)
        actual = (
            int(payload["in_features"]),
            int(payload["out_features"]),
            float(payload["context_norm"]),
        )
        if actual != expected:
            raise ValueError("receptor payload does not match architecture")
        channel = payload["channel"].detach().to(self.device, dtype=torch.long)
        polarity = payload["polarity"].detach().to(self.device, dtype=torch.float32)
        if channel.shape != (self.in_features,) or polarity.shape != (self.in_features,):
            raise ValueError("receptor payload shape does not match architecture")
        if bool((channel < 0).any()) or bool((channel >= self.out_features).any()):
            raise ValueError("receptor payload channel is outside the context space")
        if not bool(((polarity == -1.0) | (polarity == 1.0)).all()):
            raise ValueError("receptor payload polarity must be binary")
        counts = torch.bincount(channel, minlength=self.out_features).to(torch.float32)
        if bool((counts == 0).any()):
            raise ValueError("receptor payload leaves a context channel unconnected")
        return channel, polarity, counts.rsqrt().to(self.device)

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        channel, polarity, _channel_scale = self._payload_tensors(payload)
        if not torch.equal(channel, self.channel):
            raise ValueError("receptor channel map does not match architecture")
        if not torch.equal(polarity, self.polarity):
            raise ValueError("receptor polarity does not match architecture")

    def transplant_payload(self, payload: Mapping[str, Any]) -> None:
        """Adopt a validated map during an explicit architecture migration."""

        channel, polarity, channel_scale = self._payload_tensors(payload)
        self.channel = channel.detach().clone()
        self.polarity = polarity.detach().clone()
        self.channel_scale = channel_scale.detach().clone()


class ByteMotor:
    """A single action organ trained by local outcome error."""

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.receptors = SparseReceptorBank(
            config.cortical_context_dim,
            config.motor_context_dim,
            generator=generator,
            context_norm=config.motor_context_norm,
            device=self.device,
        )
        self.synapses = SparseSynapses(
            config.alphabet_size,
            config.motor_context_dim,
            config.motor_context_dim,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.bias = torch.zeros(config.alphabet_size, device=self.device)
        self.reward_baseline = 0.0
        self.reward_updates = 0

    def encode_context(self, cortical_state: torch.Tensor) -> torch.Tensor:
        return self.receptors.forward(cortical_state)

    def probabilities(
        self,
        context: torch.Tensor,
        *,
        episodic_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        evidence = self.synapses.forward(context) + self.bias
        if episodic_evidence is not None:
            if episodic_evidence.shape != (self.config.alphabet_size,):
                raise ValueError("episodic evidence dimension mismatch")
            evidence = evidence + episodic_evidence.to(self.device)
        evidence = evidence / float(self.config.motor_temperature)
        return torch.softmax(evidence, dim=0)

    @torch.no_grad()
    def learn(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        observed_symbol: int,
    ) -> torch.Tensor:
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(observed_symbol)] = 1.0
        error = target - predicted.to(self.device)
        self._apply_error(context, error)
        return error

    @torch.no_grad()
    def learn_reward(
        self,
        context: torch.Tensor,
        policy_probabilities: torch.Tensor,
        action_symbol: int,
        reward: float,
    ) -> tuple[torch.Tensor, float]:
        """Apply a local three-factor action × eligibility × reward update."""

        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if policy_probabilities.shape != (self.config.alphabet_size,):
            raise ValueError("policy probability dimension mismatch")
        if not 0 <= int(action_symbol) < self.config.alphabet_size:
            raise ValueError("action is outside the motor alphabet")
        modulation = reward - self.reward_baseline
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(action_symbol)] = 1.0
        error = modulation * (target - policy_probabilities.to(self.device))
        self._apply_error(context, error)
        self.reward_baseline += self.config.reward_baseline_rate * modulation
        self.reward_updates += 1
        return error, modulation

    @torch.no_grad()
    def _apply_error(
        self,
        context: torch.Tensor,
        error: torch.Tensor,
    ) -> None:
        self.synapses.local_update(
            error,
            context,
            learning_rate=self.config.motor_learning_rate,
            weight_decay=self.config.synapse_decay,
        )
        self.bias.add_(self.config.bias_learning_rate * error)
        self.bias.sub_(self.bias.mean())
        self.bias.clamp_(-self.config.max_weight_norm, self.config.max_weight_norm)

    def to_payload(self) -> dict[str, Any]:
        return {
            "receptors": self.receptors.to_payload(),
            "synapses": self.synapses.to_payload(),
            "bias": self.bias.detach().cpu().clone(),
            "reward_baseline": self.reward_baseline,
            "reward_updates": self.reward_updates,
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        self.receptors.load_payload(payload["receptors"])
        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("motor bias shape does not match architecture")
        self.bias = bias
        self.reward_baseline = float(payload["reward_baseline"])
        self.reward_updates = int(payload["reward_updates"])


class BytePredictiveContext:
    """Private F1 context with a sparse, locally plastic temporal residual.

    The organ receives only a read-only cortical state from the shared fabric.
    F1 decoder error can update the residual's existing temporal contacts, but
    no forward or learning path reaches shared fabric, F4 motor, episodic
    memory or identity state.  The residual starts at zero so an explicit
    v9→v10 migration preserves behaviour at its boundary when it adopts the
    former motor receptor map.
    """

    PAYLOAD_FORMAT = "taiji-byte-predictive-context-v1"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.receptors = SparseReceptorBank(
            config.cortical_context_dim,
            config.motor_context_dim,
            generator=generator,
            context_norm=config.motor_context_norm,
            device=self.device,
        )
        # R2 组合绑定实验（所有者 2026-09-23 授权）：分块受力面。默认关闭 ⇒ 上面那张 map 就是全部，
        # 行为与载荷逐位不变。开启时**额外**建两张半宽 map（各接受一半输入、只写自己那一半输出通道），
        # `encode` 改走它们；旧 map 仍然照建照存，这样迁移、消融、既有测试的引用面一个都不动。
        #
        # **刻意用独立 generator**：分块 map 的抽样不消耗主 RNG 流，所以"开"与"关"两臂里
        # **其它器官的拓扑逐位相同**，两臂的差别只剩 encode 走哪条路——否则两臂连随机初始化都不同，
        # 配对就白配了。
        self.receptors_factor: tuple[SparseReceptorBank, SparseReceptorBank] | None = None
        self._factor_split = sum(config.region_sizes)
        if bool(getattr(config, "receptors_factored", False)):
            dedicated = torch.Generator(device="cpu")
            dedicated.manual_seed(RECEPTOR_FACTOR_SEED)
            half = config.motor_context_dim // 2
            self.receptors_factor = (
                SparseReceptorBank(
                    self._factor_split,
                    half,
                    generator=dedicated,
                    context_norm=config.motor_context_norm,
                    device=self.device,
                ),
                SparseReceptorBank(
                    config.cortical_context_dim - self._factor_split,
                    config.motor_context_dim - half,
                    generator=dedicated,
                    context_norm=config.motor_context_norm,
                    device=self.device,
                ),
            )
        # A one-unit context cannot exclude its only input.  Normal profiles
        # forbid self contacts so the residual carries preceding context rather
        # than an instantaneous self gain.
        allow_self = config.motor_context_dim <= 1
        self.recurrent = SparseSynapses(
            config.motor_context_dim,
            config.motor_context_dim,
            config.predictive_context_fan_in,
            generator=generator,
            init_scale=0.0,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
            allow_self=allow_self,
        )

    def encode(
        self,
        cortical_state: torch.Tensor,
        *,
        prior_context: torch.Tensor | None,
        recurrent_override: DevelopmentalSynapseBank | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current context and exact prior trace used to form it."""

        if self.receptors_factor is None:
            base = self.receptors.forward(cortical_state)
        else:
            # 分块：前半（各区 activity 的拼接）只进前一半通道，后半（trace）只进后一半。
            activity_bank, trace_bank = self.receptors_factor
            base = torch.cat(
                [
                    activity_bank.forward(cortical_state[: self._factor_split]),
                    trace_bank.forward(cortical_state[self._factor_split :]),
                ],
                dim=0,
            )
        if prior_context is None:
            trace = torch.zeros(self.config.motor_context_dim, device=self.device)
        else:
            if prior_context.shape != (self.config.motor_context_dim,):
                raise ValueError("predictive context trace dimension mismatch")
            trace = prior_context.detach().to(self.device).clone()
        recurrent = (
            self.recurrent.forward(trace)
            if recurrent_override is None
            else recurrent_override.forward(trace)
        )
        residual = float(self.config.predictive_context_recurrent_gain) * recurrent
        context = bound_norm(base + residual, self.config.motor_context_norm)
        return context, trace

    @torch.no_grad()
    def learn(
        self,
        trace: torch.Tensor,
        feedback: torch.Tensor,
        *,
        learning_rate_scale: float = 1.0,
    ) -> None:
        """Apply F1 decoder feedback to private temporal contacts only."""

        learning_rate_scale = float(learning_rate_scale)
        if not math.isfinite(learning_rate_scale) or learning_rate_scale < 0.0:
            raise ValueError("learning_rate_scale must be finite and non-negative")

        if trace.shape != (self.config.motor_context_dim,):
            raise ValueError("predictive context trace dimension mismatch")
        if feedback.shape != (self.config.motor_context_dim,):
            raise ValueError("predictive context feedback dimension mismatch")
        # The first F1 tick has no predecessor.  Skipping it keeps the causal
        # eligibility trace explicit instead of relying on a zero update.
        if learning_rate_scale == 0.0 or not bool(trace.detach().abs().any()):
            return
        self.recurrent.local_update(
            feedback,
            trace,
            learning_rate=(self.config.predictive_context_learning_rate * learning_rate_scale),
            weight_decay=self.config.synapse_decay,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.PAYLOAD_FORMAT,
            "receptors": self.receptors.to_payload(),
            "recurrent": self.recurrent.to_payload(),
        }
        if self.receptors_factor is not None:
            payload["receptors_factor_activity"] = self.receptors_factor[0].to_payload()
            payload["receptors_factor_trace"] = self.receptors_factor[1].to_payload()
        return payload

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != self.PAYLOAD_FORMAT:
            raise ValueError("unsupported predictive context payload")
        # 分块受力面的开关必须与载荷**一致**：任一侧单边存在就报错，
        # 不许"以为是分块结果加载了未分块的权重"这种静默错配。
        factor_keys = ("receptors_factor_activity", "receptors_factor_trace")
        present = [key for key in factor_keys if key in payload]
        if self.receptors_factor is None and present:
            raise ValueError(
                "checkpoint carries a factored receptor map but the architecture is not factored "
                "(config.receptors_factored is false)"
            )
        if self.receptors_factor is not None and len(present) != len(factor_keys):
            raise ValueError("checkpoint is missing the factored receptor maps")
        if self.receptors_factor is not None:
            for bank, key in zip(self.receptors_factor, factor_keys, strict=True):
                bank.load_payload(payload[key])
        # A v9→v10 migration may have transplanted the old motor map.  That
        # valid private topology is now persistent F1 state, so a later v10
        # restore must load it from its own content-addressed payload rather
        # than regenerate the new-organ seed topology.
        self.receptors.transplant_payload(payload["receptors"])
        self.recurrent.load_payload(payload["recurrent"])

    @torch.no_grad()
    def load_legacy_motor_payload(self, payload: Mapping[str, Any]) -> None:
        """Migrate v8/v9's fixed motor basis into the private F1 owner."""

        receptors = payload.get("receptors")
        if not isinstance(receptors, Mapping):
            raise ValueError("legacy motor payload is missing receptors")
        self.receptors.transplant_payload(receptors)
        self.recurrent.edge_weight.zero_()


class GatedMultiTimescaleTemporalResidual:
    """Optional F1 candidate with fast and slow causal eligibility traces.

    The candidate is deliberately separate from ``BytePredictiveContext`` so
    an old v10 checkpoint can remain byte-for-byte compatible when the
    candidate is not enabled.  Both residual banks start at zero: attaching
    the candidate therefore does not change the current prediction until it
    has learned.  The gate is activity-dependent, while the decay constants
    are explicit candidate metadata rather than hidden architecture values.
    """

    PAYLOAD_FORMAT = "taiji-gated-multiscale-temporal-residual-v1"
    PAYLOAD_VERSION = 1

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        fast_decay: float,
        slow_decay: float,
        gate_temperature: float,
        residual_gain: float,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.fast_decay = float(fast_decay)
        self.slow_decay = float(slow_decay)
        self.gate_temperature = float(gate_temperature)
        self.residual_gain = float(residual_gain)
        self._validate_hyperparameters()
        allow_self = config.motor_context_dim <= 1
        self.fast = SparseSynapses(
            config.motor_context_dim,
            config.motor_context_dim,
            config.predictive_context_fan_in,
            generator=generator,
            init_scale=0.0,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
            allow_self=allow_self,
        )
        self.slow = SparseSynapses(
            config.motor_context_dim,
            config.motor_context_dim,
            config.predictive_context_fan_in,
            generator=generator,
            init_scale=0.0,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
            allow_self=allow_self,
        )

    def _validate_hyperparameters(self) -> None:
        if not 0.0 <= self.fast_decay < 1.0:
            raise ValueError("fast_decay must be in [0, 1)")
        if not 0.0 < self.slow_decay < 1.0:
            raise ValueError("slow_decay must be in (0, 1)")
        if not self.gate_temperature > 0.0:
            raise ValueError("gate_temperature must be positive")
        if not self.residual_gain > 0.0:
            raise ValueError("residual_gain must be positive")

    def _slow_trace(
        self,
        prior_context: torch.Tensor,
        prior_slow_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if prior_context.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate prior context dimension mismatch")
        if prior_slow_context is None:
            prior_slow_context = torch.zeros_like(prior_context)
        elif prior_slow_context.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate slow context dimension mismatch")
        return self.slow_decay * prior_slow_context + (1.0 - self.slow_decay) * prior_context

    def encode(
        self,
        base_context: torch.Tensor,
        *,
        prior_context: torch.Tensor | None,
        prior_slow_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a gated fast/slow residual and return the next slow trace."""

        if base_context.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate base context dimension mismatch")
        if prior_context is None:
            prior_context = torch.zeros_like(base_context)
        else:
            prior_context = prior_context.detach().to(self.device).clone()
        slow_trace = self._slow_trace(prior_context, prior_slow_context)
        fast_norm = prior_context.norm()
        slow_norm = slow_trace.norm()
        slow_gate = torch.sigmoid((slow_norm - fast_norm) / float(self.gate_temperature))
        fast_gate = 1.0 - slow_gate
        residual = float(self.residual_gain) * (
            fast_gate * self.fast.forward(prior_context) + slow_gate * self.slow.forward(slow_trace)
        )
        return (
            bound_norm(
                base_context + residual,
                self.config.motor_context_norm,
            ),
            slow_trace,
        )

    @torch.no_grad()
    def learn(
        self,
        prior_context: torch.Tensor,
        prior_slow_context: torch.Tensor | None,
        feedback: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        """Write both eligibility scales from the same causal F1 feedback."""

        if feedback.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate feedback dimension mismatch")
        prior_context = prior_context.detach().to(self.device).clone()
        slow_trace = self._slow_trace(prior_context, prior_slow_context)
        self.fast.local_update(
            feedback,
            prior_context,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.slow.local_update(
            feedback,
            slow_trace,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.fast.edge_weight, self.slow.edge_weight

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.PAYLOAD_FORMAT,
            "version": self.PAYLOAD_VERSION,
            "fast_decay": self.fast_decay,
            "slow_decay": self.slow_decay,
            "gate_temperature": self.gate_temperature,
            "residual_gain": self.residual_gain,
            "fast": self.fast.to_payload(),
            "slow": self.slow.to_payload(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != self.PAYLOAD_FORMAT:
            raise ValueError("unsupported gated temporal residual payload")
        if int(payload.get("version", -1)) != self.PAYLOAD_VERSION:
            raise ValueError("unsupported gated temporal residual version")
        values = {
            "fast_decay": float(payload["fast_decay"]),
            "slow_decay": float(payload["slow_decay"]),
            "gate_temperature": float(payload["gate_temperature"]),
            "residual_gain": float(payload["residual_gain"]),
        }
        self.fast_decay = values["fast_decay"]
        self.slow_decay = values["slow_decay"]
        self.gate_temperature = values["gate_temperature"]
        self.residual_gain = values["residual_gain"]
        self._validate_hyperparameters()
        self.fast.load_payload(payload["fast"])
        self.slow.load_payload(payload["slow"])


class BytePredictiveReadout:
    """Dedicated F1 next-byte decoder over its private predictive context.

    ``ByteMotor`` is an action policy: its reward update owns F4 decisions and
    must not be repurposed as a language loss.  The predictive readout owns
    separate physical synapses and a bias over the F1 private context, so a
    byte error never writes action policy or episodic value evidence.

    The legacy checkpoint migration intentionally seeds this decoder from the
    former shared motor decoder.  It preserves an already learned F1 surface
    at the migration boundary; all subsequent byte updates are isolated here.
    """

    PAYLOAD_FORMAT = "taiji-byte-predictive-readout-v1"
    #: PLAN-R2-01：可选 UTF-8 位置列在 payload 里的键。关闭时不写 ⇒ 旧档逐位可载。
    POSITION_PAYLOAD_KEY = "position_weight"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.synapses = SparseSynapses(
            config.alphabet_size,
            config.motor_context_dim,
            config.motor_context_dim,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.bias = torch.zeros(config.alphabet_size, device=self.device)
        #: PLAN-R2-01：一张**独立的 4 列**位置输入权重，而不是把 4 维并进
        #: `SparseSynapses.in_features` —— 后者的固定扇入会在建库时按 `in_features`
        #: 重抽边，把已有每一行的连接全部换掉（旧 payload 载不回，`gate` 律也失效）。
        #: **零初始化** ⇒ 开启但未训时与关闭逐位同；关闭时不建（形状/payload/digest 不动）。
        self.position_weight = (
            torch.zeros(
                config.alphabet_size,
                UTF8_POSITION_DIM,
                device=self.device,
            )
            if config.readout_utf8_position_input
            else None
        )
        self._position_probability_steps = 0
        self._position_learn_steps = 0

    @property
    def position_input_enabled(self) -> bool:
        return self.position_weight is not None

    @property
    def position_probability_steps(self) -> int:
        """读到位置输入的预测步数（'被走到'计数，防静默未接线）。"""

        return int(self._position_probability_steps)

    @property
    def position_learn_steps(self) -> int:
        """写到位置列的更新步数（与上一条分开，防只接预测不接学习）。"""

        return int(self._position_learn_steps)

    def _position_one_hot(self, position_state: int) -> torch.Tensor:
        index = int(position_state)
        if not 0 <= index < UTF8_POSITION_DIM:
            raise ValueError("utf-8 position state is outside the 0..3 DFA range")
        vector = torch.zeros(UTF8_POSITION_DIM, device=self.device)
        vector[index] = 1.0
        return vector

    def _require_position_one_hot(self, position_state: int | None) -> torch.Tensor | None:
        """把调用方给的位置类变成可加的一列；开启却没给 ⇒ 响亮失败。

        这条守卫就是本仓反复付学费的那一类：「训练学的不是发射用的」或"以为接上了"。
        默认关闭 ⇒ 全部既有调用点行为不变。
        """

        if self.position_weight is None:
            return None
        if position_state is None:
            raise ValueError(
                "readout_utf8_position_input is enabled but no utf-8 position state "
                "was supplied; training and generation must feed the same input"
            )
        return self._position_one_hot(position_state)

    @torch.no_grad()
    def adopt_position_input(self, source: BytePredictiveReadout) -> None:
        """把另一个读出器的位置列搬过来（派生读出器 fork 时用）。"""

        if self.position_weight is None or source.position_weight is None:
            return
        self.position_weight = source.position_weight.detach().clone()

    def probabilities(
        self,
        context: torch.Tensor,
        *,
        episodic_evidence: torch.Tensor | None = None,
        synapses_override: DevelopmentalSynapseBank | None = None,
        position_state: int | None = None,
    ) -> torch.Tensor:
        synapses = (
            self.synapses.forward(context)
            if synapses_override is None
            else synapses_override.forward(context)
        )
        evidence = synapses + self.bias
        one_hot = self._require_position_one_hot(position_state)
        if one_hot is not None:
            evidence = evidence + self.position_weight @ one_hot
            self._position_probability_steps += 1
        if episodic_evidence is not None:
            if episodic_evidence.shape != (self.config.alphabet_size,):
                raise ValueError("episodic evidence dimension mismatch")
            evidence = evidence + episodic_evidence.to(self.device)
        evidence = evidence / float(self.config.motor_temperature)
        return torch.softmax(evidence, dim=0)

    def prediction_error(
        self,
        predicted: torch.Tensor,
        observed_symbol: int,
    ) -> torch.Tensor:
        if predicted.shape != (self.config.alphabet_size,):
            raise ValueError("predictive probability dimension mismatch")
        if not 0 <= int(observed_symbol) < self.config.alphabet_size:
            raise ValueError("observed symbol is outside the predictive alphabet")
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(observed_symbol)] = 1.0
        return target - predicted.to(self.device)

    def context_feedback(
        self,
        error: torch.Tensor,
        *,
        synapses_override: DevelopmentalSynapseBank | None = None,
    ) -> torch.Tensor:
        """Project causal F1 error into the private context space."""

        if error.shape != (self.config.alphabet_size,):
            raise ValueError("predictive error dimension mismatch")
        # Callers take this before ``learn`` mutates decoder contacts, so the
        # temporal residual learns from the surface that made the prediction.
        synapses = self.synapses if synapses_override is None else synapses_override
        return synapses.backproject(error) / float(self.config.motor_temperature)

    @torch.no_grad()
    def learn(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        observed_symbol: int,
        *,
        preservation_probabilities: torch.Tensor | None = None,
        preservation_strength: float = 0.0,
        learning_rate_scale: float = 1.0,
        position_state: int | None = None,
    ) -> torch.Tensor:
        learning_rate_scale = float(learning_rate_scale)
        if not math.isfinite(learning_rate_scale) or learning_rate_scale < 0.0:
            raise ValueError("learning_rate_scale must be finite and non-negative")
        strength = float(preservation_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("preservation_strength must be finite and non-negative")
        if strength > 0.0 and preservation_probabilities is None:
            raise ValueError("positive preservation_strength requires reference probabilities")
        error = self.prediction_error(predicted, observed_symbol)
        if preservation_probabilities is not None:
            if preservation_probabilities.shape != (self.config.alphabet_size,):
                raise ValueError("preservation probability dimension mismatch")
            reference = preservation_probabilities.to(self.device)
            if not bool(torch.isfinite(reference).all()):
                raise ValueError("preservation probabilities must be finite")
            error = error + strength * (reference - predicted.to(self.device))
        if learning_rate_scale == 0.0:
            return error
        self.synapses.local_update(
            error,
            context,
            # M2-2f preserves the old F1 local-update scale while moving its
            # destination.  A future evidence-backed schedule may add a
            # dedicated rate, but the migration itself must not silently tune
            # the training semantics it is measuring.
            learning_rate=self.config.motor_learning_rate * learning_rate_scale,
            weight_decay=self.config.synapse_decay,
        )
        self.bias.add_(self.config.bias_learning_rate * learning_rate_scale * error)
        self.bias.sub_(self.bias.mean())
        self.bias.clamp_(-self.config.max_weight_norm, self.config.max_weight_norm)
        self._learn_position(error, learning_rate_scale, position_state)
        return error

    @torch.no_grad()
    def _learn_position(
        self,
        error: torch.Tensor,
        learning_rate_scale: float,
        position_state: int | None,
    ) -> None:
        """把同一条局部误差写进那 4 列位置输入（PLAN-R2-01）。

        位置输入是**外生的**（由字节流确定，不由读出预测），所以它只进前向证据、
        不进 `context_feedback` 的语境反投影。规则与 `SparseSynapses.local_update` 同形：
        被点亮的那一列按 `motor_learning_rate` 收紧，未点亮的列按 `synapse_decay` 松弛
        （one-hot ⇒ 非零迹数 = 1 ⇒ 缩放因子 1，与稀疏库一致）；越界按 bias 那套
        逐元素 ±`max_weight_norm` 钳位。**注意调用点顺序**：先取 `one_hot` 再计数，
        保证"开启却没给"能响亮失败而不是被静默写坏。
        """

        one_hot = self._require_position_one_hot(position_state)
        if one_hot is None:
            return
        decay = float(self.config.synapse_decay)
        if decay:
            silent = (one_hot == 0).to(self.position_weight.dtype)
            self.position_weight.mul_(1.0 - decay * silent)
        self.position_weight.add_(
            float(self.config.motor_learning_rate)
            * float(learning_rate_scale)
            * torch.outer(error, one_hot)
        )
        self.position_weight.clamp_(-self.config.max_weight_norm, self.config.max_weight_norm)
        self._position_learn_steps += 1

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "format": self.PAYLOAD_FORMAT,
            "synapses": self.synapses.to_payload(),
            "bias": self.bias.detach().cpu().clone(),
        }
        if self.position_weight is not None:
            payload[self.POSITION_PAYLOAD_KEY] = self.position_weight.detach().cpu().clone()
        return payload

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != self.PAYLOAD_FORMAT:
            raise ValueError("unsupported predictive readout payload")
        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("predictive readout bias shape does not match architecture")
        self.bias = bias
        self._load_position_weight(payload.get(self.POSITION_PAYLOAD_KEY))

    def _load_position_weight(self, stored: Any) -> None:
        """载入可选位置列：关闭而不能载 ⇒ 响亮失败；开启而档里没有 ⇒ 保持零。

        「档里没有」= 由未开此特性的基底载入而来，零初始化正是"开启但未训
        与关闭逐位同"的那条纪律（PLAN-R2-01 §5 守卫②）。
        """

        if stored is None:
            return
        if self.position_weight is None:
            raise ValueError(
                "checkpoint carries UTF-8 position columns but "
                "readout_utf8_position_input is disabled"
            )
        value = stored.detach().to(self.device).clone()
        if value.shape != (self.config.alphabet_size, UTF8_POSITION_DIM):
            raise ValueError("predictive readout position weight shape does not match architecture")
        self.position_weight = value

    def load_legacy_motor_payload(self, payload: Mapping[str, Any]) -> None:
        """Copy the pre-M2-2f shared decoder into this isolated F1 owner."""

        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("legacy motor bias shape does not match architecture")
        self.bias = bias


class ResponsePlanReadout(BytePredictiveReadout):
    """Native byte renderer with legacy and factorized response-plan modes.

    The legacy ``single`` mode is preserved for H3.5/H3.6 checkpoint
    compatibility.  H3.7's ``factorized_v1`` mode keeps the same total plan
    width but consumes ordered slots over the response and routes causal byte
    error into the plan bridge and the active planner rows.
    """

    PAYLOAD_FORMAT = "taiji-response-plan-readout-v1"
    FACTORIZED_PAYLOAD_FORMAT = "taiji-response-plan-readout-factorized-v1"
    VARIANT_SINGLE = "single"
    VARIANT_FACTORIZED = "factorized_v1"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        plan_width: int = 32,
        variant: str = VARIANT_SINGLE,
        plan_slots: int = 4,
        phase_stride: int = 16,
        bridge_learning_rate_scale: float = 1.0,
        slot_credit_scale: float = 0.25,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(config, generator=generator, device=device)
        self.plan_width = int(plan_width)
        self.variant = str(variant)
        self.plan_slots = int(plan_slots)
        self.phase_stride = int(phase_stride)
        self.bridge_learning_rate_scale = float(bridge_learning_rate_scale)
        self.slot_credit_scale = float(slot_credit_scale)
        if self.plan_width <= 0:
            raise ValueError("plan_width must be positive")
        if self.variant not in {self.VARIANT_SINGLE, self.VARIANT_FACTORIZED}:
            raise ValueError("unsupported response plan variant")
        if self.variant == self.VARIANT_SINGLE:
            self.plan_slots = 1
            self.phase_stride = max(1, self.phase_stride)
        else:
            if self.plan_slots <= 1 or self.plan_width % self.plan_slots != 0:
                raise ValueError("factorized response plan width must divide into multiple slots")
            if self.phase_stride <= 0:
                raise ValueError("factorized response plan phase_stride must be positive")
        if (
            not math.isfinite(self.bridge_learning_rate_scale)
            or self.bridge_learning_rate_scale < 0.0
        ):
            raise ValueError("bridge_learning_rate_scale must be finite and non-negative")
        if not math.isfinite(self.slot_credit_scale) or self.slot_credit_scale < 0.0:
            raise ValueError("slot_credit_scale must be finite and non-negative")
        self.plan_slot_width = self.plan_width // self.plan_slots
        scale = float(config.weight_init_scale)
        self.planner_weight = (
            torch.randn(
                self.plan_width,
                config.motor_context_dim,
                generator=generator,
                device=self.device,
            )
            * scale
        )
        self.planner_bias = torch.zeros(self.plan_width, device=self.device)
        self.plan_bridge = (
            torch.randn(
                config.motor_context_dim,
                self.plan_width,
                generator=generator,
                device=self.device,
            )
            * scale
        )
        self._plan_state: torch.Tensor | None = None
        self._plan_source: torch.Tensor | None = None
        self._plan_step = 0
        self._ablation_mode: str | None = None

    @property
    def plan_state(self) -> torch.Tensor | None:
        return None if self._plan_state is None else self._plan_state.detach().clone()

    @property
    def plan_slots_state(self) -> torch.Tensor | None:
        if self._plan_state is None:
            return None
        return self._plan_state.detach().reshape(self.plan_slots, self.plan_slot_width).clone()

    @property
    def plan_phase(self) -> int:
        if self.variant != self.VARIANT_FACTORIZED:
            return 0
        return min(self._plan_step // self.phase_stride, self.plan_slots - 1)

    @property
    def plan_step(self) -> int:
        return int(self._plan_step)

    @torch.no_grad()
    def begin_plan(self, context: torch.Tensor) -> torch.Tensor:
        source = context.detach().to(self.device)
        if source.shape != (self.config.motor_context_dim,):
            raise ValueError("response plan context dimension mismatch")
        self._plan_source = source.clone()
        self._plan_state = torch.tanh(self.planner_weight @ source + self.planner_bias)
        self._plan_step = 0
        return self._plan_state.detach().clone()

    @torch.no_grad()
    def advance_phase(self) -> int:
        if self._plan_state is None:
            raise RuntimeError("response plan has not been created")
        if self.variant == self.VARIANT_FACTORIZED:
            self._plan_step += 1
        return self.plan_phase

    def clear_plan(self) -> None:
        self._plan_state = None
        self._plan_source = None
        self._plan_step = 0

    def set_ablation_mode(self, mode: str | None) -> None:
        """Set a transient read-only diagnostic mode excluded from payloads."""

        if mode not in {None, "slot_credit"}:
            raise ValueError("unsupported response plan ablation mode")
        if mode == "slot_credit" and self.variant != self.VARIANT_FACTORIZED:
            raise ValueError("slot-credit ablation requires the factorized variant")
        self._ablation_mode = mode

    @property
    def ablation_mode(self) -> str | None:
        """Return the transient diagnostic mode without serializing it."""

        return self._ablation_mode

    def _active_plan_vector(self) -> torch.Tensor:
        if self._plan_state is None:
            raise RuntimeError("response plan has not been created")
        if self.variant != self.VARIANT_FACTORIZED:
            return self._plan_state
        slots = self._plan_state.reshape(self.plan_slots, self.plan_slot_width)
        phase = self.plan_phase
        active = slots[phase].clone()
        if phase > 0:
            active = 0.75 * active + 0.25 * slots[0]
            if self._ablation_mode == "slot_credit":
                active = 0.25 * slots[0]
        vector = torch.zeros_like(self._plan_state)
        start = phase * self.plan_slot_width
        vector[start : start + self.plan_slot_width] = active
        return vector

    def _plan_state_feedback(self, feedback: torch.Tensor) -> torch.Tensor:
        """Transpose-Jacobian of the slot mixture, before planner tanh."""
        if self.variant != self.VARIANT_FACTORIZED:
            return feedback
        result = torch.zeros_like(feedback)
        start = self.plan_phase * self.plan_slot_width
        stop = start + self.plan_slot_width
        if self.plan_phase == 0:
            result[: self.plan_slot_width] = feedback[: self.plan_slot_width]
        else:
            if self._ablation_mode != "slot_credit":
                result[start:stop] = 0.75 * feedback[start:stop]
            result[: self.plan_slot_width] = 0.25 * feedback[start:stop]
        return result

    def _conditioned_context(self, context: torch.Tensor) -> torch.Tensor:
        vector = self._active_plan_vector()
        context = context.to(self.device)
        return torch.tanh(context + self.plan_bridge @ vector)

    def probabilities(self, context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return super().probabilities(self._conditioned_context(context), **kwargs)

    def ablated_probabilities(
        self,
        context: torch.Tensor,
        *,
        position_state: int | None = None,
    ) -> torch.Tensor:
        """Read the same candidate renderer with the plan bridge removed."""

        return super().probabilities(
            context.to(self.device),
            position_state=position_state,
        )

    def context_feedback(self, error: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return super().context_feedback(error, **kwargs)

    @torch.no_grad()
    def learn(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        observed_symbol: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        learning_rate_scale = float(kwargs.get("learning_rate_scale", 1.0))
        conditioned = self._conditioned_context(context)
        error = self.prediction_error(predicted, observed_symbol)
        if (
            self.variant == self.VARIANT_FACTORIZED
            and learning_rate_scale > 0.0
            and self.bridge_learning_rate_scale > 0.0
        ):
            if self._plan_source is None or self._plan_state is None:
                raise RuntimeError("factorized response plan has not been created")
            # This is the explicit H3.7 causal bridge: use the same surface
            # that produced the byte prior, then project its error through the
            # local tanh derivative before changing bridge or planner rows.
            feedback = super().context_feedback(error)
            conditioned_delta = feedback * (1.0 - conditioned.square())
            active_vector = self._active_plan_vector()
            bridge_before = self.plan_bridge.detach().clone()
            plan_feedback = bridge_before.T @ conditioned_delta
            bridge_rate = (
                float(self.config.predictive_context_learning_rate)
                * learning_rate_scale
                * self.bridge_learning_rate_scale
            )
            self.plan_bridge.add_(bridge_rate * torch.outer(conditioned_delta, active_vector))
            slot_feedback = self._plan_state_feedback(plan_feedback)
            state = self._plan_state
            slot_delta = slot_feedback * (1.0 - state.square())
            planner_rate = bridge_rate * self.slot_credit_scale
            self.planner_weight.add_(planner_rate * torch.outer(slot_delta, self._plan_source))
            self.planner_bias.add_(
                float(self.config.bias_learning_rate)
                * learning_rate_scale
                * self.bridge_learning_rate_scale
                * self.slot_credit_scale
                * slot_delta
            )
            limit = float(self.config.max_weight_norm)
            self.plan_bridge.clamp_(-limit, limit)
            self.planner_weight.clamp_(-limit, limit)
            self.planner_bias.clamp_(-limit, limit)
            self._plan_state = torch.tanh(
                self.planner_weight @ self._plan_source + self.planner_bias
            )
        return super().learn(conditioned, predicted, observed_symbol, **kwargs)

    @torch.no_grad()
    def learn_plan_target(self, target: torch.Tensor) -> torch.Tensor:
        if self._plan_state is None or self._plan_source is None:
            raise RuntimeError("response plan has not been created")
        target = target.detach().to(self.device)
        if target.shape != (self.plan_width,):
            raise ValueError("response plan target dimension mismatch")
        error = target - self._plan_state
        derivative = 1.0 - self._plan_state.square()
        delta = error * derivative
        rate = float(self.config.motor_learning_rate)
        self.planner_weight.add_(rate * torch.outer(delta, self._plan_source))
        self.planner_bias.add_(float(self.config.bias_learning_rate) * delta)
        limit = float(self.config.max_weight_norm)
        self.planner_weight.clamp_(-limit, limit)
        self.planner_bias.clamp_(-limit, limit)
        self._plan_state = torch.tanh(self.planner_weight @ self._plan_source + self.planner_bias)
        return error.detach().clone()

    @property
    def active_parameter_count(self) -> int:
        count = (
            self.synapses.edge_count
            + self.bias.numel()
            + self.planner_weight.numel()
            + self.planner_bias.numel()
            + self.plan_bridge.numel()
        )
        if self.position_weight is not None:
            count += self.position_weight.numel()
        return int(count)

    def to_payload(self) -> dict[str, Any]:
        common = {
            "plan_width": self.plan_width,
            "renderer": super().to_payload(),
            "planner_weight": self.planner_weight.detach().cpu().clone(),
            "planner_bias": self.planner_bias.detach().cpu().clone(),
            "plan_bridge": self.plan_bridge.detach().cpu().clone(),
            "plan_state": (
                None if self._plan_state is None else self._plan_state.detach().cpu().clone()
            ),
            "plan_source": (
                None if self._plan_source is None else self._plan_source.detach().cpu().clone()
            ),
        }
        if self.variant == self.VARIANT_SINGLE:
            return {"format": self.PAYLOAD_FORMAT, **common}
        return {
            "format": self.FACTORIZED_PAYLOAD_FORMAT,
            "variant": self.variant,
            "plan_slots": self.plan_slots,
            "plan_slot_width": self.plan_slot_width,
            "phase_stride": self.phase_stride,
            "bridge_learning_rate_scale": self.bridge_learning_rate_scale,
            "slot_credit_scale": self.slot_credit_scale,
            "plan_step": self.plan_step,
            **common,
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        expected_format = (
            self.PAYLOAD_FORMAT
            if self.variant == self.VARIANT_SINGLE
            else self.FACTORIZED_PAYLOAD_FORMAT
        )
        if payload.get("format") != expected_format:
            raise ValueError("unsupported response plan readout payload")
        if int(payload.get("plan_width", -1)) != self.plan_width:
            raise ValueError("response plan width does not match architecture")
        if self.variant == self.VARIANT_FACTORIZED:
            if payload.get("variant") != self.variant:
                raise ValueError("response plan variant does not match architecture")
            if int(payload.get("plan_slots", -1)) != self.plan_slots:
                raise ValueError("response plan slot count does not match architecture")
            if int(payload.get("plan_slot_width", -1)) != self.plan_slot_width:
                raise ValueError("response plan slot width does not match architecture")
            if int(payload.get("phase_stride", -1)) != self.phase_stride:
                raise ValueError("response plan phase stride does not match architecture")
            for name, expected in (
                ("bridge_learning_rate_scale", self.bridge_learning_rate_scale),
                ("slot_credit_scale", self.slot_credit_scale),
            ):
                actual = float(payload.get(name, float("nan")))
                if not math.isfinite(actual) or abs(actual - expected) > 1e-12:
                    raise ValueError(f"response plan {name} does not match architecture")
        super().load_payload(payload["renderer"])
        for name, shape in (
            ("planner_weight", (self.plan_width, self.config.motor_context_dim)),
            ("planner_bias", (self.plan_width,)),
            ("plan_bridge", (self.config.motor_context_dim, self.plan_width)),
        ):
            value = payload[name].detach().to(self.device).clone()
            if value.shape != shape:
                raise ValueError(f"{name} shape does not match architecture")
            setattr(self, name, value)
        self._plan_state = (
            None
            if payload.get("plan_state") is None
            else payload["plan_state"].detach().to(self.device).clone()
        )
        self._plan_source = (
            None
            if payload.get("plan_source") is None
            else payload["plan_source"].detach().to(self.device).clone()
        )
        if self._plan_state is not None and self._plan_state.shape != (self.plan_width,):
            raise ValueError("response plan state shape does not match architecture")
        if self._plan_source is not None and self._plan_source.shape != (
            self.config.motor_context_dim,
        ):
            raise ValueError("response plan source shape does not match architecture")
        self._plan_step = int(payload.get("plan_step", 0))
        if self._plan_step < 0:
            raise ValueError("response plan phase step must be non-negative")
        self._ablation_mode = None
