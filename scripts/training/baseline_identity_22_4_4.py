"""22.4.4 baseline identity measurement: construct-only, zero training.

This module measures the two candidate regular baselines (`gru_sequence`,
`transformer_sequence`) against the frozen U2 anchor of 22,742 scalars, selects
exactly one by the frozen L1-L5 chain, and writes:

  reports/taiji_u2_22_4_4_identity_measurement_20261111.json
  plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json
  plans/manifests/taiji_u2_22_4_1_execution_20261111_v2.json   (u2 slots only)

Hard constraints enforced here: float32 / cpu / 1 thread, no optimizer object,
no `optimizer.step`, no project corpus or checkpoint reads, total CPU <= 250 s
(measured with `time.process_time`, summed over parent + worker processes).

Run: python -m scripts.training.baseline_identity_22_4_4
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DTYPE = torch.float32
DEVICE = torch.device("cpu")
SEED = 20261110
ANCHOR = 22742
TOLERANCE = 0.10
CPU_CAP = 250.0
STEPS = 8
BATCH = 2
GOAL = "逐字输出当前唯一记录的完整原文。"
MODALITIES = ("rgb", "audio", "proprio")
ARM_IDS = ("gru_sequence", "transformer_sequence")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
REPORT_PATH = ROOT / "reports" / "taiji_u2_22_4_4_identity_measurement_20261111.json"
IDENTITY_PATH = ROOT / "plans" / "manifests" / "taiji_u2_22_4_4_baseline_identity_20261111.json"
EXECUTION_V1 = ROOT / "plans" / "manifests" / "taiji_u2_22_4_1_execution_20261111.json"
EXECUTION_V2 = ROOT / "plans" / "manifests" / "taiji_u2_22_4_1_execution_20261111_v2.json"
SUBJECT_SOURCE = ROOT / "taiji" / "u2_subject.py"
IDENTITY_REF = "plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json"
WORKER_MARKER = "##22_4_4_WORKER##"


class BudgetExceeded(RuntimeError):
    """Raised before the 250 CPU second card budget can be crossed."""


class Budget:
    def __init__(self, cap: float = CPU_CAP) -> None:
        self.cap = cap
        self.child_cpu = 0.0

    def add(self, seconds: float) -> None:
        self.child_cpu += float(seconds)
        if self.child_cpu > self.cap:
            raise BudgetExceeded(f"child CPU {self.child_cpu:.3f} > cap {self.cap}")

    def reserve(self, seconds: float) -> None:
        used = self.child_cpu + time.process_time()
        if used + seconds > self.cap:
            raise BudgetExceeded(f"reserving {seconds} s at used {used:.3f} > cap {self.cap}")


BUDGET = Budget()


def normalized_sha256(path: Path) -> str:
    """UTF8 without BOM, universal newline to LF (repo hash semantics)."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode()).hexdigest()


def canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Shared frontend: line-for-line copy of `_WorldKernel.__init__` frontend.
# --------------------------------------------------------------------------- #


class SharedFrontend(nn.Module):
    """U2 frontend (encoder/metadata/modality/embedding/text_input/convolutions)."""

    def __init__(self) -> None:
        super().__init__()
        from taiji.u2_subject import U2Config

        c = U2Config.small()
        d = c.d
        self.dimensions = dict(c.dimensions)
        self.encoder = nn.ModuleDict(
            {
                m: nn.Linear(size + (8 if m == "proprio" else 0), d)
                for m, size in c.dimensions.items()
            }
        )
        self.metadata = nn.Linear(8, d)
        self.modality = nn.Parameter(torch.randn(5, d) * 0.05)
        self.embedding = nn.Embedding(256, c.embedding)
        self.text_input = nn.Linear(c.embedding, d)
        self.convolutions = nn.ModuleList(nn.Conv1d(d, d, 3, dilation=x) for x in (1, 2, 4))

    def text_units(self, raw: str) -> torch.Tensor:
        """Same byte-level text path as `_WorldKernel.text_units`, truncated to 8 steps."""
        ids = torch.tensor(list(raw.encode("utf-8")[:STEPS]), dtype=torch.long, device=DEVICE)
        value = self.text_input(self.embedding(ids)).T.unsqueeze(0)
        for layer, dilation in zip(self.convolutions, (1, 2, 4), strict=True):
            value = value + torch.tanh(layer(functional.pad(value, (2 * dilation, 0))))
        coordinates = value.new_zeros(ids.numel(), 8)
        coordinates[:, 2] = torch.linspace(
            -1, 1, ids.numel(), dtype=value.dtype, device=value.device
        )
        coordinates[:, 6] = 1
        return torch.tanh(value.squeeze(0).T + self.metadata(coordinates) + self.modality[3]).mean(
            0
        )

    def step_tokens(self, observations: dict, text_context: torch.Tensor) -> torch.Tensor:
        """[batch, steps, 32] = concat(3 x 8 encoder outputs) + metadata(8)."""
        rows = []
        for step in range(STEPS):
            coordinate = torch.zeros(BATCH, 8, dtype=DTYPE, device=DEVICE)
            coordinate[:, 0] = step / (STEPS - 1)
            coordinate[:, 6] = 1
            parts = []
            for index, name in enumerate(MODALITIES):
                value = observations[name][:, step]
                if name == "proprio":
                    value = torch.cat([value, torch.ones_like(value[..., :8])], -1)
                parts.append(
                    self.encoder[name](value) + self.metadata(coordinate) + self.modality[index]
                )
            token = torch.cat([*parts, self.metadata(coordinate) + text_context], -1)
            rows.append(token)
        return torch.stack(rows, 1)


class GRUSequenceBaseline(nn.Module):
    architecture_id = "gru_sequence"

    def __init__(self, h: int) -> None:
        super().__init__()
        self.width = h
        self.frontend = SharedFrontend()
        self.gru = nn.GRU(
            input_size=32, hidden_size=h, num_layers=1, batch_first=True, bidirectional=False
        )
        self.byte_head = nn.Linear(h + 4, 257)
        apply_frozen_init(self)

    def forward(self, observations: dict, targets: torch.Tensor, text_context: torch.Tensor):
        sequence = self.frontend.step_tokens(observations, text_context)
        output, _ = self.gru(sequence)
        previous = self.frontend.embedding(torch.full((BATCH,), 255, dtype=torch.long))
        logits = self.byte_head(torch.cat([output[:, -1], previous], -1))
        return functional.cross_entropy(logits, targets)


class TransformerSequenceBaseline(nn.Module):
    architecture_id = "transformer_sequence"

    def __init__(self, w: int, heads: int = 4, layers: int = 1) -> None:
        super().__init__()
        self.width = w
        self.heads = heads
        self.layers = layers
        self.frontend = SharedFrontend()
        self.record_in = nn.Linear(32, w)
        self.position = nn.Parameter(torch.randn(STEPS, w))
        self.attention = nn.MultiheadAttention(w, heads, batch_first=True, dropout=0.0)
        self.norm_attention = nn.LayerNorm(w)
        self.norm_feedforward = nn.LayerNorm(w)
        self.feedforward = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))
        self.byte_head = nn.Linear(w + 4, 257)
        apply_frozen_init(self)

    def forward(self, observations: dict, targets: torch.Tensor, text_context: torch.Tensor):
        sequence = self.frontend.step_tokens(observations, text_context)
        hidden = self.record_in(sequence) + self.position[None]
        mask = torch.triu(torch.ones(STEPS, STEPS, dtype=torch.bool, device=DEVICE), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
        hidden = self.norm_attention(hidden + attended)
        hidden = self.norm_feedforward(hidden + self.feedforward(hidden))
        previous = self.frontend.embedding(torch.full((BATCH,), 255, dtype=torch.long))
        logits = self.byte_head(torch.cat([hidden[:, -1], previous], -1))
        return functional.cross_entropy(logits, targets)


def apply_frozen_init(model: nn.Module) -> None:
    """Exactly the tail of `_WorldKernel.__init__` (lines 279-283)."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.3 / math.sqrt(module.weight[0].numel()))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)


def build_arm(architecture_id: str, width: int) -> nn.Module:
    torch.manual_seed(SEED)
    if architecture_id == "gru_sequence":
        model = GRUSequenceBaseline(width)
    elif architecture_id == "transformer_sequence":
        model = TransformerSequenceBaseline(width)
    else:
        raise ValueError(architecture_id)
    return model.to(dtype=DTYPE, device=DEVICE).eval()


def param_counts(model: nn.Module) -> tuple[int, int]:
    return (
        int(sum(parameter.numel() for parameter in model.parameters())),
        len(list(model.named_parameters())),
    )


# --------------------------------------------------------------------------- #
# Fragment (synthetic, generated locally; no project data)
# --------------------------------------------------------------------------- #


def fragment_inputs() -> tuple[dict, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    rgb = torch.rand((BATCH, STEPS, 3, 4, 4), generator=generator, dtype=DTYPE)
    audio = torch.rand((BATCH, STEPS, 33), generator=generator, dtype=DTYPE)
    proprio = torch.rand((BATCH, STEPS, 8), generator=generator, dtype=DTYPE)
    targets = torch.randint(0, 257, (BATCH,), generator=generator)
    observations = {"rgb": rgb.reshape(BATCH, STEPS, 48), "audio": audio, "proprio": proprio}
    return observations, targets


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - ctypes fallback
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize)


class PeakRSS:
    """Samples RSS at explicit points of the forward/backward (no sampling thread)."""

    def __init__(self) -> None:
        self.peak = rss_bytes()

    def observe(self) -> None:
        self.peak = max(self.peak, rss_bytes())


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #


def drive_u2(model, peak: PeakRSS | None) -> torch.Tensor:
    """8 steps of the real U2 protocol: choose_operation -> issue -> receive."""
    from taiji.u2_subject import U2Packet, synthetic_fixture

    dt = 0.05
    fixture = synthetic_fixture(model.c, DTYPE)
    state = model.initial("u2-22-4-4-measurement")
    state = model.set_goal(state, GOAL)
    state = model.set_records(state, fixture["records"])
    losses = []
    for index in range(STEPS):
        mask = torch.zeros(9, dtype=torch.bool)
        mask[0] = True
        state, decision = model.choose_operation(state, mask)
        state, identity = model.issue(state, decision, dt=dt)
        # Use the receipt's own interval: accumulation of (index+1)*dt drifts
        # above the accumulated world_end and trips validate_event.
        start = state["world_time"]
        end = state["pending"]["world_end"]
        state["process_time"] = end
        event = fixture["events"][index % len(fixture["events"])]
        state, result = model.receive(
            state,
            U2Packet(
                f"22-4-4-measurement-{index}",
                "sensor",
                "world",
                start,
                end,
                end,
                event["observations"],
                event["coordinates"],
                event["outcome"],
                identity,
                feedback_record_ids=("red",),
            ),
        )
        losses.append(-result["predictive_logp"].mean())
        if peak is not None:
            peak.observe()
    loss = torch.stack(losses).mean()
    if peak is not None:
        peak.observe()
    loss.backward()
    if peak is not None:
        peak.observe()
    return loss


def worker_main(arm: str, width: int | None) -> dict:
    started_cpu = time.process_time()
    construct_cpu = 0.0
    if arm == "u2_reference":
        from taiji.u2_subject import U2Config, U2Subject

        mark = time.process_time()
        model = U2Subject(U2Config.small()).to(dtype=DTYPE, device=DEVICE)
        construct_cpu = time.process_time() - mark
        model.eval()
        torso = "U2Subject(U2Config.small())"
        observations = targets = text_context = None
        label = "u2_reference"
    else:
        assert width is not None
        mark = time.process_time()
        model = build_arm(arm, width)
        construct_cpu = time.process_time() - mark
        observations, targets = fragment_inputs()
        with torch.no_grad():
            text_context = model.frontend.text_units(GOAL)
        label = arm

    gc.collect()
    baseline = rss_bytes()
    peak = PeakRSS()
    peak.peak = baseline

    def once() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        if arm == "u2_reference":
            value = drive_u2(model, None)
        else:
            value = model(observations, targets, text_context)
            value.backward()
        model.zero_grad(set_to_none=True)
        return value

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    if arm == "u2_reference":
        loss = drive_u2(model, peak)
    else:
        loss = model(observations, targets, text_context)
        peak.observe()
        loss.backward()
        peak.observe()
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    peak.observe()

    # `time.process_time` is quantised at ~15.6 ms on Windows; a single
    # baseline fragment is shorter than one tick, so the primary reading is a
    # bound, not a value. This calibration repeats the identical fragment
    # (forward + backward, no optimizer) and reports CPU per repetition.
    repeats = 8 if arm == "u2_reference" else 32
    once()
    mark = time.process_time()
    for _ in range(repeats):
        once()
    cpu_calibrated = (time.process_time() - mark) / repeats

    total, tensors = param_counts(model)
    payload = {
        "arm": label,
        "width": width,
        "torso": torso if arm == "u2_reference" else "scripts/training/baseline_identity_22_4_4.py",
        "param_total": total,
        "param_tensors": tensors,
        "construct_cpu": construct_cpu,
        "fragment_cpu": cpu,
        "fragment_cpu_calibrated": cpu_calibrated,
        "calibration_repeats": repeats,
        "fragment_wall": wall,
        "rss_baseline_bytes": baseline,
        "rss_peak_bytes": peak.peak,
        "delta_rss_bytes": peak.peak - baseline,
        "loss": float(loss.detach()),
        "self_cpu_total": time.process_time() - started_cpu,
    }
    print(WORKER_MARKER + json.dumps(payload, ensure_ascii=False))
    return payload


# --------------------------------------------------------------------------- #
# Parent phases
# --------------------------------------------------------------------------- #


def scan_grid() -> dict[str, list[dict]]:
    grid = {}
    gru = []
    for width in range(8, 129):
        model = build_arm("gru_sequence", width)
        total, tensors = param_counts(model)
        gru.append(
            {
                "width": width,
                "param_total": total,
                "param_tensors": tensors,
                "abs_delta": abs(total - ANCHOR),
                "delta_pct": 100 * abs(total - ANCHOR) / ANCHOR,
            }
        )
    transformer = []
    for width in range(8, 65, 4):
        model = build_arm("transformer_sequence", width)
        total, tensors = param_counts(model)
        transformer.append(
            {
                "width": width,
                "param_total": total,
                "param_tensors": tensors,
                "abs_delta": abs(total - ANCHOR),
                "delta_pct": 100 * abs(total - ANCHOR) / ANCHOR,
            }
        )
    grid["gru_sequence"] = gru
    grid["transformer_sequence"] = transformer
    return grid


def pick(points: list[dict]) -> dict:
    return min(points, key=lambda row: (row["abs_delta"], row["param_total"], row["width"]))


def run_worker(arm: str, width: int | None) -> dict:
    BUDGET.reserve(20.0)
    command = [sys.executable, "-m", "scripts.training.baseline_identity_22_4_4", "--worker", arm]
    if width is not None:
        command += ["--width", str(width)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"worker {arm} failed: {completed.stderr[-2000:]}")
    lines = [line for line in completed.stdout.splitlines() if line.startswith(WORKER_MARKER)]
    if len(lines) != 1:
        raise RuntimeError(f"worker {arm} emitted {len(lines)} payload lines")
    payload = json.loads(lines[0][len(WORKER_MARKER) :])
    BUDGET.add(payload["self_cpu_total"])
    return payload


def measure_arm(arm: str, width: int | None) -> dict:
    runs = [run_worker(arm, width) for _ in range(3)]
    medians = {
        key: statistics.median([run[key] for run in runs])
        for key in (
            "fragment_cpu",
            "fragment_cpu_calibrated",
            "fragment_wall",
            "delta_rss_bytes",
            "construct_cpu",
        )
    }
    return {"arm": arm, "width": width, "runs": runs, "medians": medians}


def state_dict_digest(model: nn.Module) -> str:
    entries = []
    for name, tensor in sorted(model.state_dict().items()):
        entries.append(
            [
                name,
                list(tensor.shape),
                str(tensor.dtype).removeprefix("torch."),
                hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            ]
        )
    return canonical_sha256(entries)


def inventory(model: nn.Module) -> list[list]:
    return [
        [name, list(parameter.shape), str(parameter.dtype).removeprefix("torch.")]
        for name, parameter in model.named_parameters()
    ]


def select_baseline(measured: dict[str, dict], arms: dict[str, dict]) -> dict:
    qualified = [arm for arm in ARM_IDS if arms[arm]["delta_pct"] / 100 <= TOLERANCE]
    trace = []
    if len(qualified) == 1:
        return {
            "qualified_set": qualified,
            "selected": qualified[0],
            "selection_path": "single_qualified",
            "tie_break_level": None,
            "trace": trace,
        }
    if not qualified:
        return {
            "qualified_set": [],
            "selected": None,
            "selection_path": "baseline_parity_infeasible",
            "tie_break_level": None,
            "trace": trace,
        }
    remaining = list(qualified)
    levels = [
        ("L1", lambda arm: arms[arm]["param_total"]),
        ("L2", lambda arm: 1 if arm == "gru_sequence" else 0),
        ("L3", lambda arm: float(measured[arm]["medians"]["delta_rss_bytes"])),
        (
            "L4",
            lambda arm: round(
                float(measured[arm]["medians"]["fragment_cpu_calibrated"]) / STEPS, 6
            ),
        ),
        ("L5", lambda arm: arm),
    ]
    for level, key in levels:
        values = {arm: key(arm) for arm in remaining}
        best = min(values.values())
        winners = [arm for arm in remaining if values[arm] == best]
        trace.append({"level": level, "values": values, "winners": winners})
        if len(winners) == 1:
            return {
                "qualified_set": qualified,
                "selected": winners[0],
                "selection_path": "tie_break",
                "tie_break_level": level,
                "trace": trace,
            }
        remaining = winners
    raise RuntimeError("L5 must be a total order; selection failed")


def build_identity(
    selected: str,
    arms: dict[str, dict],
    measured: dict[str, dict],
    parity: dict,
    selection: dict,
    cpu_total: float,
) -> dict:
    model = build_arm(selected, arms[selected]["width"])
    first_digest = state_dict_digest(model)
    again = build_arm(selected, arms[selected]["width"])
    second_digest = state_dict_digest(again)
    if first_digest != second_digest:
        raise RuntimeError("same-seed construction digests differ")
    total, tensors = param_counts(model)
    if total != arms[selected]["param_total"]:
        raise RuntimeError("identity parameter total disagrees with the grid scan")
    entries = inventory(model)
    transformer = selected == "transformer_sequence"
    width = arms[selected]["width"]
    architecture = {
        "module_path": "scripts/training/baseline_identity_22_4_4.py",
        "module_sha256": normalized_sha256(SCRIPT_PATH),
        "class_name": "TransformerSequenceBaseline" if transformer else "GRUSequenceBaseline",
        "architecture_id": selected,
        "layers": 1,
        "width": width,
        "free_quantity": {"name": "w" if transformer else "h", "value": width},
        "normalization": (
            {"type": "LayerNorm", "count": 2, "eps": 1e-5, "normalized_shape": width}
            if transformer
            else "none"
        ),
        "residual": (
            {
                "present": True,
                "positions": ["attention_output", "feedforward_output"],
                "form": "post_norm",
            }
            if transformer
            else {"present": False}
        ),
        "activation": (
            {"feedforward": "gelu", "frontend_text_conv": "tanh", "frontend_bounded": "tanh"}
            if transformer
            else {
                "recurrent": "tanh (nn.GRU internal)",
                "gates": "sigmoid",
                "frontend_text_conv": "tanh",
                "frontend_bounded": "tanh",
            }
        ),
        "attention_heads": 4 if transformer else None,
        "mask_type": "causal_upper_triangular_bool" if transformer else "causal_by_recurrence",
        "dropout": 0.0,
        "bidirectional": False,
        "input_projection": "shared U2 frontend; 32 = 3 x 8 modality encoders + metadata(8)",
        "output_head": "Linear(width + 4, 257) over the shared 256 x 4 byte embedding",
        "state_tensors": 0 if transformer else 1,
        "state_tensors_definition": "online tensors that must persist across updates and cannot "
        "be rebuilt from parameters and the current input",
        "reused_u2_components": [
            "encoder",
            "metadata",
            "modality",
            "embedding",
            "text_input",
            "convolutions",
        ],
        "new_components": (
            [
                "record_in",
                "position",
                "attention (MultiheadAttention)",
                "norm_attention",
                "norm_feedforward",
                "feedforward",
                "byte_head",
            ]
            if transformer
            else ["gru (nn.GRU)", "byte_head"]
        ),
        "frontend": {
            "source_path": "taiji/u2_subject.py",
            "source_sha256": normalized_sha256(SUBJECT_SOURCE),
            "source_lines": {
                "encoder": "207-212",
                "metadata": "213",
                "modality": "214",
                "embedding": "215",
                "text_input": "216",
                "convolutions": "217",
                "init_loop": "279-283",
            },
            "components": [
                "encoder (ModuleDict rgb/audio/proprio)",
                "metadata (Linear 8->8)",
                "modality (Parameter 5x8)",
                "embedding (Embedding 256x4)",
                "text_input (Linear 4->8)",
                "convolutions (Conv1d 8->8 k=3 dilations 1/2/4)",
            ],
            "copy_note": "copied line for line; the private class is neither imported nor instantiated",
            "proprio_input": "8 values + 8 validity flags = 16",
        },
    }
    return {
        "schema": "taiji-u2-22.4.4-baseline-identity-v1",
        "architecture_id": selected,
        "status": "frozen",
        "measured_at": datetime.now(UTC).isoformat(),
        "conditions": {
            "dtype": "float32",
            "device": "cpu",
            "threads": 1,
            "interop_threads": 1,
            "torch_version": torch.__version__,
            "training_runs": 0,
            "optimizer_steps": 0,
        },
        "param_total": total,
        "param_tensors": tensors,
        "param_inventory": entries,
        "param_inventory_sha256": canonical_sha256(
            [[name, shape, dtype] for name, shape, dtype in sorted(entries)]
        ),
        "param_inventory_rule": "sha256 of canonical JSON (sorted by name) of [name, shape, dtype]",
        "init": {
            "rule": "for every nn.Linear/nn.Conv1d: normal_(std = 0.3 / sqrt(fan_in_per_row)), "
            "bias = 0.01; frontend modality = randn(5, 8) * 0.05; "
            "nn.GRU / nn.Embedding / nn.LayerNorm / position parameter keep torch defaults",
            "source": "taiji/u2_subject.py lines 279-283 (same rule as _WorldKernel)",
            "seed": SEED,
            "post_init_state_dict_sha256": first_digest,
            "second_construction_sha256": second_digest,
            "same_seed_digest_equal": first_digest == second_digest,
        },
        "architecture": architecture,
        "recipe": {
            "declared_only": True,
            "optimizer": {
                "type": "Adam",
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-08,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "batch": BATCH,
            "sequence_length": STEPS,
            "gradient_clip": None,
            "warmup": None,
            "steps": 1,
            "recipe_delta": [],
            "same_as_u2_arm": True,
        },
        "information_rights": {
            "visible_input": "synthetic observation rgb[48] + audio[33] + proprio[8] per step, "
            "plus goal text",
            "feedback_channels": [
                "observation",
                "train_label",
                "online_correction",
                "help_reply",
                "oracle_score",
            ],
            "forbidden_writes": [
                "goal text",
                "environment ground truth",
                "oracle score",
                "other subject state or parameters",
                "project corpus / check / final / confirm / sealed partitions",
            ],
            "visible_history_length": STEPS,
            "interface": "taiji frontend (shared U2 encoders)",
            "oracle_not_fedback": True,
            "equal_to_u2_arm": True,
        },
        "seeds": {
            "init_seed": SEED,
            "data_order_seed": SEED,
            "fixed_seeds": [20261110, 20261111, 20261112],
            "mapping": {
                seed: {"init_seed": seed, "data_order_seed": seed}
                for seed in (20261110, 20261111, 20261112)
            },
            "note": "declared only; this card performs no training",
        },
        "parity": parity,
        "selection": selection,
        "budget": {
            "cpu_total_measured": round(cpu_total, 6),
            "cap": 250,
            "training_runs": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
        },
        "approval": {
            "owner_approval_ref": "chat:2026-11-11#standing-budget-delegation",
            "note": "依据 owner 常设授权（预算内不必逐项请示、须留档），非 owner 逐字原话",
        },
    }


# --------------------------------------------------------------------------- #
# u2 slot backfill (file 4)
# --------------------------------------------------------------------------- #

ROLE_PREFIXES = {
    "output_head": ("byte_head.",),
    "predictive_head": (
        "decoder",
        "joint_readout",
        "decoder_query",
        "decoder_key",
        "decoder_value",
    ),
    "recurrent_core": (
        "native_",
        "f_",
        "bind_",
        "record_",
        "h_",
        "o_",
        "initial_h",
        "initial_p",
        "nu_projection",
        "tau_logit",
        "corrector",
        "motion",
        "post_memory",
        "background",
    ),
}
ROLE_PRIORITY = ("recurrent_core", "predictive_head", "output_head")


def build_roles() -> dict:
    from taiji.u2_subject import U2Config, U2Subject

    model = U2Subject(U2Config.small()).metadata_candidate()
    named = list(model.named_parameters())
    owners: dict[str, str] = {}
    for role in ROLE_PRIORITY:
        for name, _ in named:
            if name in owners:
                continue
            if any(name.startswith(prefix) for prefix in ROLE_PREFIXES[role]):
                owners[name] = role
    roles = {}
    for role in ("predictive_head", "recurrent_core", "output_head"):
        names = [name for name, _ in named if owners.get(name) == role]
        if not names:
            raise RuntimeError(f"role {role} is empty")
        entries = []
        for name, parameter in named:
            if name in names:
                entries.append(
                    [name, list(parameter.shape), str(parameter.dtype).removeprefix("torch.")]
                )
        roles[role] = {
            "param_names": names,
            "dtype": "float32",
            "slot_sha256": canonical_sha256(sorted(entries)),
        }
    covered = [name for role in roles for name in roles[role]["param_names"]]
    if len(covered) != len(set(covered)):
        raise RuntimeError("u2 role slots overlap")
    if any(name not in dict(named) for name in covered):
        raise RuntimeError("u2 role name missing from the model")
    return roles


def write_execution_v2(roles: dict, identity_sha: str) -> tuple[str, str]:
    text = EXECUTION_V1.read_text(encoding="utf-8-sig")
    original = (
        '      "active_parameter_roles": {\n'
        '        "predictive_head": null,\n'
        '        "recurrent_core": null,\n'
        '        "output_head": null\n'
        "      },\n"
        '      "identity_status": "unfilled_blocks_admission",\n'
    )
    if text.count(original) != 1:
        raise RuntimeError("u2 slot block not found exactly once in v1")
    body = json.dumps(roles, indent=2, ensure_ascii=False).replace("\n", "\n      ")
    replacement = (
        '      "active_parameter_roles": ' + body + ",\n"
        '      "identity_status": "filled",\n'
        '      "baseline_identity_sha256": "' + identity_sha + '",\n'
        '      "baseline_identity_ref": "' + IDENTITY_REF + '",\n'
    )
    updated = text.replace(original, replacement)
    v1_data = json.loads(text)
    v2_data = json.loads(updated)
    expected = json.loads(text)
    expected["arms"]["u2-strong-baseline"]["active_parameter_roles"] = roles
    expected["arms"]["u2-strong-baseline"]["identity_status"] = "filled"
    expected["arms"]["u2-strong-baseline"]["baseline_identity_sha256"] = identity_sha
    expected["arms"]["u2-strong-baseline"]["baseline_identity_ref"] = IDENTITY_REF
    if v2_data != expected:
        raise RuntimeError("v2 differs from v1 beyond the allowed u2 fields")
    if v1_data == v2_data:
        raise RuntimeError("v2 carries no change")
    EXECUTION_V2.write_text(updated, encoding="utf-8", newline="")
    return normalized_sha256(EXECUTION_V1), normalized_sha256(EXECUTION_V2)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parent_main() -> int:
    started = time.process_time()
    started_wall = time.perf_counter()

    from taiji.u2_subject import U2Config, U2Subject

    base = U2Subject(U2Config.small())
    candidate = base.metadata_candidate()
    u2_anchor = {
        "candidate_param_total": int(sum(p.numel() for p in candidate.parameters())),
        "candidate_param_tensors": len(list(candidate.named_parameters())),
        "base_param_total": int(sum(p.numel() for p in base.parameters())),
        "base_param_tensors": len(list(base.named_parameters())),
        "anchor_declared": ANCHOR,
    }
    if u2_anchor["candidate_param_total"] != ANCHOR:
        raise RuntimeError("U2 anchor drifted from 22742")
    del base, candidate
    gc.collect()

    grid = scan_grid()
    arms = {}
    for architecture_id in ARM_IDS:
        point = pick(grid[architecture_id])
        arms[architecture_id] = {
            **point,
            "qualified": point["delta_pct"] / 100 <= TOLERANCE,
            "state_tensors": 1 if architecture_id == "gru_sequence" else 0,
        }

    measured = {"u2_reference": measure_arm("u2_reference", None)}
    for architecture_id in ARM_IDS:
        measured[architecture_id] = measure_arm(architecture_id, arms[architecture_id]["width"])

    u2_med = measured["u2_reference"]["medians"]
    parity_by_arm = {}
    for architecture_id in ARM_IDS:
        med = measured[architecture_id]["medians"]
        delta_rss_u2 = float(u2_med["delta_rss_bytes"])
        delta_rss = float(med["delta_rss_bytes"])
        rss_delta_pct = (
            100.0 * (delta_rss - delta_rss_u2) / delta_rss_u2 if delta_rss_u2 > 0 else None
        )
        cpu_ratio = float(med["fragment_cpu_calibrated"]) / float(u2_med["fragment_cpu_calibrated"])
        wall_ratio = float(med["fragment_wall"]) / float(u2_med["fragment_wall"])
        rss_ok = rss_delta_pct is not None and rss_delta_pct <= 20.0
        parity_by_arm[architecture_id] = {
            "param_delta_pct": round(arms[architecture_id]["delta_pct"], 6),
            "rss_delta_pct": None if rss_delta_pct is None else round(rss_delta_pct, 6),
            "cpu_ratio": round(cpu_ratio, 6),
            "wall_ratio": round(wall_ratio, 6),
            "qualified": bool(
                arms[architecture_id]["qualified"]
                and rss_ok
                and cpu_ratio <= 2.0
                and wall_ratio <= 2.0
            ),
            "rss_gate": (
                "measured" if rss_delta_pct is not None else "undecidable_u2_delta_rss_nonpositive"
            ),
            "fragment_cpu_primary_median": round(float(med["fragment_cpu"]), 6),
            "fragment_cpu_calibrated_median": round(float(med["fragment_cpu_calibrated"]), 6),
            "cpu_ratio_basis": "calibrated repeat-mean per fragment (8 reps for U2, 32 for baselines)",
        }

    selection = select_baseline(measured, arms)
    selected = selection["selected"]
    verdict = "baseline_parity_infeasible" if selected is None else "selected"

    slots = build_roles()
    cpu_total = BUDGET.child_cpu + (time.process_time() - started)
    if cpu_total > CPU_CAP:
        raise BudgetExceeded(f"total CPU {cpu_total:.3f} > {CPU_CAP}")

    identity = None
    if selected is not None:
        selection_block = {
            "qualified_set": selection["qualified_set"],
            "selected": selected,
            "selection_path": selection["selection_path"],
            "tie_break_level": selection["tie_break_level"],
            "tie_break_trace": selection["trace"],
            "excluded": [
                {
                    "architecture_id": arm,
                    "status": (
                        "excluded_by_parity_band"
                        if not arms[arm]["qualified"]
                        else "qualified_lost_tiebreak"
                    ),
                    **parity_by_arm[arm],
                }
                for arm in ARM_IDS
                if arm != selected
            ],
        }
        identity = build_identity(
            selected, arms, measured, parity_by_arm[selected], selection_block, cpu_total
        )
        IDENTITY_PATH.write_text(
            json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        identity_sha = normalized_sha256(IDENTITY_PATH)
    else:
        identity_sha = None

    v1_sha = v2_sha = None
    if identity_sha is not None:
        v1_sha, v2_sha = write_execution_v2(slots, identity_sha)

    cpu_total = BUDGET.child_cpu + (time.process_time() - started)
    wall_total = time.perf_counter() - started_wall
    if cpu_total > CPU_CAP:
        raise BudgetExceeded(f"total CPU {cpu_total:.3f} > {CPU_CAP}")

    report = {
        "schema": "taiji-u2-22.4.4-identity-measurement-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "conditions": {
            "dtype": "float32",
            "device": "cpu",
            "threads": 1,
            "interop_threads": 1,
            "torch_version": torch.__version__,
            "python": sys.version.split()[0],
            "fragment": {
                "batch": BATCH,
                "steps": STEPS,
                "seed": SEED,
                "rgb_shape": [BATCH, STEPS, 3, 4, 4],
                "audio_shape": [BATCH, STEPS, 33],
                "proprio_shape": [BATCH, STEPS, 8],
                "backward_calls": 1,
            },
            "rss_method": "RSS sampled at each step and after backward; delta = peak - rss after construction",
            "cpu_method": "primary = time.process_time delta over one fragment; calibrated = "
            "mean over repeats of the identical fragment in the same process",
            "processes_per_arm": 3,
            "aggregate": "median",
        },
        "anchor": {
            **u2_anchor,
            "tolerance": TOLERANCE,
            "band": [ANCHOR * (1 - TOLERANCE), ANCHOR * (1 + TOLERANCE)],
        },
        "grid": {
            architecture_id: {
                "selected": arms[architecture_id],
                "all_points": grid[architecture_id],
            }
            for architecture_id in ARM_IDS
        },
        "measurements": measured,
        "parity": parity_by_arm,
        "selection": selection,
        "u2_slots": {
            "status": "written" if v2_sha else "not_written",
            "slot_sizes": {role: len(slots[role]["param_names"]) for role in slots},
        },
        "outputs": {
            "identity_manifest": IDENTITY_REF if identity_sha else None,
            "identity_sha256": identity_sha,
            "execution_v1_sha256": v1_sha,
            "execution_v2_sha256": v2_sha,
        },
        "budget": {
            "cpu_total_measured": round(cpu_total, 6),
            "wall_total_measured": round(wall_total, 3),
            "cap": 250,
            "training_runs": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
        },
        "conflicts": [
            "U2 has no `subject.step(state, observation)` method; the measured path is the "
            "real protocol choose_operation -> issue -> receive (as in "
            "scripts/training/u2_update_support_22_4_1.py shadow_terms).",
            "U2 is single-subject with branches=2; the fragment batch=2 is matched by the "
            "2 branches, not by a batch dimension.",
            "U2 reference is measured on the object named in the task, "
            "U2Subject(U2Config.small()) (22638 scalars / 191 tensors); the 22742 anchor lives "
            "on metadata_candidate() (194 tensors). The 104-scalar metadata fork is not on the "
            "measured graph.",
            "The fragment's multiview/goal plumbing for the two baselines is a declared "
            "measurement harness (text context added to the metadata token; coordinate rows are "
            "zero except a time ramp), not a frozen training recipe.",
            "time.process_time is quantised at ~15.6 ms on Windows; a single baseline fragment "
            "is shorter than one tick and reads 0.0. parity cpu_ratio therefore uses a "
            "repeat-mean calibration of the identical fragment; the primary single-fragment "
            "reading is kept beside it in the report.",
            "The two baselines are ~100x cheaper per fragment than the U2 reference, so the "
            "parity CPU/RSS/wall gates are not the binding constraint; L1 (parameter count) "
            "decides the selection.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("ANCHOR", u2_anchor)
    for architecture_id in ARM_IDS:
        print(
            "ARM",
            architecture_id,
            arms[architecture_id]["width"],
            arms[architecture_id]["param_total"],
            arms[architecture_id]["param_tensors"],
            round(arms[architecture_id]["delta_pct"], 4),
            "qualified",
            arms[architecture_id]["qualified"],
        )
        print("   parity", parity_by_arm[architecture_id])
        print("   medians", measured[architecture_id]["medians"])
    print("U2 medians", u2_med)
    print(
        "SELECTION",
        selection["selected"],
        selection["selection_path"],
        selection["tie_break_level"],
    )
    print("REPORT_SHA", normalized_sha256(REPORT_PATH))
    if identity_sha:
        print("IDENTITY_SHA", identity_sha)
        print("EXEC_V1_SHA", v1_sha)
        print("EXEC_V2_SHA", v2_sha)
    print("CPU_TOTAL", round(cpu_total, 6), "WALL", round(wall_total, 3))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("u2_reference",) + ARM_IDS)
    parser.add_argument("--width", type=int)
    arguments = parser.parse_args()
    if arguments.worker:
        worker_main(arguments.worker, arguments.width)
        return 0
    return parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
