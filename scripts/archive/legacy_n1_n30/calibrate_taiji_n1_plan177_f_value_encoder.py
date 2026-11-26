"""F0 train-only calibration for the byte-level value encoder.

The encoder sees only UTF-8 value bytes from the E1 train families. It learns
the colour identity through a linear readout; copy-only (one-hot address) and
no-state (zero feature) are reported as lower/upper-reference baselines. Zero
and final checkpoints are restored in a fresh process. No query/holdout text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.joint_state_consumption_course import E1_COLORS, e1_families  # noqa: E402
from taiji.joint_state_readout import FeatureValueReadoutV1  # noqa: E402
from taiji.joint_state_value_encoder import ValueByteEncoderV1  # noqa: E402

OUTPUT_ROOT = ROOT / "data/checkpoints/n1_plan177_f_value_encoder_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f_value_encoder_f0_20261029.json"
F0_CPU_LIMIT = 2000
F0_WALL_LIMIT = 1500


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _samples() -> list[tuple[torch.Tensor, int, str]]:
    rows = []
    for family in e1_families()["train"]:
        for colour in (family.value, family.old_value):
            rows.append(
                (
                    torch.tensor(list(colour.encode("utf-8")), dtype=torch.long),
                    E1_COLORS.index(colour),
                    colour,
                )
            )
    return rows


def _batch(rows, indices):
    return torch.stack([rows[index][0] for index in indices]), torch.tensor(
        [rows[index][1] for index in indices], dtype=torch.long
    )


def _logits_from_checkpoint(path: Path, colour: str) -> list[float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    encoder = ValueByteEncoderV1(
        byte_dim=int(payload["byte_dim"]), hidden_dim=int(payload["hidden_dim"])
    )
    readout = FeatureValueReadoutV1(feature_dim=int(payload["hidden_dim"]))
    encoder.load_state_dict(payload["encoder"])
    readout.load_state_dict(payload["readout"])
    with torch.no_grad():
        feature = encoder.encode_text(colour)
        logits = readout.linear(feature)
    return [float(item) for item in logits.tolist()]


def _child_restore(path: Path, colour: str) -> dict:
    code = (
        "import json, sys\n"
        "from scripts.training.calibrate_taiji_n1_plan177_f_value_encoder import _logits_from_checkpoint\n"
        "print(json.dumps({'logits': _logits_from_checkpoint(__import__('pathlib').Path(sys.argv[1]), sys.argv[2])}))\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), colour],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"status": "failed", "stderr": result.stderr}
    return {"status": "passed", **json.loads(result.stdout.strip().splitlines()[-1])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F0 train-only value-encoder calibration")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.report.exists():
        raise ValueError("F0 value-encoder calibration is write-once")
    if args.steps <= 0 or args.batch <= 0:
        raise ValueError("F0 steps and batch must be positive")
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    rows = _samples()
    labels = torch.tensor([row[1] for row in rows], dtype=torch.long)
    if len(rows) < 16 or len(set(labels.tolist())) != len(E1_COLORS):
        raise ValueError("F0 train facts do not cover the frozen colour set")
    encoder = ValueByteEncoderV1()
    readout = FeatureValueReadoutV1(feature_dim=encoder.hidden_dim)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(readout.parameters()), lr=args.lr
    )
    args.output_root.mkdir(parents=True)
    zero_path = args.output_root / "zero.pt"
    final_path = args.output_root / "final.pt"
    torch.save(
        {
            "byte_dim": encoder.byte_dim,
            "hidden_dim": encoder.hidden_dim,
            "encoder": encoder.state_dict(),
            "readout": readout.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        zero_path,
    )
    zero_restore = _child_restore(zero_path, "蓝色")
    if zero_restore["status"] != "passed":
        raise RuntimeError("F0 zero checkpoint failed fresh-process restore")
    losses = []
    generator = torch.Generator().manual_seed(1771029)
    for _step in range(args.steps):
        indices = torch.randint(0, len(rows), (args.batch,), generator=generator).tolist()
        raw, target = _batch(rows, indices)
        features = encoder(raw)
        logits = readout.linear(features)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if time.process_time() - cpu_start > F0_CPU_LIMIT:
            raise TimeoutError("F0 calibration reached its CPU guard")
        if time.monotonic() - wall_start > F0_WALL_LIMIT:
            raise TimeoutError("F0 calibration reached its wall guard")
    with torch.no_grad():
        all_raw = torch.stack([row[0] for row in rows])
        all_logits = readout.linear(encoder(all_raw))
        predictions = all_logits.argmax(dim=-1)
        train_accuracy = float((predictions == labels).float().mean())
        no_state_logits = readout.linear(torch.zeros(1, encoder.hidden_dim))
        no_state_prediction = int(no_state_logits.argmax(dim=-1))
        no_state_accuracy = float((labels == no_state_prediction).float().mean())
    torch.save(
        {
            "byte_dim": encoder.byte_dim,
            "hidden_dim": encoder.hidden_dim,
            "encoder": encoder.state_dict(),
            "readout": readout.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        final_path,
    )
    final_restore = _child_restore(final_path, "蓝色")
    if final_restore["status"] != "passed":
        raise RuntimeError("F0 final checkpoint failed fresh-process restore")
    same_final = max(
        abs(a - b) for a, b in zip(zero_restore["logits"], final_restore["logits"], strict=True)
    )
    report = {
        "schema": "taiji-n177-f-value-encoder-f0-v1",
        "status": "f0_calibration_completed" if train_accuracy >= 0.75 else "f0_encoder_failed",
        "feature_source": "value_span_bytes_gru",
        "train_samples": len(rows),
        "train_colours": len(set(labels.tolist())),
        "steps": args.steps,
        "batch": args.batch,
        "learning_rate": args.lr,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "train_accuracy": train_accuracy,
        "copy_only_accuracy": 1.0,
        "no_state_accuracy": no_state_accuracy,
        "zero_restore": zero_restore,
        "final_restore": final_restore,
        "zero_final_logit_difference": same_final,
        "checkpoint": {
            "zero": str(zero_path.relative_to(ROOT)),
            "zero_sha256": _sha256_file(zero_path),
            "final": str(final_path.relative_to(ROOT)),
            "final_sha256": _sha256_file(final_path),
        },
        "optimizer_updates": args.steps,
        "sealed_text_read": 0,
        "query_holdout_read": 0,
        "model_calls": 0,
        "can_promote": False,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.monotonic() - wall_start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "train_accuracy": train_accuracy,
                "no_state_accuracy": no_state_accuracy,
                "zero_final_logit_difference": same_final,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "f0_calibration_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
