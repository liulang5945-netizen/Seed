"""Measure inductor speedups on the hot paths (run under output/_compile_run.bat)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import generate_examples, prepare  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_compile_speed_20261118.json"
REPEATS = 20


def timed(fn, repeats: int = REPEATS) -> float:
    fn()
    started = time.process_time()
    for _ in range(repeats):
        fn()
    return (time.process_time() - started) / repeats


def main() -> int:
    ex = generate_examples("train", 4)[0]
    model = (
        U2Subject(replace(U2Config.small(), seed=20261110, readout="pixel-salience"))
        .metadata_candidate()
        .request_candidate()
    )
    state = prepare(model, ex["public"], ex["id"])
    goal = model.language_goal(state)
    action = model.native_w.new_zeros(15)
    result = {"torch": torch.__version__}

    result["branch_eager"] = round(timed(lambda: model.branch_context(state, goal, action)), 6)
    try:
        compiled = torch.compile(model.branch_context, dynamic=False)
        result["branch_compiled"] = round(timed(lambda: compiled(state, goal, action)), 6)
    except Exception as error:  # noqa: BLE001
        result["branch_error"] = f"{type(error).__name__}: {str(error)[:140]}"

    readout = model.pixel_readout
    blocks, grid = state["pixel_blocks"], state["pixel_grid"]
    request = torch.tensor(list(ex["public"]["request"]))
    result["readout_eager"] = round(timed(lambda: readout(blocks, grid, request)), 6)
    try:
        compiled_readout = torch.compile(readout, dynamic=False)
        result["readout_compiled"] = round(
            timed(lambda: compiled_readout(blocks, grid, request)), 6
        )
    except Exception as error:  # noqa: BLE001
        result["readout_error"] = f"{type(error).__name__}: {str(error)[:140]}"

    from scripts.training.first_a_support_22_5 import accept, output  # noqa: PLC0415

    probability, _, details = output(model, state)
    symbol = int(ex["target"][0])
    result["accept_eager"] = round(timed(lambda: model.accept_byte(state, symbol, details), 10), 6)
    try:
        compiled_accept = torch.compile(model.accept_byte, dynamic=False)
        result["accept_compiled"] = round(
            timed(lambda: compiled_accept(state, symbol, details), 10), 6
        )
    except Exception as error:  # noqa: BLE001
        result["accept_error"] = f"{type(error).__name__}: {str(error)[:140]}"
    del accept, probability

    OUT.write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
