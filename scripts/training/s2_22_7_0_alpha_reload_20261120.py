"""Independent-process reload helper for the 22.7.0-alpha interface probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.u2_neural_units import (  # noqa: E402
    LEGAL_INPUT_CHANNELS,
    RouteBGraph,
    UnitGraphSpec,
    apply_capsule,
    load_capsule,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capsule", required=True)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    capsule = load_capsule(args.capsule)
    spec = UnitGraphSpec(**capsule["runtime"]["spec"])
    graph = RouteBGraph(spec)
    apply_capsule(graph, capsule)
    features = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    channels = {
        name: torch.tensor(features[key])
        for name, key in zip(LEGAL_INPUT_CHANNELS, features, strict=False)
    }
    windows = channels[LEGAL_INPUT_CHANNELS[0]].shape[0]
    gates = torch.ones(windows, graph.units.n_units)
    with torch.no_grad():
        readout = graph(channels, gates, graph.initial_state())["readout"]
    print(hashlib.sha256(readout.detach().numpy().tobytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
