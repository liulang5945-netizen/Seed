"""Independent-process reload helper for the 22.7.0-pre lifecycle capsule.

Loads a capsule saved by the lifecycle probe, rebuilds the RouteB graph from the
recorded runtime spec, applies the capsule strictly, runs one deterministic
forward pass and writes the readout digest + registry summary as JSON.
"""

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

TIME_STEPS = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capsule", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    capsule = load_capsule(args.capsule)
    spec_payload = capsule["runtime"]["spec"]
    spec = UnitGraphSpec(**spec_payload)
    graph = RouteBGraph(spec)
    apply_capsule(graph, capsule)
    generator = torch.Generator().manual_seed(20261120)
    channels = {
        name: torch.randn(TIME_STEPS, spec.input_dim, dtype=spec.torch_dtype, generator=generator)
        for name in LEGAL_INPUT_CHANNELS
    }
    gates = torch.ones(TIME_STEPS, graph.units.n_units)
    with torch.no_grad():
        readout = graph(channels, gates, graph.initial_state())["readout"]
    digest = hashlib.sha256(readout.detach().numpy().tobytes()).hexdigest()
    payload = {
        "digest": digest,
        "units": graph.units.n_units,
        "retired": sum(
            1 for unit in graph.units.identities["units"] if unit["lifecycle"] == "retired"
        ),
    }
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
