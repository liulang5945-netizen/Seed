"""E2b-3/E3-precheck: bundle the value-token feature path and restore it fresh.

The bundle contains the trainable value-token encoder, query selector, closed-
form value readout, the active content store and the replacement facts. A fresh
process reloads it and reruns the same relevant/irrelevant/cut/rescue controls.
No optimizer update, no E1 sealed text and no producer/V4 reload is needed for
this component-level restore.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

READOUT_ROOT = ROOT / "data/checkpoints/n1_plan177_e1_readout_value_token_20261029"
SOURCE_REPORT = ROOT / "reports/taiji_n1_plan177_e1_value_readout_value_token_20261029.json"
BUNDLE_ROOT = ROOT / "data/checkpoints/n1_plan177_e1_value_feature_bundle_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_e1_value_feature_restore_20261029.json"
BUNDLE_FORMAT = "taiji-n177-e1-value-feature-bundle-v1"
TOLERANCE = 1e-9


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bundle() -> dict:
    from taiji.joint_state_consumption import (
        PersistentContentFeatureV1,
        PersistentContentStoreV1,
        QueryConditionedSelectorV1,
        ValueTokenFeatureEncoderV1,
    )
    from taiji.joint_state_consumption_course import E1_COLORS
    from taiji.joint_state_readout import FeatureValueReadoutV1

    readout_path = READOUT_ROOT / "readout.pt"
    prototype_path = READOUT_ROOT / "prototypes.pt"
    readout_payload = torch.load(readout_path, map_location="cpu", weights_only=False)
    prototype_payload = torch.load(prototype_path, map_location="cpu", weights_only=False)
    feature_dim = int(readout_payload["feature_dim"])
    hidden_dim = int(readout_payload["hidden_dim"])
    encoder = ValueTokenFeatureEncoderV1(dim=feature_dim)
    selector = QueryConditionedSelectorV1()
    readout = FeatureValueReadoutV1(feature_dim=feature_dim, hidden_dim=hidden_dim)
    readout.load_state_dict(readout_payload["state_dict"])
    values = list(prototype_payload["values"])
    sources = list(prototype_payload["sources"])
    if values != list(E1_COLORS) or len(sources) != len(values):
        raise ValueError("value-token prototype ledger differs from the frozen colour order")

    def fact(colour: str, revision: int, entity: str) -> PersistentContentFeatureV1:
        index = values.index(colour)
        return PersistentContentFeatureV1.create(
            entity=entity,
            attribute="颜色",
            value=colour,
            source_input_id=str(sources[index]),
            source_sha256=_source_sha256(f"{entity}的颜色是{colour}。"),
            turn_index=0,
            revision=revision,
            valid_from=0,
            confidence=0.9,
            continuous=tuple(float(item) for item in encoder.encode(colour).tolist()),
        )

    def fixture(colour: str, revision: int) -> PersistentContentFeatureV1:
        text = f"杯子的颜色是{colour}。"
        return PersistentContentFeatureV1.create(
            entity="杯子",
            attribute="颜色",
            value=colour,
            source_input_id=f"e2b3-fixture:cup:{colour}",
            source_sha256=_source_sha256(text),
            turn_index=0,
            revision=revision,
            valid_from=0,
            confidence=0.9,
            continuous=tuple(float(item) for item in encoder.encode(colour).tolist()),
        )

    key_blue = fact("蓝色", 0, "钥匙")
    key_green = fact("绿色", 1, "钥匙")
    cup_black = fixture("黑色", 0)
    cup_white = fixture("白色", 1)
    store = PersistentContentStoreV1(()).add(key_blue).add(cup_black)
    return {
        "format": BUNDLE_FORMAT,
        "version": 1,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "encoder_state": encoder.state_dict(),
        "selector_state": selector.state_dict(),
        "readout_state": readout.state_dict(),
        "store": store.to_payload(),
        "replacements": {
            "key_green": key_green.to_payload(),
            "cup_white": cup_white.to_payload(),
        },
        "prototype_features": [list(map(float, row)) for row in prototype_payload["features"]],
        "prototype_hiddens": [list(map(float, row)) for row in prototype_payload["hiddens"]],
        "values": values,
        "source_sha256": {
            "readout": _sha256_file(readout_path),
            "prototypes": _sha256_file(prototype_path),
            "source_report": _sha256_file(SOURCE_REPORT),
        },
    }


def _controls(bundle_path: Path) -> dict:
    from taiji.joint_state_consumption import (
        PersistentContentFeatureV1,
        PersistentContentStoreV1,
        QueryConditionedSelectorV1,
        ValueTokenFeatureEncoderV1,
    )
    from taiji.joint_state_readout import FeatureValueReadoutV1

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("format") != BUNDLE_FORMAT or bundle.get("version") != 1:
        raise ValueError("E2b-3 bundle format differs")
    encoder = ValueTokenFeatureEncoderV1(dim=int(bundle["feature_dim"]))
    selector = QueryConditionedSelectorV1()
    readout = FeatureValueReadoutV1(
        feature_dim=int(bundle["feature_dim"]), hidden_dim=int(bundle["hidden_dim"])
    )
    encoder.load_state_dict(bundle["encoder_state"])
    selector.load_state_dict(bundle["selector_state"])
    readout.load_state_dict(bundle["readout_state"])
    store = PersistentContentStoreV1.from_payload(bundle["store"])
    replacements = {
        key: PersistentContentFeatureV1.from_payload(value)
        for key, value in bundle["replacements"].items()
    }
    prototypes = {
        value: (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(hidden, dtype=torch.float32),
        )
        for value, feature, hidden in zip(
            bundle["values"], bundle["prototype_features"], bundle["prototype_hiddens"], strict=True
        )
    }

    def selected(current_store):
        selection = selector.select(
            entity="钥匙", attribute="颜色", candidates=current_store.active_features()
        )
        if selection is None:
            return None
        feature, hidden = prototypes[selection.selected_feature.value]
        result = readout.read(feature, hidden)
        return {
            "selected_feature_id": selection.selected_feature.feature_id,
            "selected_value": result.selected_value,
            "logits": list(result.logits),
            "copy_only_value": selection.selected_feature.value,
            "copy_only_matches": selection.selected_feature.value == result.selected_value,
        }

    baseline = selected(store)
    relevant_store = store.swap("钥匙", "颜色", replacements["key_green"])
    relevant = selected(relevant_store)
    irrelevant_store = relevant_store.swap("杯子", "颜色", replacements["cup_white"])
    irrelevant = selected(irrelevant_store)
    cut_store = irrelevant_store.cut("钥匙", "颜色")
    cut = selected(cut_store)
    rescued_store = cut_store.add(replacements["key_green"])
    rescued = selected(rescued_store)
    return {
        "baseline": baseline,
        "relevant_swap": relevant,
        "irrelevant_swap": irrelevant,
        "cut": cut,
        "rescue": rescued,
    }


def _compare(parent: dict, child: dict) -> dict:
    max_difference = 0.0
    same_values = True
    for key in ("baseline", "relevant_swap", "irrelevant_swap", "rescue"):
        left, right = parent[key], child[key]
        same_values &= left["selected_value"] == right["selected_value"]
        for a, b in zip(left["logits"], right["logits"], strict=True):
            max_difference = max(max_difference, abs(float(a) - float(b)))
    same_values &= parent["cut"] is None and child["cut"] is None
    return {
        "same_selected_values": bool(same_values),
        "max_logit_difference": max_difference,
        "passed": bool(same_values and max_difference <= TOLERANCE),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2b-3 value-feature bundle restore precheck")
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    if args.bundle_root.exists() or args.report.exists():
        raise ValueError("E2b-3 bundle/report are write-once")
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    args.bundle_root.mkdir(parents=True)
    bundle_path = args.bundle_root / "bundle.pt"
    bundle = _bundle()
    torch.save(bundle, bundle_path)
    parent = _controls(bundle_path)
    child_code = (
        "import json, sys\n"
        "from scripts.training.precheck_taiji_n1_plan177_e1_value_feature_restore import _controls\n"
        "print(json.dumps(_controls(sys.argv[1]), ensure_ascii=False))\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", child_code, str(bundle_path)],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        child = None
        comparison = {
            "same_selected_values": False,
            "max_logit_difference": None,
            "passed": False,
            "child_error": result.stderr,
        }
    else:
        child = json.loads(result.stdout.strip().splitlines()[-1])
        comparison = _compare(parent, child)
    report = {
        "schema": "taiji-n177-e1-value-feature-restore-v1",
        "status": (
            "zero_step_restore_passed" if comparison["passed"] else "zero_step_restore_failed"
        ),
        "bundle": str(bundle_path.relative_to(ROOT)),
        "bundle_sha256": _sha256_file(bundle_path),
        "bundle_source_sha256": bundle["source_sha256"],
        "feature_dim": bundle["feature_dim"],
        "hidden_dim": bundle["hidden_dim"],
        "parent_controls": parent,
        "child_controls": child,
        "comparison": comparison,
        "copy_only": {
            "baseline": parent["baseline"]["copy_only_value"],
            "relevant_swap": parent["relevant_swap"]["copy_only_value"],
            "matches_main": all(
                parent[key]["copy_only_matches"]
                for key in ("baseline", "relevant_swap", "irrelevant_swap", "rescue")
            ),
            "interpretation": (
                "copy-only is a lower-bound shortcut over the discrete value span; "
                "it is not evidence of learned selection"
            ),
        },
        "optimizer_updates": 0,
        "model_calls": 0,
        "sealed_text_read": 0,
        "checkpoint_loads": 2,
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
                "same_values": comparison["same_selected_values"],
                "max_logit_difference": comparison["max_logit_difference"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
