"""Evaluate the Taiji A1 assembly-relation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_a1_relation_manifest import (  # noqa: E402
    relation_corpus_from_manifest,
)
from taiji import AssemblyRelationEvaluator, TaijiConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    corpus = relation_corpus_from_manifest(manifest)
    report = AssemblyRelationEvaluator(TaijiConfig()).evaluate(corpus)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
