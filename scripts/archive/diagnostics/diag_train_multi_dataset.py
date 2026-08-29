"""复现：多资料一次连续训练 + loss 事件 + 生命数据。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from seed_platform.paths import get_external_path  # noqa: E402


def probe_dataset_listing() -> None:
    from api.training.datasets import _get_all_data_dirs, list_train_files

    print("== data dirs ==")
    for d in _get_all_data_dirs():
        print("  ", d, Path(d).is_dir())
    listed = list_train_files()["files"]
    print("== /api/train/files ==")
    print("  count:", len(listed))
    for name in listed:
        print("  -", name)


def probe_resolve(names: list[str]) -> None:
    from api.training.resume import _resolve_datasets

    resolved, missing = _resolve_datasets(names)
    print("== _resolve_datasets ==")
    print("  requested:", names)
    print("  resolved :", [str(p) for p in resolved])
    print("  missing  :", missing)


def probe_multi_stream() -> None:
    from seed.datasets import inspect_native_dataset
    from api.training.resume import _iter_symbols

    data_dir = Path(get_external_path("data"))
    a = data_dir / "_diag_multi_a.jsonl"
    b = data_dir / "_diag_multi_b.jsonl"
    a.write_text(
        "\n".join(json.dumps({"text": f"甲{i}"}, ensure_ascii=False) for i in range(3)),
        encoding="utf-8",
    )
    b.write_text(
        "\n".join(json.dumps({"text": f"乙{i}"}, ensure_ascii=False) for i in range(3)),
        encoding="utf-8",
    )
    try:
        probe_resolve([a.name, b.name])
        reports = [inspect_native_dataset(p) for p in (a, b)]
        total = sum(r.total_text_bytes for r in reports)
        print("== inspect ==")
        for r in reports:
            print("  ", Path(r.path).name, r.native_trainable, r.documents, r.total_text_bytes)
        print("  total_bytes:", total)
        counted = sum(flag for _, flag in _iter_symbols([a, b], 256))
        print("== _iter_symbols counted bytes ==", counted, "== total?", counted == total)
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)


def probe_life() -> None:
    from seed_platform.runtime_service import _life_section
    from seed_platform.dependencies import legacy_requested

    print("== life ==")
    print("  legacy_requested:", legacy_requested())
    print("  _life_section:", _life_section())
    try:
        from api.seed_runtime import get_seed_runtime, is_seed_active

        print("  is_seed_active:", is_seed_active())
        runtime = get_seed_runtime()
        if runtime is not None:
            status = runtime.status()
            print("  status keys:", sorted(status))
    except Exception as exc:
        print("  seed runtime unavailable:", exc)


if __name__ == "__main__":
    probe_dataset_listing()
    probe_multi_stream()
    probe_life()
