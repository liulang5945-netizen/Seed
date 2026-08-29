"""复现：全选数据集触发 400（列表混入非语料文件）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from fastapi import HTTPException  # noqa: E402

from api.training.datasets import list_train_files  # noqa: E402
from api.training.native import _resolve_native_datasets  # noqa: E402
from seed.datasets import inspect_native_dataset  # noqa: E402


def main() -> None:
    listed = list_train_files()["files"]
    print("前端可见数据集（全选即发送这些）:", listed)

    print("\n== 逐个 native 合同体检 ==")
    for name in listed:
        try:
            hits = [
                p
                for p in (Path(_ROOT, "data", name),)
                if p.is_file()
            ]
            if not hits:
                print(f"  {name}: 未在 data/ 命中")
                continue
            report = inspect_native_dataset(hits[0])
            print(
                f"  {name}: native_trainable={report.native_trainable} "
                f"docs={report.documents} invalid={report.invalid_records}"
            )
        except Exception as exc:
            print(f"  {name}: 体检异常 {exc}")

    print("\n== 模拟『全选 → 提交训练』 ==")
    try:
        paths = _resolve_native_datasets(listed)
        print("  通过，解析到:", [p.name for p in paths])
    except HTTPException as exc:
        detail = exc.detail
        msg = detail.get("message") if isinstance(detail, dict) else detail
        print(f"  ❌ HTTP {exc.status_code}: {msg}")
        if isinstance(detail, dict):
            for item in detail.get("datasets", []):
                print("     拒绝:", Path(item["path"]).name, item["errors"][:1])

    print("\n== 真实语料是否可见 ==")
    for corpus in sorted(Path(_ROOT, "data", "simple_zh").glob("*.jsonl")):
        print(f"  {corpus.relative_to(_ROOT)} 在列表中: {corpus.name in listed}")


if __name__ == "__main__":
    main()
