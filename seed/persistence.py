"""Checkpoint persistence utilities for the seed-native-v1 envelope.

两个工程硬化能力（2026-08-23 公测路线图 M0）：

- **原子落盘**：先写同目录临时文件再 ``os.replace``。任何时刻崩溃都只
  损失本次落盘本身，既有的目标文件永远是完整的上一版本。进程被杀等
  无法执行清理的场景会留下半写临时文件；下一次成功落盘前会先清扫同目
  录的陈旧 ``<目标名>.*.tmp``，保证检查点目录自愈。
- **信封元数据**：``attach_metadata`` 在信封顶层加 ``metadata`` 键
  （tick / 训练画像摘要 / 时间戳 / 可选语料指纹）。``Seed.restore`` 只
  校验 ``format`` / ``config`` / ``substrate``，多余键被忽略，因此旧检查
  点照常被加载，新信封也向后兼容。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import logging

logger = logging.getLogger(__name__)


def atomic_save(payload: Mapping[str, Any], path: Path | str) -> Path:
    """``torch.save`` with crash-safe replace semantics.

    The temporary file is created in the destination directory so the final
    ``os.replace`` is atomic on the same filesystem.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Self-heal: a killed process cannot run the cleanup branch below, so a
    # successful save sweeps stale half-written temporals from the previous
    # crashed attempt before writing its own.
    for stale in target.parent.glob(target.name + ".*.tmp"):
        try:
            stale.unlink()
        except OSError as e:
            logger.debug("【atomic_save】处理失败（非致命）: %s", e)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(dict(payload), tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def attach_metadata(
    envelope: Mapping[str, Any],
    *,
    tick: Optional[int] = None,
    corpus_fingerprint: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a shallow copy of ``envelope`` carrying a ``metadata`` block.

    Existing metadata (e.g. from a resumed envelope) is preserved and only
    overwritten for keys supplied here, so resume chains keep the original
    corpus fingerprint unless it is re-stated.
    """

    updated = dict(envelope)
    metadata = dict(updated.get("metadata") or {})
    metadata["saved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if tick is not None:
        metadata["tick"] = int(tick)
    if corpus_fingerprint is not None:
        metadata["corpus_fingerprint"] = str(corpus_fingerprint)
    config = updated.get("config")
    if isinstance(config, Mapping) and "taiji" in config:
        taiji_cfg = config.get("taiji")
        if isinstance(taiji_cfg, Mapping):
            metadata.setdefault(
                "profile",
                {
                    "region_sizes": list(taiji_cfg.get("region_sizes", [])),
                    "memory_units": taiji_cfg.get("memory_units"),
                    "alphabet_size": taiji_cfg.get("alphabet_size"),
                },
            )
    if extra:
        metadata.update(dict(extra))
    updated["metadata"] = metadata
    return updated


def corpus_fingerprint(paths: Any) -> str:
    """Stable, cheap fingerprint: sorted names + byte sizes (no full hashing).

    Full-content hashing of multi-GB corpora is prohibitively expensive on
    every checkpoint; name+size detects the realistic drift (corpus swapped
    or truncated) without reading file contents.
    """

    entries = []
    for raw in paths:
        path = Path(raw)
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        entries.append({"name": path.name, "bytes": size})
    entries.sort(key=lambda item: item["name"])
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
