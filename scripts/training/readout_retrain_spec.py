"""R2 受控重训语言读出：臂规格与语料血缘切片的**唯一来源**。

合同草案：``plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md`` §3（三条臂）与 §4（血缘切片）。
标定仪器（``calibrate_taiji_r2_readout_retrain.py``）与正式 runner
（``train_taiji_r2_readout_retrain.py``）都从这里取定义——两处各写一份慢慢漂移，本仓已经吃过这种亏。

**臂的开关为什么长这样**（读码结论，不是猜的）：``Taiji.observe`` 有两个互斥写入站——
``motor.learn`` 挂在 ``readout == "action"``（``taiji/model.py`` 的 action 分支），
``predictive_readout.learn`` 挂在 ``readout == "predictive"`` 的读出门控下。所以：

* A 臂走 ``readout="predictive"``，只开 ``learn_predictive_readout``，其余全关 ⇒ 只动读出头；
* B 臂走 ``readout="action"``，只动运动面 ⇒ 读出冻结；
* C 臂 ``learn=False`` ⇒ 两处都不动，是"权重没变时的漂移"对照。

⚠️ ``readout`` **不允许在同一个 dynamics episode 内切换**（``model.py`` 会在切换时抛
``readout changed inside an active dynamics episode``）⇒ 每次开跑前必须 ``reset_dynamics()``。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from train_seed_corpus import iter_corpus_symbols  # noqa: E402

#: ``Taiji`` 上能被本件改动的两处权重面。报告的写入面判定只认这两个指纹。
SURFACES = ("motor", "predictive_readout")

#: 三条臂。``write_surface`` 是**预期**写入面；跑完要核，核不过就是仪器/配置不可信。
ARMS: dict[str, dict[str, Any]] = {
    "A": {
        "description": "只训读出头：运动面冻结，predictive_readout 单点写入",
        "observe_kwargs": {
            "learn": True,
            "readout": "predictive",
            "learn_motor": False,
            "learn_fabric": False,
            "learn_predictive_context": False,
            "learn_predictive_readout": True,
        },
        "write_surface": ("predictive_readout",),
    },
    "B": {
        "description": "只训运动面：原运动面路径，读出冻结",
        "observe_kwargs": {
            "learn": True,
            "readout": "action",
            "learn_motor": True,
            "learn_predictive_readout": False,
        },
        "write_surface": ("motor",),
    },
    "C": {
        "description": "双冻结漂移对照：符号流过，两处都不学",
        "observe_kwargs": {"learn": False, "readout": "action"},
        "write_surface": (),
    },
}

#: 所有者 2026-09-20 批准的符号上限（每臂）。代码里也钉一道，免得"批了 16M、跑了 160M"。
APPROVED_SYMBOL_CEILING = 16_000_000

#: 训练器名。跑器写进 checkpoint 信封，判决器据此确认"这三份权重确实是本件训出来的"。
TRAINER_NAME = "train_taiji_r2_readout_retrain"

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "p3b_all_fresh.jsonl"
#: 这份清单里写着它自己的血缘推导（``skip_derivation`` 与 ``skip_symbols``）——
#: 也就是"该副本已经吃过的前缀"是怎么算出来的。不许手抄。
DEFAULT_LINEAGE_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_all_fresh_manifest.json"


def arm_spec(name: str) -> dict[str, Any]:
    key = name.strip().upper()
    if key not in ARMS:
        raise SystemExit(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[key]


def arm_observe_kwargs(name: str) -> dict[str, Any]:
    return dict(arm_spec(name)["observe_kwargs"])


def arm_write_surface(name: str) -> tuple[str, ...]:
    return tuple(arm_spec(name)["write_surface"])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lineage_skip(manifest_path: Path, corpus_path: Path) -> dict[str, Any]:
    """Read the lineage-derived skip point for ``corpus_path`` out of its manifest.

    The point of going through the manifest is that the skip is *derived* from checkpoint lineage
    (``skip_derivation``) rather than typed by hand.  A corpus that is not the manifest's own
    output has no such derivation, and this function says so instead of guessing a number.
    """

    if not manifest_path.is_file():
        raise SystemExit(f"lineage manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = str(manifest.get("output", ""))
    if Path(recorded).name != corpus_path.name:
        raise SystemExit(
            f"{corpus_path.name} is not the output recorded in {manifest_path.name} "
            f"(it says {recorded!r}); there is no lineage-derived skip for it"
        )
    skip = manifest.get("skip_symbols")
    if not isinstance(skip, int) or skip < 0:
        raise SystemExit(f"{manifest_path.name} carries no usable skip_symbols")
    try:
        recorded_path = manifest_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # A manifest outside the repo is legitimate for a fixture; do not invent a relative path.
        recorded_path = str(manifest_path)
    return {
        "manifest": recorded_path,
        "skip_symbols": skip,
        "first_emitted_row": manifest.get("first_emitted_row"),
        "replay_symbols_from_seen_region": manifest.get("replay_symbols_from_seen_region"),
        "skip_derivation": manifest.get("skip_derivation"),
    }


def iter_corpus_window(
    corpus_paths: Sequence[Path], skip_symbols: int
) -> Iterator[int]:
    """Stream the corpus from *after* the already-consumed prefix.

    A generator on purpose: the formal arms take 16M symbols, and materialising that as a list
    would cost hundreds of megabytes of Python objects for no reason.
    """

    stream: Iterator[int] = iter_corpus_symbols(corpus_paths)
    for consumed in range(skip_symbols):
        try:
            next(stream)
        except StopIteration as exc:  # pragma: no cover - short corpus
            raise SystemExit(
                f"corpus is shorter than the lineage skip ({skip_symbols} symbols; "
                f"exhausted after {consumed})"
            ) from exc
    yield from stream


def take_symbols(corpus_paths: Sequence[Path], skip_symbols: int, count: int) -> list[int]:
    """The next ``count`` symbols after the already-consumed prefix (small-count helper)."""

    taken: list[int] = []
    for symbol in iter_corpus_window(corpus_paths, skip_symbols):
        taken.append(symbol)
        if len(taken) >= count:
            return taken
    raise SystemExit(f"corpus ran out after {len(taken)} of {count} symbols")
