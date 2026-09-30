"""B-3 C-3 黄金等价测试：`_fingerprint_route` 抽离到 `_cortex_routing.fingerprint_route` 后，
**对同一装配（同种子）复现迁移前实现**的路由结果（15 组：5 prompt × 3 top_k）。

夹具：`assemble_cortex()` + `torch.manual_seed(20260926)`（与采集时同种子 ⇒ fallback 神经元随机初始化确定。
⚠️ 覆盖说明：fallback general 神经元无 embed_adapter ⇒ **相似度分支未被覆盖**（覆盖早退与空 sims 路径）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neuroplex.brain import _cortex_routing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (PROJECT_ROOT / "reports" / "cortex_fingerprint_golden_20260926.json").read_text(
        encoding="utf-8"
    )
)
FIX_SEED = int(GOLDEN["seed"])


@pytest.fixture(scope="module")
def cortex():
    torch.manual_seed(FIX_SEED)
    from neuroplex.loader import assemble_cortex

    cortex_obj, _tok, _extra = assemble_cortex()
    return cortex_obj


def test_helper_reproduces_golden(cortex) -> None:
    for row in GOLDEN["rows"]:
        general_ids = list(cortex._general_sp.encode(row["prompt"]))
        got = _cortex_routing.fingerprint_route(
            cortex.neurons,
            cortex._shared_embedding,
            cortex.device,
            general_ids,
            row["top_k"],
        )
        assert list(got) == row["out"], f"prompt={row['prompt']!r} top_k={row['top_k']} 不一致"


def test_delegate_still_matches_golden(cortex) -> None:
    """委托（Cortex 内）也应等于迁移前实现 ⇒ 证明调用点语义不变。"""

    for row in GOLDEN["rows"]:
        general_ids = list(cortex._general_sp.encode(row["prompt"]))
        assert list(cortex._fingerprint_route(general_ids, top_k=row["top_k"])) == row["out"]


def test_delegate_and_helper_agree_on_edge_cases(cortex) -> None:
    for general_ids in ([], [1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        for top_k in (0, 1, 99):
            via_delegate = list(cortex._fingerprint_route(general_ids, top_k=top_k))
            via_helper = list(
                _cortex_routing.fingerprint_route(
                    cortex.neurons,
                    cortex._shared_embedding,
                    cortex.device,
                    general_ids,
                    top_k,
                )
            )
            assert via_delegate == via_helper


def test_empty_neurons_returns_empty_selection() -> None:
    """无神经元 ⇒ 返回空列表（与原实现同语义）。"""

    picked = _cortex_routing.fingerprint_route({}, None, torch.device("cpu"), [1, 2, 3], 2)
    assert list(picked) == []
