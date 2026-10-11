"""DEBT-G106 修法①（owner 2026-10-11 四裁①之先行批文）的实施格契约册。

判据出处＝`plans/active/roadmap/05_TECH_DEBT_REGISTER.md` 的 DEBT-G106 行（先冻，原文不追改）：
* `J-N6b-1` 兼容锚——实施后**同一组输入**复跑，命中序列必须与封存基线**逐位相同**；
  不相等 ⇒「默认保持现状」这句作废并回 owner 重裁，**而不是**改基线（`G-N6b-1`）。
* `J-N6b-2` 新参数正反两支——显式 `episode_id="ep-A"` 时 `q_near_m0` 那一支**必须不含 `m3`**；
  不显式给 ⇒ 落回 `J-N6b-1` 那一份读数。
* `J-N6b-3` 默认位结论不变——默认位不动，所以实施后仍**不许**写「会话隔离已成立」，
  只许写「有口可筛、默认未筛」；本行原判「VISION §5.3 在存储层不成立」在默认位上继续为真。
* `G-N6b-2` 不许按名字判定——本仓另有两枚**同名不同类**的 `retrieve`，
  测与仪器一律按**类＋关键字参数名**分派，裸 grep 命中数不作证据（㊵-701 的第三层教训）。

夹具不自己造：记录表／查询表／store 全部复用封存基线那台仪器（`capture_taiji_episodic_retrieve_baseline`），
否则"同一组输入"这句话就是我手写出来的第二份真相（[[dont-reimplement-the-chain]]）。
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.episodic_memory import EpisodicMemoryStore  # noqa: E402

CAPTURE = PROJECT_ROOT / "scripts" / "training" / "capture_taiji_episodic_retrieve_baseline.py"
BASELINE_REPORT = PROJECT_ROOT / "reports" / "taiji_episodic_retrieve_baseline_20261011.json"

#: `G-N6b-1`：钉死**字面量**，不是"和件里一样"。谁想动锚，必须先让这一支红并回到 owner。
FROZEN_HITS_DIGEST = "3a51afedf04af23bea315436ee71cc7a238f3bede7c87a776a96482674a9bd1d"


def _capture():
    spec = importlib.util.spec_from_file_location("n6_07_capture", CAPTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hits(store, cue: torch.Tensor, **kwargs) -> list[str]:
    return [hit.record.memory_id for hit in store.retrieve(cue, limit=3, **kwargs)]


def test_default_face_still_matches_the_sealed_baseline() -> None:
    """`J-N6b-1` 的正支：默认位复跑必须与封存件逐位相同（含 digest）。"""

    module = _capture()
    rebuilt = module.build_baseline()
    sealed = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    assert rebuilt["hits"] == sealed["hits"], (rebuilt["hits"], sealed["hits"])
    assert rebuilt["hits_digest"] == sealed["hits_digest"]
    assert rebuilt["hits_digest"] == FROZEN_HITS_DIGEST, rebuilt["hits_digest"]
    #: 这一句是 `J-N6b-3` 的形状：默认位上跨会话命中仍然是**在场的事实**。
    assert rebuilt["cross_episode_hits_present_in_baseline"] is True, rebuilt


def test_retrieve_gains_a_keyword_only_optional_episode_id() -> None:
    """参数形状本身也要钉：必须是 kw-only、可选、默认 `None`＝不筛。"""

    parameters = inspect.signature(EpisodicMemoryStore.retrieve).parameters
    assert "episode_id" in parameters, dict(parameters)
    episode = parameters["episode_id"]
    assert episode.kind is inspect.Parameter.KEYWORD_ONLY, episode
    assert episode.default is None, episode


def test_explicit_scoping_drops_the_other_session() -> None:
    """`J-N6b-2` 能为假的那一支：显式给 ep-A 时，`q_near_m0` 必须筛掉属于 ep-B 的 `m3`。"""

    module = _capture()
    store = module._build_store()
    cue = torch.tensor(module.QUERY_TABLE["q_near_m0"], dtype=torch.float32)
    unscoped = _hits(store, cue)
    assert "m3" in unscoped, unscoped
    scoped = _hits(store, cue, episode_id="ep-A")
    assert "m3" not in scoped, scoped
    assert set(scoped) <= {memory_id for memory_id, ep in module.EPISODE_OF.items() if ep == "ep-A"}


def test_scoping_to_the_other_session_selects_its_own_records() -> None:
    """不是"少了几条"而是"换了候选集"：ep-B 那一侧必须只出 ep-B。"""

    module = _capture()
    store = module._build_store()
    cue = torch.tensor(module.QUERY_TABLE["q_axis_y"], dtype=torch.float32)
    scoped = _hits(store, cue, episode_id="ep-B")
    assert scoped, scoped
    assert set(scoped) <= {memory_id for memory_id, ep in module.EPISODE_OF.items() if ep == "ep-B"}


def test_unscoped_call_is_byte_identical_to_scoping_to_all() -> None:
    """默认位等价于"不筛"，而不是"沿用当前会话"——这两件事在批文里不是同一件。"""

    module = _capture()
    store = module._build_store()
    for name, values in module.QUERY_TABLE.items():
        cue = torch.tensor(values, dtype=torch.float32)
        assert _hits(store, cue) == _hits(store, cue, episode_id=None), name


def test_unknown_episode_scope_returns_empty_not_an_error() -> None:
    """筛到一个不存在的会话应当是空命中；把"没查到"变成异常会改变默认链的错误面。"""

    module = _capture()
    store = module._build_store()
    cue = torch.tensor(module.QUERY_TABLE["q_near_m0"], dtype=torch.float32)
    assert _hits(store, cue, episode_id="ep-does-not-exist") == []


def test_limit_and_scoping_compose_without_moving_the_order() -> None:
    """`limit` 仍作用在**筛后的候选集**上，且相对次序不因筛选而变。"""

    module = _capture()
    store = module._build_store()
    cue = torch.tensor(module.QUERY_TABLE["q_near_m0"], dtype=torch.float32)
    full = _hits(store, cue, episode_id="ep-A")
    one = [hit.record.memory_id for hit in store.retrieve(cue, limit=1, episode_id="ep-A")]
    assert one == full[:1], (one, full)


def test_the_other_retrieve_do_not_gain_the_parameter() -> None:
    """`G-N6b-2`：本仓有同名不同类的 `retrieve`（现读 `taiji/concept_formation.py:269`；
    `taiji/adapter.py:10845`／`:10908` 是它的**调用点**不是定义）——分派必须按**类**，不许按名字。

    这一支同时钉住"我没有顺手把会话参数塞进别人的同名方法"：那会把一次改动扩成两次。
    """

    import taiji.concept_formation as concept_formation
    import taiji.episodic_memory as episodic_memory

    found: dict[str, list[str]] = {}
    for holder in (episodic_memory, concept_formation):
        for attribute in vars(holder).values():
            if isinstance(attribute, type) and "retrieve" in attribute.__dict__:
                key = f"{holder.__name__}.{attribute.__name__}"
                found[key] = list(inspect.signature(attribute.retrieve).parameters)
    target = "taiji.episodic_memory.EpisodicMemoryStore"
    assert target in found, found
    #: 对照必须**存在**，否则这一支就成了"只有目标、无人可证"的恒真式。
    others = {key: names for key, names in found.items() if key != target}
    assert others, found
    assert all("episode_id" in names for names in [found[target]]), found[target]
    for key, names in others.items():
        assert "episode_id" not in names, (key, names)
