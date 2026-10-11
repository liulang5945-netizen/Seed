"""`audit_taiji_episodic_read_sharing_face.py` 的契约册：分派靠接收者类型，两支都要能为假。

* 不许把概念器官的 `retrieve` 数进 E 存储的分子（`taiji/adapter.py:10845`／`:10908` 是
  `_concept_formation`，不是 `EpisodicMemoryStore`）；
* 零命中必须**响亮失败**，不许读成「没人跨会话读」；
* 合成材料那条支正反都要测到（有 `torch.eye` 的必须被抓、用真语料变量的不许被硬算成合成）。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_episodic_read_sharing_face.py"


def _load():
    spec = importlib.util.spec_from_file_location("n5_24_episodic_read_face", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "taiji").mkdir(parents=True, exist_ok=True)
    (tmp_path / "taiji" / "episodic_memory.py").write_text(
        "class EpisodicMemoryStore:\n"
        "    def retrieve(self, cue, *, limit=1):\n        return ()\n"
        "    def write(self, record):\n        return None\n",
        encoding="utf-8",
    )
    return tmp_path


def test_concept_organ_retrieve_is_not_counted_as_store_read(tmp_path: Path) -> None:
    module = _load()
    root = _fixture_root(tmp_path)
    (root / "mix.py").write_text(
        "class Holder:\n"
        "    def __init__(self):\n"
        "        self._episodic_memory = None\n"
        "        self._concept_formation = None\n"
        "    def step(self, features):\n"
        "        a = self._concept_formation.retrieve(features, limit=2)\n"
        "        b = self._episodic_memory.retrieve(features, limit=3)\n"
        "        return a, b\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], capture_output=True, check=True)
    reads = module.scan(root)["read_sites"]
    assert [row["line"] for row in reads] == [7], reads
    assert reads[0]["episode_id_mentioning"] is False


def test_local_store_binding_is_found_and_episode_mention_is_a_weak_signal(tmp_path: Path) -> None:
    module = _load()
    root = _fixture_root(tmp_path)
    (root / "local.py").write_text(
        "def run(store, episode_id):\n" "    return store.retrieve(store.cue, limit=1)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], capture_output=True, check=True)
    reads = module.scan(root)["read_sites"]
    assert reads == []  # 裸参数名 store 没有绑定成 EpisodicMemoryStore，静态读认不出＝已知盲区
    (root / "bound.py").write_text(
        "def run(episode_id):\n"
        "    store = EpisodicMemoryStore(capacity=4)\n"
        "    return store.retrieve(store.cue, limit=1)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(root), "add", "bound.py"], capture_output=True, check=True)
    reads = module.scan(root)["read_sites"]
    assert len(reads) == 1
    assert reads[0]["episode_id_mentioning"] is True


def test_zero_read_hits_fail_closed_instead_of_reading_as_no_callers(tmp_path: Path) -> None:
    module = _load()
    root = _fixture_root(tmp_path)
    (root / "quiet.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], capture_output=True, check=True)
    try:
        module.build_report(root)
    except ValueError as exc:
        assert "零命中" in str(exc)
    else:
        raise AssertionError("零命中必须响亮失败，不许落半件")


def test_synthetic_marker_branch_both_ways(tmp_path: Path) -> None:
    module = _load()
    root = _fixture_root(tmp_path)
    (root / "syn.py").write_text(
        "import torch\n"
        "def build():\n"
        "    store = EpisodicMemoryStore(capacity=4)\n"
        "    record = EpisodicMemoryRecord(cue=torch.eye(4)[0], memory_id='m')\n"
        "    store.write(record)\n"
        "    return store\n",
        encoding="utf-8",
    )
    (root / "real.py").write_text(
        "def build(corpus_cues):\n"
        "    store = EpisodicMemoryStore(capacity=4)\n"
        "    record = EpisodicMemoryRecord(cue=corpus_cues[0], memory_id='m')\n"
        "    store.write(record)\n"
        "    return store\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], capture_output=True, check=True)
    face = module.scan(root)
    files = {row["file"] for row in face["synthetic_records"]}
    assert files == {"syn.py"}, files
    writes = {row["file"] for row in face["write_sites"]}
    assert {"syn.py", "real.py"} <= writes


def test_real_repository_face_is_nonempty_and_gate_verdict_is_signature_derived() -> None:
    module = _load()
    report = module.build_report(PROJECT_ROOT)
    assert report["files_parsed"] > 1000
    assert report["read_site_count"] > 0
    assert report["verdicts"]["someone_in_product_chain_reads_the_store"] is True
    verdicts = report["verdicts"]
    assert verdicts["retrieve_signature_has_episode_param"] is True, verdicts
    assert verdicts["retrieve_episode_param_defaults_to_none"] is True, verdicts
    #: `J-N6b-3`：有口可筛**且**默认未筛 ⇒ 会话闸在默认位上仍不成立，这条不许被翻绿。
    assert verdicts["session_gate_enforced_by_default"] is False, verdicts
    assert report["limits"]
    assert "authoritative_enumeration" in report["coverage_gap"]


def test_signature_gate_face_needs_three_separate_readings(tmp_path: Path) -> None:
    """三档签名要出三组不同的三元组。

    改前只有一枚布尔（"签名里没有 episode 参数"），于是"加了口但默认不筛"与
    "默认就筛"会塌成同一个 False——而 `J-N6b-3` 要钉的恰是这两件事不一样。
    """

    module = _load()
    cases = {
        "(self, cue, *, limit=1)": (False, False, False),
        "(self, cue, *, limit=1, episode_id=None)": (True, True, False),
        "(self, cue, *, limit=1, episode_id='ep-A')": (True, False, True),
    }
    for index, (signature, expected) in enumerate(cases.items()):
        root = tmp_path / f"case{index}"
        (root / "taiji").mkdir(parents=True)
        source = (
            "class EpisodicMemoryStore:\n" "    def retrieve" + signature + ":\n        return ()\n"
        )
        (root / "taiji" / "episodic_memory.py").write_text(source, encoding="utf-8", newline="\n")
        face = module._retrieve_session_gate_face(root)
        got = (
            face["retrieve_signature_has_episode_param"],
            face["retrieve_episode_param_defaults_to_none"],
            face["session_gate_enforced_by_default"],
        )
        assert got == expected, (signature, got, expected)


def test_missing_retrieve_definition_fails_closed(tmp_path: Path) -> None:
    """判据对象不在盘上时必须响亮失败，而不是把三枚读数默认成 False（那看起来像"没有会话闸"）。"""

    module = _load()
    root = tmp_path / "empty"
    (root / "taiji").mkdir(parents=True)
    (root / "taiji" / "episodic_memory.py").write_text(
        "class EpisodicMemoryStore:\n    pass\n", encoding="utf-8", newline="\n"
    )
    try:
        module._retrieve_session_gate_face(root)
    except ValueError as exc:
        assert "判据对象不在盘上" in str(exc), exc
    else:
        raise AssertionError("找不到 retrieve 定义必须拒绝，不许默认出版三枚 False")
