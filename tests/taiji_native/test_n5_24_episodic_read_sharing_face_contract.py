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
    assert report["verdicts"]["retrieve_signature_has_no_episode_param"] is True
    assert report["limits"]
    assert "authoritative_enumeration" in report["coverage_gap"]
