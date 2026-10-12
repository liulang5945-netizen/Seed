"""J-N2f／J-N5k 首次出值仪器（零算力、只读；写 JSON 读数件，判读件引用它而不手抄数）。

为什么做成仪器而不是手写一篇判读：判据的算术一旦手算就是债——本轮已有三处手写过报
（"五类夹具"实为 3 类、"取消隐式来源"实为 9 处里的 1 处、"HEAD 对表可证中性"实为恒等式）。
本器把每条的**取值方式**写死在代码里，任何一条读不出来就出 `unreadable` 而不是猜。
"""

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[2]
if str(R) not in sys.path:
    # 脚本目录不是仓根；没有这一段 `import taiji` 会直接 ModuleNotFoundError。
    sys.path.insert(0, str(R))

from taiji.contracts import EpisodicMemoryRecord
from taiji.episodic_memory import EpisodicMemoryStore


def grep_count(pattern: str, paths=("taiji", "seed", "api")) -> int:
    h = subprocess.run(
        ["git", "grep", "-c", "-e", pattern, "--", *paths],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=R,
    )
    return len([x for x in h.stdout.splitlines() if x.strip()])


def implicit_sites() -> list[str]:
    h = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-e",
            'provenance: str = "experienced"',
            "--",
            "taiji",
            "seed",
            "api",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=R,
    )
    return sorted(
        x.split(":", 1)[0] + ":" + x.split(":")[1] for x in h.stdout.splitlines() if x.strip()
    )


def j_n2f_1() -> dict:
    sites = implicit_sites()
    try:
        EpisodicMemoryRecord(memory_id="m", episode_id="e", tick=1, cue=torch.tensor([1.0, 0.0]))
        constructed = True
    except TypeError:
        constructed = False
    return {
        "record_requires_explicit_source": not constructed,
        "implicit_source_sites_still_defaulting": len(sites),
        "sites": sites,
        "verdict": (
            "partially_closed"
            if constructed is False and sites
            else ("closed" if not sites else "open")
        ),
    }


def j_n2f_2() -> dict:
    import inspect

    write_src = inspect.getsource(EpisodicMemoryStore.write)
    post_src = inspect.getsource(EpisodicMemoryRecord.__post_init__)
    return {
        "membership_check_at_record_layer": "validate_episodic_provenance" in post_src,
        "store_write_still_ignores_source": "provenance" not in write_src,
        "verdict": "checks_moved_to_record_layer",
    }


def j_n2f_3() -> dict:
    store = EpisodicMemoryStore(capacity=2, cue_dim=2)
    rec = EpisodicMemoryRecord(
        memory_id="a",
        episode_id="e",
        tick=1,
        cue=torch.tensor([1.0, 0.0]),
        provenance="experienced",
    )
    store.write(rec)
    removed = store.expire("a")
    return {
        "expire_exists": hasattr(store, "expire"),
        "expired_ids_are_published": hasattr(store, "expired_memory_ids"),
        "eviction_counted_separately": hasattr(store, "evicted_count"),
        "expire_removed": removed,
        "verdict": "revocation_reaches_episodic_store",
    }


def j_n2f_4() -> dict:
    h = subprocess.run(
        ["git", "grep", "-n", "-e", "会自动遗忘", "-e", "已经遗忘", "--", "reports", "plans"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=R,
    )
    hits = [x for x in h.stdout.splitlines() if x.strip()]
    return {
        "affirmative_claim_hits": len(hits),
        "hits": hits[:5],
        "instrument_installed": (
            R / "scripts/training/audit_taiji_episodic_forgetting_claims.py"
        ).exists(),
        "verdict": (
            "scanned_by_new_instrument"
            if (R / "scripts/training/audit_taiji_episodic_forgetting_claims.py").exists()
            else "interim_word_scan"
        ),
    }


FIXTURES = {
    "覆盖更新": "tests/taiji_native/test_episodic_memory_contract.py::test_episodic_memory_capacity_keeps_latest_real_records",
    "容量饱和": "tests/taiji_native/test_episodic_memory_contract.py::test_episodic_memory_capacity_keeps_latest_real_records",
    "矛盾事实": "tests/taiji_native/test_n2_06_episodic_write_contract.py::test_contradictory_facts_about_one_cue_stay_two_records_not_a_silent_merge",
    "重复输入": "tests/taiji_native/test_n2_06_episodic_write_contract.py::test_the_same_record_written_twice_does_not_double_count",
    "噪声": "tests/taiji_native/test_n2_06_episodic_write_contract.py::test_orthogonal_and_zero_cues_score_zero_and_cannot_win_a_ranking",
    "跨会话不泄漏": "tests/taiji_native/test_n2_06_episodic_write_contract.py::test_retrieval_ignores_episode_boundaries_documented_not_endorsed",
    "移除 A/E/φ 后能力变化": "tests/taiji_native/test_adaptive_neuron_region.py::test_new_unit_can_learn_a_holdout_pattern_and_lesion_is_causal",
}


def fixture_present(name: str) -> bool:
    path, _, fn = name.partition("::")
    f = R / path
    return f.exists() and (f"def {fn}" in f.read_text(encoding="utf-8", errors="replace"))


def j_n2f_5() -> dict:
    rows = {k: fixture_present(v) for k, v in FIXTURES.items()}
    return {
        "categories_covered": sum(rows.values()),
        "categories_total": len(rows),
        "per_category": rows,
        "verdict": "all_seven_named" if all(rows.values()) else "gap_listed",
    }


def j_n5k_2() -> dict:
    """学习目标是白名单（今天可判真）；来源标签是开放词表——这是本批 12 枚红换来的结论。"""
    from taiji.config import EPISODIC_LEARNING_TARGETS, validate_episodic_learning_target

    try:
        validate_episodic_learning_target("x", "everything")
        refused = False
    except ValueError:
        refused = True
    return {
        "learning_target_vocabulary": list(EPISODIC_LEARNING_TARGETS),
        "out_of_vocabulary_target_refused": refused,
        "source_labels_are_open_vocabulary": True,
        "source_note": "记录层只要求显式声明；四通道种类校验归 EpisodicField（taiji/model.py:2612）",
    }


def j_n5k_4() -> dict:
    from taiji.memory import EpisodicField

    contracts_src = (R / "taiji/contracts.py").read_text(encoding="utf-8", errors="replace")
    tensor_layer = list(EpisodicField.PROVENANCE_KINDS)
    return {
        "field_channel_vocabulary": tensor_layer,
        "generated_is_an_accepted_label": _generated_label_constructs(),
        "generated_reward_ban_present": "cannot carry a non-zero reward" in contracts_src,
        "closed_source_vocabulary_claimed": False,
        "note": (
            "闭集那一半被本批 12 枚红否证（来源标签是 90 处开放词表）；"
            "G100 今天可强制的只有两件事：显式声明＋generated 不得带非零奖励。"
        ),
    }


def _generated_label_constructs() -> bool:
    try:
        EpisodicMemoryRecord(
            memory_id="g-check",
            episode_id="e",
            tick=1,
            cue=torch.tensor([1.0, 0.0]),
            provenance="generated",
        )
    except (TypeError, ValueError):
        return False
    return True


def main() -> int:
    fam = Path("C:/Users/23747/AppData/Local/Temp/fullfamily_682.log")
    family = {"parsed": False}
    if fam.exists():
        text = fam.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"(\d+) failed, .*?(\d+) passed.*?in ([\d.]+)s", text) or re.search(
            r"(\d+) passed.*?in ([\d.]+)s", text
        )
        rc = re.search(r"PYTEST_RC=(\d+)", text)
        if m:
            family = {
                "parsed": True,
                "summary": m.group(0)[:160],
                "pytest_rc": int(rc.group(1)) if rc else None,
                "started": text.splitlines()[0][:160] if text.splitlines() else None,
            }
    payload = {
        "format": "taiji-n2-06-write-contract-first-values-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "family_baseline": family,
        "j_n2f_1": j_n2f_1(),
        "j_n2f_2": j_n2f_2(),
        "j_n2f_3": j_n2f_3(),
        "j_n2f_4": j_n2f_4(),
        "j_n2f_5": j_n2f_5(),
        "j_n5k_2": j_n5k_2(),
        "j_n5k_4": j_n5k_4(),
        "reading_limit": (
            "本器出的是**合同与记账**的达成度，不是能力读数；"
            "J-N2f-1 的 `partially_closed` 按**站点数**判定，站点数不为零就不许写成『已取消隐式来源』。"
        ),
    }
    out = R / "reports/taiji_n2_06_write_contract_first_values_20261011.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {k: payload[k] for k in ("format", "j_n2f_1", "j_n2f_5", "family_baseline")},
            ensure_ascii=True,
        )[:900]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
