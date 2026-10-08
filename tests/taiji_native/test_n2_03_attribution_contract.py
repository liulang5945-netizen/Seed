"""PLAN-N2-03 归因判读器（`count_taiji_n2_03_attribution.py`）的契约测。

这台仪器只出一格归因，所以它的全部风险都在于**把缺件/换尺读成一个归因结论**。
正反两支都实走：

* 三档判级各走一次（`cost_from_wake_reset` / `partially_resolved` / `cost_from_weight_update`），
  数字全由夹具喂出，不碰在库件；
* 五条拒判支各走一次（缺件／臂不干净／面被装载失败污染／CAP-0 题集不同源／复述面不同源／主列题集 sha 不同源或缺），
  每条都要 rc=2 且**没有异常**——拒绝路径本身再炸一次就是把响亮拒绝伪装成崩溃；
* 分母为 0 的列必须**被点名剔除并缩小列数**，不许静默当 0 归因；
* 主列（§2 末句不入归因分数）即便在臂档上好到 0 枚，也不许挪动判级——这条防的是"旁证偷偷参与判级"。

真实在库件只读，被改的副本一律落 pytest 的 `tmp_path`。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/count_taiji_n2_03_attribution.py"

ITEMS_SHA = "8b974e9b62d8e4ca"
PENALTIES = (0.0, 0.5, 1.0, 2.0)


def _load_module():
    spec = importlib.util.spec_from_file_location("n2_03_attribution_reader", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


READER = _load_module()


def _cap0_face(correct_e: int, eval_set: str = "evalset-v3", load_ok: bool = True) -> dict:
    dims: dict[str, Any] = {}
    for key, correct in (
        ("B", 18),
        ("C", 17),
        ("D", 15),
        ("E", correct_e),
        ("G", 16),
    ):
        dims[key] = {
            "tally": {
                "machine_scored_correct": correct,
                "machine_scored_items": 20,
                "pending_human_review_items": 0,
            },
            "item_count": 20,
            "items": [{"load_ok": load_ok}] * 20,
        }
    return {"eval_set": eval_set, "checkpoint": "x.pt", "dimensions": dims}


def _replay_face(
    strict: dict[str, int],
    well: dict[str, int],
    items: int = 24,
    offset: int = 0,
    first: str = "V001",
) -> dict:
    return {
        "items": items,
        "item_offset": offset,
        "first_item": first,
        "arms": [
            {
                "repetition_penalty": p,
                "items": items,
                "strict_hits": strict[str(p)],
                "well_formed_texts": well[str(p)],
            }
            for p in PENALTIES
        ],
    }


def _main_face(count: int, items_sha: str | None = ITEMS_SHA) -> dict:
    entry: dict[str, Any] = {
        "report": "some_stop_face.json",
        "never_lf_eaters_counted": count,
        "generation_rows_seen": 72,
        "checkpoint_sha256": "504f7c342ad0f942",
        "status": "ok",
        "meets_frozen_line": False,
    }
    if items_sha is not None:
        entry["items_sha256"] = items_sha
    return {"faces": [entry]}


def _arm_report(
    family_clean: bool = True,
    digest_clean: bool = True,
    loadable: bool = True,
) -> dict:
    return {
        "wake_episode_id": "wake-after-sleep",
        "g_n6_1_parameter_family_unchanged": digest_clean,
        "g_n6_2_arm_loadable_and_params_identical": loadable,
        "family_diff": {
            "parameter_family": {
                "changed": 0 if family_clean else 3,
                "dropped": 0 if family_clean else 1,
                "added": 0,
            }
        },
        "params_digest_before": "a" * 16,
        "params_digest_after": "a" * 16,
        "excluded_changed_paths": [],
        "mother_tick": 273,
        "tick_after_reset": 0,
    }


#: 巩固前基线（母档面）：E 维 1，复述命中/成句全为高位。
BEFORE_STRICT = {"0.0": 12, "0.5": 11, "1.0": 10, "2.0": 9}
BEFORE_WELL = {"0.0": 20, "0.5": 19, "1.0": 18, "2.0": 17}
#: run-2 治疗后：七列全部跌破（E 1→0，命中两档各跌 3，成句四档各跌 5）。
AFTER_STRICT = {"0.0": 12, "0.5": 11, "1.0": 7, "2.0": 6}
AFTER_WELL = {"0.0": 15, "0.5": 14, "1.0": 13, "2.0": 12}


def _faces(
    tmp: Path,
    *,
    arm_e: int,
    arm_strict: dict[str, int],
    arm_well: dict[str, int],
    cap0_eval_set: str = "evalset-v3",
    load_ok: bool = True,
    replay_items: int = 24,
    main_sha: str | None = ITEMS_SHA,
    main_count: int = 47,
    arm: dict | None = None,
) -> dict[str, str]:
    """把七枚面写成件并返回**旗标 → 绝对路径**（判读器按仓根拼路径，绝对路径原样透传）。"""

    payloads = {
        "--arm-report": arm if arm is not None else _arm_report(),
        "--cap0-before": _cap0_face(1),
        "--cap0-after": _cap0_face(0),
        "--cap0-arm": _cap0_face(arm_e, eval_set=cap0_eval_set, load_ok=load_ok),
        "--replay-before": _replay_face(BEFORE_STRICT, BEFORE_WELL, items=replay_items),
        "--replay-after": _replay_face(AFTER_STRICT, AFTER_WELL, items=replay_items),
        "--replay-arm": _replay_face(arm_strict, arm_well, items=replay_items),
        "--main-column": _main_face(main_count, main_sha),
        "--main-column-baseline": _main_face(56),
    }
    paths: dict[str, str] = {}
    for key, payload in payloads.items():
        path = tmp / f"{key.strip('-')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        paths[key] = str(path)
    return paths


def _run(paths: dict[str, str], tmp: Path) -> tuple[int, dict[str, Any]]:
    out = tmp / "verdict.json"
    argv: list[str] = []
    for key, value in paths.items():
        argv += [key, value]
    rc = READER.main(argv + ["--out", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def _full_reproduce() -> tuple[dict[str, int], dict[str, int]]:
    """臂档＝把 run-2 的跌破原样复现（七列 attrib_R 全 1）。"""

    return copy.deepcopy(AFTER_STRICT), copy.deepcopy(AFTER_WELL)


def test_verdict_cost_from_wake_reset(tmp_path: Path) -> None:
    strict, well = _full_reproduce()
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well)
    rc, payload = _run(paths, tmp_path)
    assert rc == 0
    assert payload["verdict"] == "cost_from_wake_reset"
    assert payload["criteria"]["n_reproduced"] == 7
    assert payload["criteria"]["columns_considered"] == 7
    assert payload["criteria"]["columns_excluded"] == []
    assert all(row["attrib_R"] == 1.0 for row in payload["per_column"].values())


def test_verdict_partially_resolved_is_the_named_middle_band(tmp_path: Path) -> None:
    #: 只复现三列（E 维＋成句两档），其余四列停在巩固前 ⇒ attrib 0 ⇒ 中间带有自己的名字。
    strict = {"0.0": 12, "0.5": 11, "1.0": 10, "2.0": 9}
    well = {"0.0": 15, "0.5": 14, "1.0": 18, "2.0": 17}
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well)
    rc, payload = _run(paths, tmp_path)
    assert rc == 0
    assert payload["verdict"] == "partially_resolved"
    assert payload["criteria"]["n_reproduced"] == 3


def test_verdict_cost_from_weight_update(tmp_path: Path) -> None:
    #: 臂档七列全停在巩固前 ⇒ 收束没复现任何一列 ⇒ 代价归权重更新。
    paths = _faces(
        tmp_path,
        arm_e=1,
        arm_strict=copy.deepcopy(BEFORE_STRICT),
        arm_well=copy.deepcopy(BEFORE_WELL),
    )
    rc, payload = _run(paths, tmp_path)
    assert rc == 0
    assert payload["verdict"] == "cost_from_weight_update"
    assert payload["criteria"]["n_reproduced"] == 0
    assert all(row["attrib_R"] == 0.0 for row in payload["per_column"].values())


def test_arm_reproduces_below_the_line_is_not_counted(tmp_path: Path) -> None:
    #: 臂只把七列各往跌破方向挪一点点（命中 1/3、成句 2/5、E 维 0）⇒ attrib 全在 0.5 线下 ⇒ 一列都不算。
    strict = {"0.0": 12, "0.5": 11, "1.0": 9, "2.0": 8}
    well = {"0.0": 18, "0.5": 17, "1.0": 16, "2.0": 15}
    paths = _faces(tmp_path, arm_e=1, arm_strict=strict, arm_well=well)
    rc, payload = _run(paths, tmp_path)
    assert rc == 0
    assert payload["criteria"]["n_reproduced"] == 0
    assert payload["verdict"] == "cost_from_weight_update"


def test_missing_face_refuses_and_names_it(tmp_path: Path) -> None:
    strict, well = _full_reproduce()
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well)
    Path(paths["--replay-arm"]).unlink()
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["verdict"] == "not_judged_missing_faces"
    assert payload["missing"] == [paths["--replay-arm"]]
    assert "per_column" not in payload


def test_dirty_arm_refuses_whole_experiment(tmp_path: Path) -> None:
    #: 臂的参数族一旦有差（changed=3），这一臂就不是"只收束"，单变量分离不成立 ⇒ 整件不判。
    strict, well = _full_reproduce()
    paths = _faces(
        tmp_path,
        arm_e=0,
        arm_strict=strict,
        arm_well=well,
        arm=_arm_report(family_clean=False),
    )
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["verdict"] == "not_judged_arm_not_clean"
    assert payload["arm_guards"]["g_n6_1b_parameter_entries_stable"] is False


def test_arm_digest_or_load_guard_refuses(tmp_path: Path) -> None:
    strict, well = _full_reproduce()
    paths = _faces(
        tmp_path,
        arm_e=0,
        arm_strict=strict,
        arm_well=well,
        arm=_arm_report(digest_clean=False),
    )
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["verdict"] == "not_judged_arm_not_clean"
    assert payload["arm_guards"]["g_n6_1_parameter_family_unchanged"] is False
    assert "per_column" not in payload


def test_unanswered_items_refuse_the_face(tmp_path: Path) -> None:
    #: 臂面里有项 `load_ok` 不为真 ⇒ 那一项根本没被答，"能力＝0"是仪器缺陷不是读数。
    strict, well = _full_reproduce()
    paths = _faces(
        tmp_path,
        arm_e=0,
        arm_strict=strict,
        arm_well=well,
        cap0_eval_set="evalset-v3",
        load_ok=False,
    )
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["verdict"] == "not_judged_face_invalid"
    assert payload["cap0_load_failures_by_face"]["--cap0-arm"] == 100


def test_cap0_eval_set_mismatch_refuses(tmp_path: Path) -> None:
    strict, well = _full_reproduce()
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well, cap0_eval_set="other-set")
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["verdict"] == "not_judged_faces_not_comparable"
    assert payload["mismatch"]["cap0_eval_set"]["--cap0-arm"] == "other-set"


def test_replay_face_offset_mismatch_refuses(tmp_path: Path) -> None:
    #: 臂的复述面换了行数或起点＝换尺；这台仪器只认面自述的 items/item_offset/first_item。
    strict, well = _full_reproduce()
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well)
    arm_replay = json.loads(Path(paths["--replay-arm"]).read_text(encoding="utf-8"))
    arm_replay["item_offset"] = 24
    Path(paths["--replay-arm"]).write_text(json.dumps(arm_replay) + "\n", encoding="utf-8")
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["verdict"] == "not_judged_faces_not_comparable"
    assert "replay_identity" in payload["mismatch"]


def test_main_column_missing_items_sha_refuses(tmp_path: Path) -> None:
    #: "两边都没记 sha"不许被读成"sha 相同"——守卫必须能为假。
    strict, well = _full_reproduce()
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well, main_sha=None)
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["verdict"] == "not_judged_faces_not_comparable"
    blocked = payload["mismatch"]["main_column_items_sha256"]
    assert blocked["arm"] == ["None"]
    assert blocked["baseline"] == [ITEMS_SHA]


def test_main_column_different_set_refuses(tmp_path: Path) -> None:
    strict, well = _full_reproduce()
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well, main_sha="f" * 16)
    rc, payload = _run(paths, tmp_path)
    assert rc == 2
    assert payload["mismatch"]["main_column_items_sha256"]["arm"] == ["f" * 16]


def test_main_column_is_corroboration_only(tmp_path: Path) -> None:
    #: 臂的主列好到 0 枚（比冻结线还好）也不许挪动判级：七列归"权重更新"时它必须还是那个判级。
    paths = _faces(
        tmp_path,
        arm_e=1,
        arm_strict=copy.deepcopy(BEFORE_STRICT),
        arm_well=copy.deepcopy(BEFORE_WELL),
        main_count=0,
    )
    rc, payload = _run(paths, tmp_path)
    assert rc == 0
    assert payload["verdict"] == "cost_from_weight_update"
    assert payload["criteria"]["n_reproduced"] == 0
    side = payload["main_column_corroboration_not_judged"]
    assert side["arm"]["some_stop_face.json"]["never_lf_eaters_counted"] == 0
    assert side["baseline"]["some_stop_face.json"]["never_lf_eaters_counted"] == 56


def test_zero_denominator_column_is_named_and_dropped_from_the_count(tmp_path: Path) -> None:
    #: 把臂面的 E 维做成"与巩固前同"但让巩固后的 E 与巩固前相等（分母 0）⇒ 这列必须被点名剔除。
    strict, well = _full_reproduce()
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well)
    cap0_after = json.loads(Path(paths["--cap0-after"]).read_text(encoding="utf-8"))
    cap0_after["dimensions"]["E"]["tally"]["machine_scored_correct"] = 1
    Path(paths["--cap0-after"]).write_text(json.dumps(cap0_after) + "\n", encoding="utf-8")
    rc, payload = _run(paths, tmp_path)
    assert rc == 0
    assert payload["criteria"]["columns_excluded"] == ["cap0:E"]
    assert payload["criteria"]["columns_considered"] == 6
    assert "cap0:E" not in payload["per_column"]
    #: 其余六列仍复现 ⇒ 判级不变，但分母列数如实缩到 6。
    assert payload["criteria"]["n_reproduced"] == 6
    assert payload["verdict"] == "cost_from_wake_reset"


def test_overshoot_past_the_drop_still_counts(tmp_path: Path) -> None:
    #: 臂把列推到**比 run-2 还低**（attrib>1）仍按预注册的 0.5 线记为复现——这条钉的是判级线
    #: 只问"到没到 0.5"、不问"是否恰好等于 run-2"，防止实现里偷偷加一条上界把复现判没。
    strict = {"0.0": 12, "0.5": 11, "1.0": 3, "2.0": 2}
    well = {"0.0": 5, "0.5": 4, "1.0": 3, "2.0": 2}
    paths = _faces(tmp_path, arm_e=0, arm_strict=strict, arm_well=well)
    rc, payload = _run(paths, tmp_path)
    assert rc == 0
    assert payload["per_column"]["replay_strict_hits:1.0"]["attrib_R"] > 1.0
    assert payload["criteria"]["n_reproduced"] == 7
