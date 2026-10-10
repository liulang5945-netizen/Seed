"""N5 影子通电验收判读器（PLAN-N5-02 的 J-N5b-1／J-N5b-2／J-N5b-3 前半，机械执行、不手算）。

为什么要有这台仪器：㊵-597 冻的判据里，「在场性」「学习是否发生」「同批配对能否成立」三支可以纯机械判，
而本仓已两次栽在「缺列被读成满足」与「手对齐两个均值」（㊵-590 的 `get()` 假 null、`J-N3a′` 的手算末段）。
所以这三支由脚本出数，人只抄它的输出。

**本器不实现 J-N5b-4 的收益比较**：母量口径（`online_accuracy` 末两段段均值、自取噪声带、
`wrong_top1_rate`）在 N3／N4 的专门仪器里已有冻结算法，在这里重抄一份就是第二条链
（本仓纪律：别在新仪器里重抄生成链）。〔原文写于㊵-597：「本器因此把 `j_n5b_4`／`j_n5b_5`／`j_n5b_6`
一律记成 `unverified_*`／`not_adjudicable_*`，让合取条件读起来必然是「未判」而不是「成立」」——
该句已在 **㊵-660／㊵-663／㊵-665** 三笔里逐格作废：4 的算术仍不在本器内做，只从
`--n3a-pair-file`（N3 那台仪器出的段均值）做三步算术；5 自 ㊵-663 起按 PLAN-N5-08 §2 的取数式出值；
6 自本笔起按 PLAN-N5-09 §3 的三形出值。**不重抄生成链**这条约束本身一字未松。〕

用法：

    python scripts/training/adjudicate_taiji_n5_shadow_gate.py --arm-treated <ckpt.pt> \
        [--arm-control <ckpt.pt>] [--face-treated <pressure.jsonl>] [--face-control <...>] \
        --out <json>

判据映射（全部 fail-closed）：

* **J-N5b-1 自述在场性**：`n5_shadow` 块与五枚键必须齐（用 `key in env` 断言，不用 `get`）；
  缺任一 ⇒ 该臂 `ran_not_measured`，`rc=2`。
* **J-N5b-2 学习确曾发生**：`gate > 0` 且两枚效用不全为 0 ⇒ `shadow_learned`；
  `gate > 0` 而两枚效用同为 0 ⇒ `shadow_inert`（`rc=1`）；`gate == 0` ⇒ `not_powered`（`rc=1`）。
* **J-N5b-3 同批配对**：两臂 `shadow_gate_requested` 必须不同（相同 ⇒ `pairing_invalid`）；
  给了两枚压强面时再核对四元组（`seed`／`policy.minimum_pressure`／`readout`／
  `developmental.fast_slow_requested`／`developmental.bridge_gate_actual`），
  任一不同 ⇒ `pairing_invalid` 并点名是哪一维；全同 ⇒ `pairing_valid_on_quadruple`。
  只给一臂 ⇒ `unverified_no_control_arm`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FORMAT = "taiji-n5-shadow-gate-adjudication-v1"
REQUIRED_KEYS = (
    "gate",
    "shadow_gate_requested",
    "candidate_gate",
    "candidate_utility",
    "candidate_counterfactual_utility",
)

#: ㊵-621：过程侧在场计数器（㊵-618 落盘）。它们是**披露**，不是 J-N5b-1 的判据成员——
#: 把五枚冻结键扩成八枚会把已入库的 G/H 读数追认成 `ran_not_measured`，那是改判据而不是加读数。
PRESENCE_KEYS = ("shadow_forward_hits", "shadow_learn_hits", "shadow_branch_hits")
#: 四元组（实际五维）逐条从**面头**取，不从命令行取：臂间差异只能来自现场，不能来自我的转述。
QUADRUPLE = (
    ("seed", ("seed",)),
    ("tau", ("policy", "minimum_pressure")),
    ("readout", ("readout",)),
    ("fast_slow", ("developmental", "fast_slow_requested")),
    ("bridge_gate", ("developmental", "bridge_gate_actual")),
)


def _shadow_block(checkpoint: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """返回 (n5_shadow 块, 缺失键列表)。块不在场时缺失键＝全部五枚。"""
    import torch  # 局部导入：只有真要读检查点时才拉起 torch

    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    env = loaded.get("envelope", loaded) if isinstance(loaded, dict) else {}
    block = env.get("n5_shadow") if isinstance(env, dict) else None
    if not isinstance(block, dict):
        return None, list(REQUIRED_KEYS)
    return block, [key for key in REQUIRED_KEYS if key not in block]


def _dig(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def judge_arm(name: str, checkpoint: Path) -> dict[str, Any]:
    block, missing = _shadow_block(checkpoint)
    out: dict[str, Any] = {"arm": name, "checkpoint": str(checkpoint)}
    if missing:
        out["j_n5b_1"] = "ran_not_measured"
        out["missing_self_report_keys"] = missing
        out["j_n5b_2"] = "unverified_missing_face"
        return out
    out["j_n5b_1"] = "present"
    out["missing_self_report_keys"] = []
    for key in REQUIRED_KEYS:
        #: 在场性（J-N5b-1）与值判断（J-N5b-2）分开： 是**在场但无请求值**——
        #: 对照臂没给 --n5-shadow-gate 时  就是 None（真跑逼出来的形状，
        #: 合成夹具当初全用数值，漏了这一支）。
        value = block[key]
        out[key] = None if value is None else float(value)
    out["candidate_id"] = str(block.get("candidate_id", ""))
    #: ㊵-621：在场性仍按 `'k' in block` 判、取值按现读，缺任一枚就整组标 `absent_from_block`
    #: （不许把"没写计数器"读成"被喂了 0 次"——那是 ㊵-590 那族 `get()` 假缺席的再命中）。
    presence = {key: block[key] for key in PRESENCE_KEYS if key in block}
    out["presence_counters"] = {
        "status": "present" if len(presence) == len(PRESENCE_KEYS) else "absent_from_block",
        "missing": [key for key in PRESENCE_KEYS if key not in block],
        "values": {key: int(presence[key]) for key in PRESENCE_KEYS if key in presence},
    }
    powered = (out["gate"] or 0.0) > 0.0
    nonzero_utility = (out["candidate_utility"] or 0.0) != 0.0 or (
        out["candidate_counterfactual_utility"] or 0.0
    ) != 0.0
    if not powered:
        out["j_n5b_2"] = "not_powered"
    elif nonzero_utility:
        out["j_n5b_2"] = "shadow_learned"
    else:
        out["j_n5b_2"] = "shadow_inert"
    return out


def _face_header(path: Path) -> dict[str, Any] | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("kind") == "face":
            return row
    return None


def judge_pairing(
    treated: dict[str, Any],
    control: dict[str, Any] | None,
    treated_face: Path | None,
    control_face: Path | None,
) -> dict[str, Any]:
    """J-N5b-3：只问「这两臂能不能比」，不问「谁更好」。"""
    requested = {
        "treated_requested": treated.get("shadow_gate_requested"),
        "control_requested": control.get("shadow_gate_requested") if control else None,
    }
    if control is None:
        return {
            **requested,
            "status": "unverified_no_control_arm",
            "j_n5b_3": "unverified",
        }
    if treated.get("shadow_gate_requested") == control.get("shadow_gate_requested"):
        return {
            **requested,
            "status": "pairing_invalid",
            "reason": "identical_requested_gate",
            "j_n5b_3": "invalid",
        }
    base: dict[str, Any] = {**requested, "quadruple": {}}
    if treated_face is None or control_face is None:
        base["status"] = "gate_differs_quadruple_unverified"
        base["j_n5b_3"] = "unverified"
        return base
    head_t, head_c = _face_header(treated_face), _face_header(control_face)
    if head_t is None or head_c is None:
        return {
            **base,
            "status": "pairing_invalid",
            "reason": "face_header_missing",
            "j_n5b_3": "invalid",
        }
    differing: list[str] = []
    for label, path in QUADRUPLE:
        value_t, value_c = _dig(head_t, path), _dig(head_c, path)
        base["quadruple"][label] = {"treated": value_t, "control": value_c}
        if value_t != value_c:
            differing.append(label)
    if differing:
        return {
            **base,
            "status": "pairing_invalid",
            "reason": "quadruple_differs",
            "differing": differing,
            "j_n5b_3": "invalid",
        }
    base["status"] = "pairing_valid_on_quadruple"
    base["j_n5b_3"] = "valid"
    return base


def judge_metric_lane(
    pair_file: Path | None, role_treated: str, role_control: str, line: float
) -> dict[str, Any]:
    """J-N5b-4 的算术：段均值由 N3 那台仪器出，这里只做「末两段均值＋差值＋过线」三步算术。

    刻意**不自己分段**：分段/噪声带/缺列丢弃（DEBT-G63 的 `window_ticks=0` 收尾行）都已在
    `adjudicate_taiji_n3a_scaling_probe.py` 里冻结并各有测；在这里重抄就是第二条链。

    也刻意**不搬** N3 的 `single_variable_check`：那一支判的是「两臂同 `mean_revisits` ⇒ 仪器坏」，
    那是**剂量对照**的设计前提；本配对是「同剂量、只差通电」，两臂剂量相同正是要求 ⇒ 搬过来会误报。
    """
    if pair_file is None:
        return {"j_n5b_4": "unverified_no_metric_file", "line": line}
    payload = json.loads(pair_file.read_text(encoding="utf-8"))
    arms = payload.get("arms", {})
    missing = [role for role in (role_treated, role_control) if role not in arms]
    if missing:
        return {
            "j_n5b_4": "unverified_role_absent_in_metric_file",
            "missing_roles": missing,
            "line": line,
        }
    treated, control = arms[role_treated], arms[role_control]
    base: dict[str, Any] = {
        "metric_file": str(pair_file),
        "roles": {"treated": role_treated, "control": role_control},
        "line": line,
        "measurement_complete": {
            "treated": bool(treated.get("measurement_complete")),
            "control": bool(control.get("measurement_complete")),
        },
        "segment_means": {
            "treated": treated.get("acc_segment_means"),
            "control": control.get("acc_segment_means"),
        },
    }
    if not (base["measurement_complete"]["treated"] and base["measurement_complete"]["control"]):
        base["j_n5b_4"] = "unverified_metric_incomplete"
        return base
    for side, arm in (("treated", treated), ("control", control)):
        means = arm.get("acc_segment_means") or []
        if len(means) < 2:
            base["j_n5b_4"] = "unverified_segment_count_below_two"
            base["segment_count"] = {
                "treated": len(base["segment_means"]["treated"] or []),
                "control": len(base["segment_means"]["control"] or []),
            }
            return base
        last_two = (float(means[-1]) + float(means[-2])) / 2.0
        base.setdefault("last_two_segment_mean", {})[side] = last_two
    delta = base["last_two_segment_mean"]["treated"] - base["last_two_segment_mean"]["control"]
    base["delta_treated_minus_control"] = delta
    #: PLAN-N5-03 §1 的机械公式：`ruler_usable` **只看噪声带与过线界**，刻意不看 delta 的符号或大小
    #: ⇒ 它不可能被用来把已经看到的 delta 救成「成立」或压成「不成立」。带取自 N3 仪器原样输出。
    bands = {
        "treated": float(treated.get("acc_noise_band_adjacent_max") or 0.0),
        "control": float(control.get("acc_noise_band_adjacent_max") or 0.0),
    }
    base["bands"] = bands
    band_max = max(bands["treated"], bands["control"])
    ruler_usable = bands["treated"] > 0.0 and bands["control"] > 0.0 and line >= band_max
    base["ruler_usable"] = bool(ruler_usable)
    if not ruler_usable:
        #: §2 第一支：这把尺答不了这个问题——既不写成立也不写不成立。
        base["verdict"] = "ruler_unusable"
        base["j_n5b_4"] = "ruler_unusable"
        base["swallowed_by"] = [side for side, value in bands.items() if value >= line]
        return base
    base["exceeds_line"] = bool(delta > line)
    base["within_noise_band"] = bool(abs(delta) < band_max)
    base["verdict"] = "holds" if delta > line else "not_holds"
    base["j_n5b_4"] = base["verdict"]
    return base


def _same_source_failures(side: str, payload: dict[str, Any]) -> list[str]:
    #: 只读配对件自己出版的结论（G-N5g-1「不许重造」）；守卫名缺席＝非 `ok`＝红，不当"没这回事"。
    return [
        f"{side}.{guard}"
        for guard in ("G_N2c_3_cap0_same_source", "G_N2c_3_replay_same_source")
        if payload["guards"].get(guard, {}).get("status") != "ok"
    ]


def judge_retention_lane(
    treated_pair: Path | None,
    control_pair: Path | None,
    replicate_pair: Path | None,
) -> dict[str, Any]:
    """PLAN-N5-08 §2/§3 的 `J-N5b-5` 取数式（本器此前把这一格写死成占位字面量）。

    三条不可省的形状：地板列先剔（两个 0 会伪装成两项证据）；`noise_floor` 缺位时**只填未判**，
    不许因为"数已经能算"就出方向；列被移出分母后不足 3 列 ⇒ 整枚不判（[[guard-must-be-able-to-fail]]）。
    不给三个旗标时返回值与旧的占位行为逐字相同 ⇒ 已入库读数不被本笔追认或改写。
    """

    if treated_pair is None or control_pair is None:
        return {"j_n5b_5": "unverified_retention_lane_not_run"}
    loaded = {
        "treated": _read_pair(treated_pair),
        "control": _read_pair(control_pair),
    }
    #: G-N5g-1：同源这件事配对件自己已经判过（`adjudicate_taiji_n2_04_retention_pair.py` 的
    #: `G_N2c_3_*`），本器只读它的结论、不重造。任一非 `ok` ⇒ rc=2 且那一格保持 `unverified_*`。
    guard_failures = [
        failure for side, payload in loaded.items() for failure in _same_source_failures(side, payload)
    ]
    if guard_failures:
        return {"j_n5b_5": "unverified_same_source_guard_not_ok", "guard_failures": guard_failures}
    #: J-N5g-3：两臂各自的 `verdict` 不一致时不许挑轻的那一个——那是 PLAN-N5-04 §4 的 `arm_dependent`。
    verdicts = {side: payload["verdict"] for side, payload in loaded.items()}
    if verdicts["treated"] != verdicts["control"]:
        return {"j_n5b_5": "arm_dependent", "arm_verdicts": verdicts}
    #: 「同源」只按配对件自己出版的摘要判，**不比文件名**——两条分支的 before 面若同名而不同字节，
    #: 比名字会把「不同源」读成「同源」，那正是本式要防的那类假过（㊵-660 实施时的现改）。
    faces = {
        side: {
            "cap0_before": Path(payload["cap0_before_path"]).name,
            "replay_before": Path(payload["replay_before_path"]).name,
            "cap0_before_sha256": payload["cap0_before_sha256"],
            "replay_before_sha256": payload["replay_before_sha256"],
        }
        for side, payload in loaded.items()
    }
    if faces["treated"] != faces["control"]:
        return {"j_n5b_5": "faces_not_same_source", "baseline_faces": faces}
    #: 列名集合住在 `per_column` 的键里；`criteria.columns_compared` 是**计数**（㊵-660 现读纠正：
    #: 我预注册时把它当成了列表——见 PLAN-N5-08 §2 的带日期更正）。两者必须互相印证，
    #: 只对得上才继续，否则"7 列"这句话就是我自己数的。
    columns = sorted(loaded["treated"]["per_column"])
    declared = loaded["treated"]["criteria"].get("columns_compared")
    if not isinstance(declared, int) or declared != len(columns):
        return {"j_n5b_5": "column_count_mismatch", "declared": declared, "found": len(columns)}
    if sorted(loaded["control"]["per_column"]) != columns:
        return {"j_n5b_5": "column_sets_differ", "columns": {"treated": columns}}
    rows: dict[str, dict[str, int]] = {}
    floor: list[str] = []
    for column in columns:
        treated_cell = loaded["treated"]["per_column"][column]
        control_cell = loaded["control"]["per_column"][column]
        if treated_cell["before"] != control_cell["before"]:
            return {"j_n5b_5": "baseline_differs", "column": column}
        rows[column] = {
            "baseline": int(treated_cell["before"]),
            "treated_after": int(treated_cell["after"]),
            "control_after": int(control_cell["after"]),
        }
        if rows[column]["baseline"] == 0:
            floor.append(column)
    judgable = [column for column in columns if column not in floor]
    out: dict[str, Any] = {
        "pair_reports": {"treated": str(treated_pair), "control": str(control_pair)},
        "baseline_faces": faces["treated"],
        #: §7 发表资格前置③：两臂 `verdict` 必须随值一起出版，否则"保持侧有方向"这句话不完整。
        "arm_verdicts": verdicts,
        "columns_compared": len(columns),
        "not_judgable_floor": {column: rows[column]["baseline"] for column in floor},
        "drop_control_minus_treated": {
            column: rows[column]["control_after"] - rows[column]["treated_after"] for column in judgable
        },
    }
    if len(judgable) < 3:
        out["j_n5b_5"] = "not_judgable_below_min"
        out["judgable_columns"] = judgable
        return out
    if replicate_pair is None:
        #: PLAN-N5-08 J-N5g-2：没有同码同参的第三档，臂间差分不清是影子还是浮点漂移。
        out["j_n5b_5"] = "unverified_noise_floor_missing"
        out["judgable_columns"] = judgable
        return out
    replicate = _read_pair(replicate_pair)
    #: 第三档自己也要过 G-N5g-1：它的守卫非 `ok` ⇒ 它出版的 `noise_floor` 也不可信用。
    replicate_failures = _same_source_failures("noise_floor_arm", replicate)
    if replicate_failures:
        out["j_n5b_5"] = "unverified_same_source_guard_not_ok"
        out["guard_failures"] = replicate_failures
        out["noise_floor_pair"] = str(replicate_pair)
        return out
    #: 第三档必须量在同一张 before 面上，否则「噪声地板」量的是两张分布之差。
    if {
        key: replicate[key]
        for key in ("cap0_before_sha256", "replay_before_sha256")
    } != {key: loaded["treated"][key] for key in ("cap0_before_sha256", "replay_before_sha256")}:
        out["j_n5b_5"] = "faces_not_same_source"
        out["noise_floor_pair"] = str(replicate_pair)
        return out
    deltas = [
        abs(int(replicate["per_column"][column]["after"]) - rows[column]["control_after"])
        for column in columns
        if column in replicate["per_column"]
    ]
    if len(deltas) != len(columns):
        out["j_n5b_5"] = "noise_floor_columns_incomplete"
        return out
    #: 第三档的 before 列值要与两臂逐位同（同一张基座面）；不同就是摘要核对漏掉了什么。
    off = [
        column
        for column in columns
        if int(replicate["per_column"][column]["before"]) != rows[column]["baseline"]
    ]
    if off:
        out["j_n5b_5"] = "baseline_differs"
        out["column"] = off[0]
        return out
    noise_floor = max(deltas)
    out["noise_floor"] = noise_floor
    out["noise_floor_pair"] = str(replicate_pair)
    resolved = [column for column in judgable if abs(out["drop_control_minus_treated"][column]) > noise_floor]
    out["not_resolved"] = [column for column in judgable if column not in resolved]
    if len(resolved) < 3:
        out["j_n5b_5"] = "not_resolved_insufficient_range"
        return out
    drops = [out["drop_control_minus_treated"][column] for column in resolved]
    out["resolved_columns"] = resolved
    out["j_n5b_5"] = "holds" if all(drop <= 0 for drop in drops) else "cost_persists"
    return out


def _read_pair(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "taiji-n2-04-retention-pair-v1":
        raise ValueError(f"{path} 的 format={payload.get('format')!r} ⇒ 本取数式只认保持侧配对件")
    for key in (
        "cap0_before_path",
        "replay_before_path",
        "cap0_before_sha256",
        "replay_before_sha256",
        "per_column",
        "criteria",
        "guards",
        "verdict",
    ):
        if key not in payload:
            raise ValueError(f"{path} 缺键 {key} ⇒ 拒判（不许 .get 猜零）")
    return payload


CONJUNCTION_BRANCHES = ("j_n5b_2", "j_n5b_3", "j_n5b_4", "j_n5b_5")
_ESTABLISHED = "established"
_NOT_ESTABLISHED = "not_established"
_UNMEASURED = "unmeasured"
#: PLAN-N5-09 §2 的白名单：值 → 三态。键用判据名（发表时用的就是这串字面量），值用件里的键。
#: `ruler_unusable` 记成 **not_established** 不是 unmeasured——依据 PLAN-N5-02 §4 :52 原文
#: "结论只能是『这把尺答不了这个问题』"，它限制的是结论；把它升成"成立"或降级成"没测"都算发明。
BRANCH_VALUE_CLASS: dict[str, dict[str, str]] = {
    "j_n5b_2": {"shadow_learned": _ESTABLISHED, "shadow_inert": _NOT_ESTABLISHED, "not_powered": _NOT_ESTABLISHED},
    "j_n5b_3": {"valid": _ESTABLISHED, "invalid": _NOT_ESTABLISHED},
    "j_n5b_4": {"holds": _ESTABLISHED, "not_holds": _NOT_ESTABLISHED, "ruler_unusable": _NOT_ESTABLISHED},
    "j_n5b_5": {
        "holds": _ESTABLISHED,
        "cost_persists": _NOT_ESTABLISHED,
        "arm_dependent": _NOT_ESTABLISHED,
        "not_judgable_below_min": _UNMEASURED,
        "not_resolved_insufficient_range": _UNMEASURED,
        "faces_not_same_source": _UNMEASURED,
        "column_sets_differ": _UNMEASURED,
        "baseline_differs": _UNMEASURED,
        "column_count_mismatch": _UNMEASURED,
        "noise_floor_columns_incomplete": _UNMEASURED,
    },
}
#: 判据名 <-> 件里的键名，两处都要能查回去（§3 的 missing 数组发表时用判据名）。
BRANCH_LABELS = {
    "j_n5b_2": "J-N5b-2",
    "j_n5b_3": "J-N5b-3",
    "j_n5b_4": "J-N5b-4",
    "j_n5b_5": "J-N5b-5",
}


def classify_branch(branch: str, value: Any) -> tuple[str, bool]:
    """返回（三态，是否"未列举"）。G-N5h-5：未列举的值一律落 unmeasured，并单独点名。

    `unverified_*` 是 PLAN-N5-02／N5-08 里已documented 的**家族**（不是新造的模糊词），按前缀归类；
    除此以外的未知值既不算成立也不算不成立——它把合取推回"未判"，同时抬 rc=2 让仪器自己响。
    """
    table = BRANCH_VALUE_CLASS[branch]
    if value in table:
        return table[value], False
    if isinstance(value, str) and value.startswith("unverified"):
        return _UNMEASURED, False
    return _UNMEASURED, True


def judge_conjunction(values: dict[str, Any]) -> dict[str, Any]:
    """PLAN-N5-09 §3 的四元合取：J-N5h-1／2／3 三形之一，配 G-N5h-2 的两个数组恒在。"""
    branches: dict[str, Any] = {}
    unmeasured: list[str] = []
    missing: list[str] = []
    unknown: list[str] = []
    for branch in CONJUNCTION_BRANCHES:
        value = values.get(branch)
        state, is_unknown = classify_branch(branch, value)
        branches[branch] = {"criterion": BRANCH_LABELS[branch], "value": value, "state": state}
        if is_unknown:
            unknown.append(f"{BRANCH_LABELS[branch]}={value!r}")
        if state == _NOT_ESTABLISHED:
            missing.append(BRANCH_LABELS[branch])
        elif state == _UNMEASURED:
            unmeasured.append(BRANCH_LABELS[branch])
    #: G-N5h-3（本件最容易写错的一格）：只要有一支未测，合取就走 `unverified`，
    #: 即使另有支已经判成不成立——把"还没测"混进"测了且否证"正是 DEBT-G89 那句假陈述的成因。
    #: 因此 J-N5h-3 要求 `missing` 留空（两词共现＝一句话说两件互斥的事），信息不丢：
    #: 每支的原值与三态都在 `branches` 里。
    if unknown or unmeasured:
        verdict = "unverified"
        missing = []
    elif missing:
        verdict = "not_established"
    else:
        verdict = "established"
    return {
        "j_n5b_6": verdict,
        "prereg": "PLAN-N5-09 §3 J-N5h-1／J-N5h-2／J-N5h-3",
        "branches": branches,
        "missing": missing,
        "unmeasured": unmeasured,
        "unknown_values": unknown,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N5 影子通电验收判读（PLAN-N5-02 J-N5b-1/2/3）")
    parser.add_argument("--arm-treated", required=True)
    parser.add_argument("--arm-control", default=None)
    parser.add_argument("--face-treated", default=None)
    parser.add_argument("--face-control", default=None)
    parser.add_argument("--n3a-pair-file", default=None)
    parser.add_argument("--metric-role-treated", default="arm_A")
    parser.add_argument("--metric-role-control", default="arm_B")
    parser.add_argument("--line", type=float, default=0.02)
    #: PLAN-N5-08：保持侧配对件＋同码同参第三档。三枚都不给时输出与旧版逐字相同。
    parser.add_argument("--retention-treated", default=None, help="治疗臂的保持侧配对件 JSON")
    parser.add_argument("--retention-control", default=None, help="控制臂的保持侧配对件 JSON")
    parser.add_argument(
        "--noise-floor-arm",
        default=None,
        help="同码同参、不带 --n5-shadow-gate 的第三档配对件（J-N5g-2 的出值前置）",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    treated_path = Path(args.arm_treated)
    if not treated_path.is_file():
        print(json.dumps({"format": FORMAT, "status": "missing_face", "face": str(treated_path)}))
        return 2

    arms: dict[str, Any] = {"treated": judge_arm("treated", treated_path)}
    rc = 0
    if arms["treated"]["j_n5b_1"] == "ran_not_measured":
        rc = max(rc, 2)
    elif arms["treated"]["j_n5b_2"] != "shadow_learned":
        rc = max(rc, 1)

    control_arm: dict[str, Any] | None = None
    if args.arm_control:
        control_path = Path(args.arm_control)
        if not control_path.is_file():
            print(
                json.dumps({"format": FORMAT, "status": "missing_face", "face": str(control_path)})
            )
            return 2
        control_arm = judge_arm("control", control_path)
        arms["control"] = control_arm
        if control_arm["j_n5b_1"] == "ran_not_measured":
            rc = max(rc, 2)

    metric = judge_metric_lane(
        Path(args.n3a_pair_file) if args.n3a_pair_file else None,
        args.metric_role_treated,
        args.metric_role_control,
        float(args.line),
    )

    pairing = judge_pairing(
        arms["treated"],
        control_arm,
        Path(args.face_treated) if args.face_treated else None,
        Path(args.face_control) if args.face_control else None,
    )
    if pairing["j_n5b_3"] == "invalid":
        rc = max(rc, 2)
    #: PLAN-N5-03：尺不可用＝这次比较作废 ⇒ rc 必须抬起，
    #: 否则一次"绿色"的判读会被读成"已经判完"。
    if metric.get("j_n5b_4") == "ruler_unusable":
        rc = max(rc, 1)

    retention = judge_retention_lane(
        Path(args.retention_treated) if args.retention_treated else None,
        Path(args.retention_control) if args.retention_control else None,
        Path(args.noise_floor_arm) if args.noise_floor_arm else None,
    )
    #: 结构失败的几种（不同源、列集不齐、基线不逐位同、噪声档列不全）必须响亮，
    #: 不能停在"未判"那一档——那会把"我没接对件"读成"这条判据还没到时候"。
    if retention["j_n5b_5"] in {
        "faces_not_same_source",
        "column_count_mismatch",
        "column_sets_differ",
        "baseline_differs",
        "noise_floor_columns_incomplete",
        "unverified_same_source_guard_not_ok",
    }:
        rc = max(rc, 2)
    #: `arm_dependent`／`holds`／`cost_persists` 都是**读数**，不由这里抬 rc——
    #: rc 表示仪器与前置的健康度，方向只在键里（拿 rc 表达方向会让"判出坏结果"看起来像崩）。

    #: PLAN-N5-09 §3：四元合取由取数式出值（DEBT-G93）。四枚成员原值先齐在这里，
    #: 归类表是 §2 的白名单——本器不在这行之外另加判断。
    conjunction = judge_conjunction(
        {
            "j_n5b_2": arms["treated"]["j_n5b_2"],
            "j_n5b_3": pairing["j_n5b_3"],
            "j_n5b_4": metric["j_n5b_4"],
            "j_n5b_5": retention["j_n5b_5"],
        }
    )
    if conjunction["unknown_values"]:
        #: G-N5h-5：白名单外的值不许有默认归类 ⇒ 合取走"未判"，同时让仪器自己响。
        rc = max(rc, 2)

    payload = {
        "format": FORMAT,
        "prereg": "PLAN-N5-02 §2 J-N5b-1／J-N5b-2／J-N5b-3 前半",
        "arms": arms,
        "pairing": pairing,
        "metric_lane": metric,
        "j_n5b_4": metric["j_n5b_4"],
        #: ㊵-660：这一格从前是硬编码字面量（DEBT-G89）。三枚旗标都不给时它仍是那句原话 ⇒
        #: 已入库读数不被追认改写；给了才按 PLAN-N5-08 §2/§3 的取数式出值。
        "j_n5b_5": retention["j_n5b_5"],
        #: ㊵-665：这一格从前也是硬编码字面量（DEBT-G93，与 G89 同形）。
        #: 值取自 PLAN-N5-09 §3 三形之一；`conjunction` 块同时出版四枚**原值不翻译**（§6 前置①）。
        "j_n5b_6": conjunction["j_n5b_6"],
        "conjunction": {k: v for k, v in conjunction.items() if k != "j_n5b_6"},
        "rc": rc,
    }
    if args.retention_treated or args.retention_control or args.noise_floor_arm:
        payload["retention_lane"] = {k: v for k, v in retention.items() if k != "j_n5b_5"}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, ensure_ascii=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
