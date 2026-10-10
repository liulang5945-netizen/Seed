"""N5 保持侧配对的**跑道器**（PLAN-N5-04 §3 G-N5d-1 的机械执行面）。

**为什么要有它**：双臂跑完之后，"出四张 after 面＋逐臂判读"这件事一直是**我手拼命令**——
而本仓为手拼命令付过学费：㊵-593 那份"冻结命令"把带参旗标写成裸旗标，照抄即被 argparse 拒，
台账里那条过期处方比过期结论更贵。跑道器把这六条评测／判读命令（外加一次预检）钉成一份可复算的计划，并且**先过预检再花钱**：
预检说"题集内容不可声称同源"或"基座摘要对不上"，就不该去花那 13 分钟评测。

**硬前置（fail-closed，缺任何一条整道拒绝跑）**：
① 两枚 after 检查点都必须存在；② `check_taiji_n5_pairing_preflight` 对**将要配对的那两张 before 面**
必须给出 `cap0_identity=verified`、`cap0_checkpoint_identity=verified`、`replay_identity=content_hash_available`
——第三项就是 DEBT-G79 的来由：旧面只有偏移与首件，配对能跑但"题集逐字相同"这句话不能说；
③ 每一臂都要**点名自己的巩固语料**（㊵-565③ 冻的口径：分离机检用的材料＝本次通电自己的夜间落盘件，
所以两臂各指各的，本件**不给默认值**）。这一条也是花钱前的拒绝：判读器没有语料时会在
**最后一步**记 `G_N2c_4=unverified` 并 rc=2，那时候四张面已经付过了。
④ 点名的材料要**可能是本次跑自己产出的**——按件名时间戳与该臂 `progress_exit.json` 的运行窗口现算，
不信调用方自报。全部早于窗口 ⇒ `substituted`，必须带 `--allow-substituted-consolidation-material`
才放行；窗口或件名算不出 ⇒ `unknown`，照实出版、不得读成"材料是对的"。判定结果连同两臂记录写进
`<out-dir>/consolidation_material.json`（㊵-654：60,000 符号的 CLI 续训**结构性产不出**夜间件，
因为训练器里没有巩固生产者，所以这条跑道今天只能跑在替换材料上——那必须是一件**件里可查**的事，
不能只存在于我的记忆里）。

`--dry-run` 只打印将要执行的计划（含预检本身），不跑任何东西；`--only` 用于单独重跑某一侧的面。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAP0_EVALUATOR = "scripts/training/eval_taiji_cap0_baseline.py"
REPLAY_EVALUATOR = "scripts/training/measure_taiji_a30_repetition_penalty.py"
ADJUDICATOR = "scripts/training/adjudicate_taiji_n2_04_retention_pair.py"
PREFLIGHT = "scripts/training/check_taiji_n5_pairing_preflight.py"
#: G-N5d-4 的**现成**执行器（PLAN-N5-02 §2 J-N5b-1）。本跑道器先前只复算了保持侧七列，
#: 没有把这台仪器排进来——㊵-654 那次"双臂"就是这样花掉 2×13 分钟才发现两臂是同一枚权重。
#: 它只读信封、零算力，所以放在**花钱之前**当门用。
SHADOW_PRESENCE = "scripts/training/adjudicate_taiji_n5_shadow_gate.py"

CAP0_BEFORE = Path("reports/taiji_n2_cap0_before_20261008.json")
#: 默认指向**带内容哈希**的那张新 before 面（㊵-645 出路①）；旧面仍可指名，但预检会拒绝配对。
REPLAY_BEFORE = Path("reports/taiji_n2_replay24_before_20261010.json")
RETENTION_MANIFEST = "plans/manifests/cap0_eval_set_v2.json"

ARMS = ("treated", "control")
REQUIRED_PREFLIGHT = {
    ("cap0", "cap0_identity"): "verified",
    ("base_checkpoint", "cap0_checkpoint_identity"): "verified",
    ("replay", "replay_identity"): "content_hash_available",
}


def preflight_refusals(payload: dict[str, Any]) -> list[str]:
    """预检读数 ⇒ 拒绝清单（空表才允许花钱跑）。"""

    refusals: list[str] = []
    for (block, key), expected in REQUIRED_PREFLIGHT.items():
        observed = payload.get(block, {}).get(key)
        if observed != expected:
            refusals.append(f"{block}.{key}={observed!r}（需要 {expected!r}）")
    return refusals


SINGLE_VARIABLE_FLAGS = ("--n5-shadow-gate",)
#: 逐臂**必然**不同的三枚落点旗标。它们不参与"单变量"比较，但反过来要查一件真事：
#: 两臂不许指向同一枚档——那会让第二臂原地覆写第一臂，而 ㊵-647 那版比较把这种差异
#: 也当成"第二个变量"，于是双臂跑完之后才被本门拦死（㊵-650 的实测形状）。
ARM_LOCAL_FLAGS = ("--checkpoint", "--progress", "--pressure-record")


def corpus_refusals(arm_corpus: dict[str, list[Path]]) -> list[str]:
    """G-N2c_4 分离机检的输入是否在场——缺了它判读器要到**最后一步**才 rc=2。

    两臂各自点名（㊵-565③）：夜间材料由本次通电自己落盘，影子开关恰恰会改变它，
    所以"共用一枚默认语料"会把控制臂的分离检查建立到治疗臂的材料上。
    """

    refusals: list[str] = []
    for arm in ARMS:
        paths = arm_corpus.get(arm) or []
        if not paths:
            refusals.append(
                f"{arm}: 没点名巩固语料 ⇒ 判读器记 G_N2c_4_disjointness=unverified 并 rc=2"
            )
            continue
        for path in paths:
            resolved = path if path.is_absolute() else PROJECT_ROOT / path
            if not resolved.is_file():
                refusals.append(f"{arm}: 巩固语料不在场：{path}")
    return refusals


#: 夜间件名里的时间戳形状：`corpus-20261009T035904Z-8e1c02165ec4.jsonl`
NIGHT_NAME_PATTERN = re.compile(r"corpus-(\d{8}T\d{6}Z)")
#: 运行窗口的左右余量（秒）：落盘与收尾存盘之间可能有分钟级偏移，不该是小时级以上。
RUN_WINDOW_SLACK_SECONDS = 3600.0


def _night_timestamp(path: Path) -> datetime | None:
    match = NIGHT_NAME_PATTERN.search(path.name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def _run_window(checkpoint: Path) -> tuple[float, float] | None:
    """臂自己的运行窗口，取自**同目录的** `progress_exit.json`（训练器的完成记账）。

    为什么不用"比 `.pt` 新"当判据：夜间件是在跑**当中**落盘的，必然早于收尾存盘的档——
    那条会把合法的本夜产物全判成替换（我写这版时先按"更新"想过，被这个顺序否证）。
    """

    accounting = checkpoint.parent / "progress_exit.json"
    if not accounting.is_file():
        return None
    try:
        payload = json.loads(accounting.read_text(encoding="utf-8"))
        end = checkpoint.stat().st_mtime
        start = end - float(payload["elapsed_seconds"]) - RUN_WINDOW_SLACK_SECONDS
        return (start, end + RUN_WINDOW_SLACK_SECONDS)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def consolidation_provenance(arm: str, checkpoint: Path, paths: list[Path]) -> dict[str, Any]:
    """"点名材料是否可能出自本次跑"——按件名时间戳与运行窗口算，**不信调用方自报**。

    三档互斥：`arm_local`＝至少一枚落在窗口内；`substituted`＝窗口可算但全部更早；
    `unknown`＝窗口算不出或件名不带时间戳。⇒ 只有 `arm_local` 兑现 ㊵-565③ 那句口径；
    `substituted` 必须调用方显式确认（见 `--allow-substituted-consolidation-material`），
    `unknown` 按"没算"处理——它不能被读成"材料是对的"。
    """

    parsed = {path.name: _night_timestamp(path) for path in paths}
    stamps = {name: ts.isoformat() for name, ts in parsed.items() if ts is not None}
    window = _run_window(checkpoint)
    record: dict[str, Any] = {
        "arm": arm,
        "files": [str(path) for path in paths],
        "night_timestamps_utc": stamps,
        "run_window_utc": None
        if window is None
        else [
            datetime.fromtimestamp(window[0], UTC).isoformat(),
            datetime.fromtimestamp(window[1], UTC).isoformat(),
        ],
    }
    if window is None or len(stamps) != len(paths):
        record["provenance"] = "unknown"
        return record
    inside = [ts for ts in parsed.values() if ts is not None and window[0] <= ts.timestamp() <= window[1]]
    record["provenance"] = "arm_local" if inside else "substituted"
    if not inside:
        newest = max(ts for ts in parsed.values() if ts is not None)
        record["newest_material_before_run_utc"] = newest.isoformat()
    return record


def _normalized(path: str) -> str:
    #: 路径比较前先归一分隔符——win32 上同一枚基座会被写成 `checkpoints\x` 或 `checkpoints/x`
    #: （DEBT-I8 那族），不归一会造出假"不同源"。
    return str(path).replace("\\", "/")


def lineage_refusals(
    arms: dict[str, dict[str, Any]],
    *,
    base_checkpoint: str,
    base_sha256: str | None,
) -> list[str]:
    """G-N5d-2「两臂只差那一枚旗标」的机器执行（纯函数：入参是各臂信封里读出的元数据）。

    缺任何一条都拒绝，并点名是哪一条——配对的有效性全押在"两臂只差一个变量"上，
    而这件事今天只有我记得，没有读数。
    """

    refusals: list[str] = []
    if set(arms) != set(ARMS):
        return [f"arms={sorted(arms)}（需要两臂 {list(ARMS)}）"]
    for arm, meta in arms.items():
        argv = meta.get("argv")
        if not argv:
            refusals.append(f"{arm}: 信封里没有 command_surface.argv ⇒ 没法核对它是哪条命令产的")
            continue
        if "--resume" not in argv:
            refusals.append(
                f"{arm}: 没有 --resume ⇒ 不是从同一基件续训（从头臂会让 before 比较失效）"
            )
            continue
        resume = argv[argv.index("--resume") + 1]
        if _normalized(resume) != _normalized(base_checkpoint):
            refusals.append(f"{arm}: --resume={resume!r} 与预期基座 {base_checkpoint!r} 不是同一枚")
        if base_sha256 is None:
            refusals.append("base: 拿不到基座摘要 ⇒ 无法证明 before 面与两臂同底")
        for key in ("corpus_fingerprint", "config"):
            if meta.get(key) is None:
                refusals.append(f"{arm}: 缺 {key} ⇒ 单变量核对没依据")
    treated, control = arms.get("treated", {}), arms.get("control", {})
    for key in ("corpus_fingerprint", "config"):
        if treated.get(key) is not None and treated.get(key) != control.get(key):
            refusals.append(f"{key} 在两臂间不同 ⇒ 配对的差异不再只有那一枚旗标")
    left, right = treated.get("argv"), control.get("argv")
    if left and right:
        #: 单变量核对做成"去掉那一枚旗标与三枚逐臂落点之后两条 argv 必须逐位相同"，
        #: 而不是"允许若干差异"——后者会把 `--limit 24` 对 `--limit 25` 也放过。
        #: 比较前逐枚归一分隔符（DEBT-I8 那族：win32 路径两种写法指同一枚件）。
        stripped = SINGLE_VARIABLE_FLAGS + ARM_LOCAL_FLAGS
        if [_normalized(tok) for tok in _without_gate(left, stripped)] != [
            _normalized(tok) for tok in _without_gate(right, stripped)
        ]:
            refusals.append(
                "两臂 argv 去掉旗标与逐臂落点之后仍不同 ⇒ 单变量前提破了："
                f"treated={_without_gate(left, stripped)[:8]} "
                f"control={_without_gate(right, stripped)[:8]}"
            )
        if _gate_value(left) == _gate_value(right):
            refusals.append(
                f"两臂的 --n5-shadow-gate 请求值相同（{_gate_value(left)!r}）⇒ 这一对没有可比的两臂"
            )
        for flag in ARM_LOCAL_FLAGS:
            mine, theirs = _value_of(left, flag), _value_of(right, flag)
            if mine is not None and theirs is not None and _normalized(mine) == _normalized(theirs):
                refusals.append(
                    f"{flag}：两臂落点相同（{mine}）⇒ 第二臂会原地覆写第一臂的档，配对里剩下的那一臂不是它自己"
                )
    return refusals


def _without_gate(argv: list[str], flags: tuple[str, ...] = SINGLE_VARIABLE_FLAGS) -> list[str]:
    out = list(argv)
    for flag in flags:
        while flag in out:
            index = out.index(flag)
            del out[index]
            if index < len(out) and not str(out[index]).startswith("-"):
                del out[index]
    return out


def _value_of(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        index = argv.index(flag) + 1
        return str(argv[index]) if index < len(argv) else None
    return None


def _gate_value(argv: list[str]) -> str | None:
    for flag in SINGLE_VARIABLE_FLAGS:
        value = _value_of(argv, flag)
        if value is not None:
            return value
    return None


def build_plan(
    *,
    treated: Path,
    control: Path,
    out_dir: Path,
    consolidation_corpus: dict[str, list[str]],
    cap0_before: Path = CAP0_BEFORE,
    replay_before: Path = REPLAY_BEFORE,
    replay_limit: int,
    only: str | None,
) -> list[list[str]]:
    """返回将要执行的命令（ argv 列表，第一项是可执行程序名）。"""

    python = [sys.executable]
    plan: list[list[str]] = []
    for arm in ARMS:
        checkpoint = treated if arm == "treated" else control
        if only in (None, "cap0"):
            plan.append(
                python
                + [
                    str(PROJECT_ROOT / CAP0_EVALUATOR),
                    "--checkpoint",
                    str(checkpoint),
                    "--report",
                    str(out_dir / f"{arm}_cap0_after.json"),
                ]
            )
        if only in (None, "replay"):
            plan.append(
                python
                + [
                    str(PROJECT_ROOT / REPLAY_EVALUATOR),
                    "--checkpoint",
                    str(checkpoint),
                    "--no-circuit",
                    "--limit",
                    str(replay_limit),
                    "--out-report",
                    str(out_dir / f"{arm}_replay_after.json"),
                ]
            )
        if only in (None, "adjudicate"):
            adjudicate = python + [
                str(PROJECT_ROOT / ADJUDICATOR),
                "--cap0-before",
                str(cap0_before),
                "--cap0-after",
                str(out_dir / f"{arm}_cap0_after.json"),
                "--replay-before",
                str(replay_before),
                "--replay-after",
                str(out_dir / f"{arm}_replay_after.json"),
                "--retention-manifest",
                RETENTION_MANIFEST,
            ]
            for corpus_path in consolidation_corpus[arm]:
                adjudicate += ["--consolidation-corpus", str(corpus_path)]
            adjudicate += ["--out", str(out_dir / f"{arm}_retention_pair.json")]
            plan.append(adjudicate)
    return plan


def _run(argv: list[str]) -> int:
    print("$ " + " ".join(argv), flush=True)
    return subprocess.run(argv, check=False).returncode


def _envelope_meta(path: Path) -> dict[str, Any]:
    """从臂的检查点信封里取"它是哪条命令产的"那三样。"""

    import torch

    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(envelope, dict):
        return {"argv": None, "corpus_fingerprint": None, "config": None}
    meta = envelope.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    surface = meta.get("command_surface")
    if not isinstance(surface, dict):
        surface = envelope.get("command_surface")
    surface = surface if isinstance(surface, dict) else {}
    argv = surface.get("argv")
    return {
        "argv": [str(row) for row in argv] if isinstance(argv, list) else None,
        "corpus_fingerprint": meta.get("corpus_fingerprint"),
        "config": envelope.get("config"),
    }


def main(argv: list[str] | None = None) -> int:
    #: 本件的输出**就是**给人看的拒绝清单，而 win32 控制台默认按 GBK 编码 stdout：
    #: 清单里的 `⇒` 不可编码，实测让拒绝路径变成 `UnicodeEncodeError`＋rc=1（ traceback 取代了
    #: 设计好的 rc=2 与逐条点名）。照 `eval_taiji_artifact_consumption_policy.py:337` 的既有写法先把
    #: 输出钉成 UTF-8——fail-closed 的消息必须能fail-closed **地说话**。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="N5 retention-lane runner (static pre-flight first)"
    )
    parser.add_argument("--treated-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cap0-before", type=Path, default=CAP0_BEFORE)
    parser.add_argument("--replay-before", type=Path, default=REPLAY_BEFORE)
    parser.add_argument("--replay-limit", type=int, default=24)
    parser.add_argument(
        "--treated-consolidation-corpus",
        type=Path,
        action="append",
        default=[],
        metavar="JSONL",
        help="治疗臂本次通电落盘的夜间语料（分离机检的输入，可重复给）",
    )
    parser.add_argument(
        "--control-consolidation-corpus",
        type=Path,
        action="append",
        default=[],
        metavar="JSONL",
        help="控制臂本次通电落盘的夜间语料（不给默认值，两臂各自点名）",
    )
    parser.add_argument("--only", choices=("cap0", "replay", "adjudicate"), default=None)
    parser.add_argument(
        "--treated-face",
        type=Path,
        default=None,
        help="治疗臂的压强面（四元组核对的输入，缺省取该臂档同目录的 pressure.jsonl）",
    )
    parser.add_argument(
        "--control-face",
        type=Path,
        default=None,
        help="控制臂的压强面（同上；不在场则照实出 unverified 披露，不冒充核对过）",
    )
    parser.add_argument(
        "--allow-substituted-consolidation-material",
        action="store_true",
        help="承认点名的夜间材料早于本臂运行窗口（替换材料），分离机检结论只算跑在替换材料上",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    args = parser.parse_args(argv)

    for label, path in (
        ("treated", args.treated_checkpoint),
        ("control", args.control_checkpoint),
    ):
        if not path.is_file():
            print(f"REFUSE: {label} checkpoint 不在场：{path}")
            return 2
    for label, path in (("cap0-before", args.cap0_before), ("replay-before", args.replay_before)):
        if not (path if path.is_absolute() else PROJECT_ROOT / path).is_file():
            print(f"REFUSE: {label} 不在场：{path}")
            return 2

    arm_corpus = {
        "treated": list(args.treated_consolidation_corpus),
        "control": list(args.control_consolidation_corpus),
    }
    #: 只有真要判读的那几步才需要语料——单跑面时别把它变成假前置。
    if args.only in (None, "adjudicate"):
        missing_corpus = corpus_refusals(arm_corpus)
        if missing_corpus:
            print("REFUSE: 分离机检的巩固语料不齐（这一步在判读器里是**最后一步**才红）：")
            for row in missing_corpus:
                print("  - " + row)
            return 2

    #: ④ 材料溯源：按件名时间戳与该臂运行窗口现算，不信自报。
    arm_checkpoints = {
        "treated": args.treated_checkpoint,
        "control": args.control_checkpoint,
    }
    material: dict[str, Any] = {}
    if args.only in (None, "adjudicate"):
        substitutions: list[str] = []
        for arm in ARMS:
            record = consolidation_provenance(arm, arm_checkpoints[arm], arm_corpus[arm])
            material[arm] = record
            if record["provenance"] == "substituted":
                substitutions.append(
                    f"{arm}: 最新材料 {record.get('newest_material_before_run_utc')} 早于运行窗口 "
                    f"{record['run_window_utc'][0]}"
                )
        if substitutions and not args.allow_substituted_consolidation_material:
            print("REFUSE: 点名的巩固材料不可能是本次跑产出的（㊵-565③ 的口径落不到实处）：")
            for row in substitutions:
                print("  - " + row)
            print(
                "  确实要用替换材料 ⇒ 带 --allow-substituted-consolidation-material，"
                "结论会照实标注为『跑在替换材料上』"
            )
            return 2
        for arm in ARMS:
            record = material.get(arm)
            if record is not None:
                print(f"material[{arm}] provenance={record['provenance']} files={len(record['files'])}")

    plan = build_plan(
        treated=args.treated_checkpoint,
        control=args.control_checkpoint,
        out_dir=args.out_dir,
        consolidation_corpus=arm_corpus,
        cap0_before=args.cap0_before,
        replay_before=args.replay_before,
        replay_limit=args.replay_limit,
        only=args.only,
    )
    preflight_argv = [
        sys.executable,
        str(PROJECT_ROOT / PREFLIGHT),
        "--cap0-before",
        str(args.cap0_before),
        "--replay-before",
        str(args.replay_before),
        "--out-report",
        str(args.out_dir / "preflight.json"),
    ]
    print("preflight: $ " + " ".join(preflight_argv))
    for row in plan:
        print("plan: $ " + " ".join(row))
    #: G-N5d-4 的门：读两臂信封的 `n5_shadow` 自述块，零算力，所以排在花钱之前。
    #: 两面也一并线进去——PLAN-N5-04 §3 G-N5d-2 原文写的是"四元组由这台仪器从面头读取并出版，
    #: 本件不另写一份"，只给档不给面会让它按 `gate_differs_quadruple_unverified` 出 `unverified`，
    #: 那条"核对过了"就成了我的转述（默认取该臂档同目录的 `pressure.jsonl`，不在场就不给）。
    presence_argv: list[str] = []
    if args.only in (None, "adjudicate"):
        presence_argv = [
            sys.executable,
            str(PROJECT_ROOT / SHADOW_PRESENCE),
            "--arm-treated",
            str(args.treated_checkpoint),
            "--arm-control",
            str(args.control_checkpoint),
            "--out",
            str(args.out_dir / "shadow_gate_pair.json"),
        ]
        for flag, checkpoint, explicit in (
            ("--face-treated", args.treated_checkpoint, args.treated_face),
            ("--face-control", args.control_checkpoint, args.control_face),
        ):
            face = explicit if explicit is not None else checkpoint.parent / "pressure.jsonl"
            if face.is_file():
                presence_argv += [flag, str(face)]
            else:
                print(f"disclosure: {flag} 不在场（{face}）⇒ 该臂四元组按 unverified 出版，不冒充核对")
        print("gate: $ " + " ".join(presence_argv))
    if args.dry_run:
        print(f"DRY_RUN commands={len(plan)}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if material:
        (args.out_dir / "consolidation_material.json").write_text(
            json.dumps(material, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    rc = _run(preflight_argv)
    if rc != 0:
        print(f"REFUSE: 预检 rc={rc} ⇒ 没算成，不花评测的时间")
        return 2
    payload = json.loads((args.out_dir / "preflight.json").read_text(encoding="utf-8"))
    refusals = preflight_refusals(payload)
    if refusals:
        print("REFUSE: 配对的硬前置未满足：")
        for row in refusals:
            print("  - " + row)
        return 2
    #: G-N5d-2 的机器执行：两臂必须同底（`--resume` 指到 before 面声明的那枚基座）、
    #: 同语料指纹、同 config，且 argv 去掉 `--n5-shadow-gate` 之后逐位相同。
    base_checkpoint = str(payload.get("base_checkpoint", {}).get("checkpoint_path", ""))
    base_sha = payload.get("base_checkpoint", {}).get("declared_checkpoint_sha256")
    arms = {
        "treated": _envelope_meta(args.treated_checkpoint),
        "control": _envelope_meta(args.control_checkpoint),
    }
    lineage = lineage_refusals(arms, base_checkpoint=base_checkpoint, base_sha256=base_sha)
    if lineage:
        print("REFUSE: 单变量／同底核对未通过：")
        for row in lineage:
            print("  - " + row)
        return 2
    print(f"lineage_ok base={base_checkpoint} arms_match_single_variable=True")
    if presence_argv:
        rc = _run(presence_argv)
        if rc != 0:
            print(
                f"REFUSE: G-N5d-4 自述在场性 rc={rc} ⇒ 那枚单变量旗标在两臂的**产品自述**里没有可分辨的后果"
                "（㊵-654 实测：`n5_shadow` 整块缺席、`should_propose` 真值 0/9,727 ⇒ 两臂是同一枚权重）。"
                "四张面不跑——为塌臂的配对付 2×13 分钟是白付。"
            )
            return 2
    for row in plan:
        rc = _run(row)
        if rc != 0:
            print(f"REFUSE: 步骤失败 rc={rc}：{' '.join(row[1:3])} ⇒ 后续步骤不跑（半套面不可判）")
            return rc
    print(f"LANE_OK out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
