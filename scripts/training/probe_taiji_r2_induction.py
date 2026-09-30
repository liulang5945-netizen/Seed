"""归纳探针（零训练，分钟级）：模型能否**从上下文复制**一个出现过的序列？

## 为什么有这一件

T7 + 三协议探针（2026-09-24）确认：**CAP D+E = 0/36 的直接原因**是模型**从不照抄上下文**
（回声率 0.0000），而 D/E 的题全部是「先告知→后提问」——本质是**从上下文复制**。
架构 docstring 声称无复制/归纳机制，但那是**说法**；本件把它变成**实测**。

## 设计（先冻结）

* 30 个随机串 `S`（各 6 个常用汉字，`random.Random(20260924)` 固定种子，可复现）；
* 三种 prompt：
  1. **`S + S`（两份拷贝）** ⇒ 经典归纳测法：若模型从上下文习得"S 之后接 S 的开头"，
     续写应以 `S[0]` 开头；
  2. **`S + 。`（单次出现）** ⇒ 基线：一次出现后的重复率（无归纳信号的对照）；
  3. **`S + R`（S 后跟一个不同的随机串 R）** ⇒ 退化吸引子检查：测模型是否**无脑重复第一个串**。
* 生成：冻结的 `generate(..., "greedy")`（96 字节，greedy 确定性），取前 6 字节与首字节判命中。
* 模型：`base`(16M) / `t1`(8M) / `t5`(5M, 掩码)。

## 判读线（先冻结）

| 档 | 条件（`S+S` 的**首字命中率**） | 含义 |
|---|---|---|
| 强归纳 | ≥ 0.5 | 架构有可用归纳 ⇒ 修法应围绕"用足它" |
| 弱归纳 | ≥ 0.2（远高于单字边际概率） | 有部分机制，需放大 |
| **无归纳** | **< 0.2** | **实测确认无归纳机制** ⇒ CAP D+E 不可达是结构性的 |

**边界**：本件是**能力测量**，不是项目判决；单字命中率 0.2 的分界参考
"随机串的下一位在无归纳时基本不可预测（特定汉字的边际概率≪0.01）"。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

POOL = "的一是了我不人在他有这上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气每并别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表德少克代员许稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通师立象数四失满战远格士音轻目条呢病始达深完今提求清王化空业思切怎非找片罗钱吗语元喜曾离飞科言干流欢约各即指合反题必该论交终林请医晚制球决窢传画保读运及则房早院量苦火布品近坐产答星精视五连司巴奇管类未朋且婚台夜青北队久乎越观落尽形影红爸百令周吧识步希亚术留市半热送兴造谈容极随演收首根讲整式取照办强石古华諣拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断福城故历惊脸选包紧争另建维绝树系伤示愿持千史谁准联妇纪基买志静阿诗独复痛消社算义竟确酒需单治卡幸兰念举仅钟怕共毛句息功官待究跟穿室易游程号居考突皮哪费倒价图具刚脑永歌响商礼细专黄块脚味灵改据般破引食仍存众注笔甚某沉血备习校默务土微娘须试怀料调广蜖苏显赛查密议底列富梦错座参八除跑亮假印设线温虽掉京初养香停际致阳纸李纳验助激够严证帝饭忘趣支春集丈木研班普导顿睡展跳获艺六波察群皇段急庭创区奥器谢弟店否害草排背止组州朝封睛板角况曲馆育忙质河续哥呼若推境遇雨标姐充围案伦护冷警贝著雪索剧啊船险烟依斗值帮汉慢佛肯闻唱沙局伴学春夏秋冬东西南北上下"

STR_LEN = 6
TRIALS = 30
RNG_SEED = 20260924
CONDITIONS = ("two_copies", "one_copy", "degenerate_check")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="可多次；缺省=base/t1/t5")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from diag_taiji_r2_surface_decode import _serialize_prompt, generate

    from api.seed_runtime import SeedRuntime

    checks: list[tuple[str, Path]] = [
        ("base_16M", PROJECT_ROOT / "checkpoints" / "seed_beta.pt"),
        ("t1_8M", PROJECT_ROOT / "output/taiji_r2_t1t2/t1/checkpoint_8000000.pt"),
        ("t5_5M", PROJECT_ROOT / "output/taiji_r2_t5/t5_r0_cue/checkpoint_5000000.pt"),
    ]
    for relative in args.checkpoint:
        path = Path(relative)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        checks.append((path.stem, path))

    rng = random.Random(RNG_SEED)
    strings = ["".join(rng.choice(POOL) for _ in range(STR_LEN)) for _ in range(TRIALS)]
    others = ["".join(rng.choice(POOL) for _ in range(STR_LEN)) for _ in range(TRIALS)]

    rows: list[dict[str, object]] = []
    for name, path in checks:
        if not path.is_file():
            raise SystemExit(f"{name} 的 checkpoint 不存在：{path}")
        runtime = SeedRuntime.load(path)
        taiji = runtime.model.substrate
        per_condition: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            first_hits = 0
            full_hits = 0
            for index, s in enumerate(strings):
                if condition == "two_copies":
                    prompt = s + s
                elif condition == "one_copy":
                    prompt = s + "。"
                else:
                    prompt = s + others[index]
                text = generate(
                    taiji,
                    _serialize_prompt(runtime, prompt),
                    "greedy",
                ).decode("utf-8", errors="replace")
                first_hits += int(text.startswith(s[0]))
                full_hits += int(text.startswith(s))
            per_condition[condition] = {
                "first_char_hit_rate": round(first_hits / TRIALS, 4),
                "full_loop_hit_rate": round(full_hits / TRIALS, 4),
            }
        two = per_condition["two_copies"]["first_char_hit_rate"]
        verdict = (
            "strong_induction"
            if two >= 0.5
            else ("weak_induction" if two >= 0.2 else "no_induction_measured")
        )
        rows.append({"model": name, "per_condition": per_condition, "verdict": verdict})
        print(
            json.dumps(
                {
                    "model": name,
                    "two_copies_first": per_condition["two_copies"]["first_char_hit_rate"],
                    "one_copy_first": per_condition["one_copy"]["first_char_hit_rate"],
                    "two_copies_full": per_condition["two_copies"]["full_loop_hit_rate"],
                    "degenerate_S0_rate": per_condition["degenerate_check"]["first_char_hit_rate"],
                    "verdict": verdict,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "format": "taiji-r2-induction-probe-v1",
        "rng_seed": RNG_SEED,
        "trials": TRIALS,
        "string_len_chars": STR_LEN,
        "tiers": {"strong": ">=0.5", "weak": ">=0.2", "none": "<0.2"},
        "rows": rows,
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports" / "taiji_r2_induction_probe_20260925.json")
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
