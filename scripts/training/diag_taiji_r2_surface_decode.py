"""R2 阶段零执行件：P1 解码策略消融 + P2 塌缩与分布诊断（零训练）。

合同：plans/reference/M5_R2_SURFACE_DECODING_CONTRACT_DRAFT_20260920.md（FROZEN 2026-09-20）。
冻结口径：成句 = UTF-8 严格可解码 且 长度 ≥ 8 字符 且 单一字符占比 ≤ 0.35
且 二元组自重复率 ≤ 语料逐行 95 分位（实测 0.406122，见合同 §5.1）。
判据：D1 存在非现状臂成句率 ≥ 现状 + 20 个百分点且逐题配对符号检验 p < 0.05 方向一致；
      D2 全部替代臂与现状差 < 5 个百分点。
不训练、不写检查点、不改任何冻结门。beam-2 一臂本件未实现（需快照分支状态），如实记 not_run。
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\Seed")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.probe_taiji_cap0_byte_output import _utf8_allowed  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "seed_beta.pt"
EVAL_SET = ROOT / "plans" / "manifests" / "cap0_eval_set_v2.json"
CORPUS = ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl"
OUT = ROOT / "reports" / "taiji_r2_surface_decode_p1p2_20260920.json"

BIGRAM_UPPER_BOUND = 0.406122  # 合同 §5.1 冻结值（该判据已被反向对照否证，保留仅作记录）
MIN_LEN = 8
MAX_SINGLE_SHARE = 0.35
GEN_LENGTH = 96
TEMPERATURE_SEED = 20260920  # 冻结合同未定种子；本件按"确定性优先"补固定种子并在此披露

#: 修订判据（合同 §6 修订：撤回受影响判定，同一意图改成可检验形态）。
#: 原"四条件合取"被实测否证——把中文打乱字序后仍然 100% 通过，量具无判别力。
#: 新判据必须先过**反向对照**（真中文必过、打乱中文必不过）才允许使用，过不了就拒跑。
#: 阈值取"真中文中位数(-6.8136) 与打乱中文中位数(-8.7833) 的中点"，非手拍；
#: 由 assert_criterion_discriminates 的反向对照守住（真中文须过、打乱须挂）。
NLL_THRESHOLD = -7.8  # 每字符平均 n 元对数似然下限；由 §controls 实测标定
NEGATIVE_CONTROL_FAIL_RATE = 0.9  # 打乱样本至少 90% 要被判否，否则判据作废


def build_ngram_model(limit: int = 200000) -> tuple[dict, dict, int, int]:
    """从训练语料建 uni/bi  gram 计数（+1 平滑）。零训练，纯统计。"""

    uni: Counter = Counter()
    bi: Counter = Counter()
    with CORPUS.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            if index >= limit:
                break
            try:
                text = "".join((json.loads(raw).get("text") or "").split())
            except ValueError:
                continue
            uni.update(text)
            bi.update(zip(text, text[1:]))
    return uni, bi, len(uni), sum(uni.values())


def mean_nll(text: str, model: tuple) -> float:
    uni, bi, vocab, total = model
    s = "".join(text.split())
    if len(s) < 2:
        return 0.0
    import math as _m

    alpha = 1.0
    score = 0.0
    for a, b in zip(s, s[1:]):
        p = (bi.get((a, b), 0) + alpha) / (uni.get(a, 0) + alpha * (vocab + 1))
        score += _m.log(max(p, 1e-12))
    return score / (len(s) - 1)


def well_formed(text: str, model: tuple | None = None) -> bool:
    try:
        text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        return False
    stripped = "".join(text.split())
    if len(stripped) < MIN_LEN:
        return False
    if Counter(stripped).most_common(1)[0][1] / len(stripped) > MAX_SINGLE_SHARE:
        return False
    if model is None:  # 没有 n 元模型时不得声称"成句"——那是被否证的旧口径
        raise RuntimeError("成句判据必须带语料 n 元模型；拒绝退回已被否证的四条件合取")
    return mean_nll(stripped, model) >= NLL_THRESHOLD


def assert_criterion_discriminates(model: tuple) -> dict:
    """反向对照门：真中文要过、打乱中文要挂；否则整件拒跑（不是调阈值）。"""

    import random

    good: list[str] = []
    with CORPUS.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            if len(good) >= 300:
                break
            try:
                text = "".join((json.loads(raw).get("text") or "").split())
            except ValueError:
                continue
            if len(text) >= 12:
                good.append(text[:24])
    rng = random.Random(11)
    shuffled = ["".join(rng.sample(t, len(t))) for t in good]
    pass_real = sum(1 for t in good if well_formed(t, model)) / len(good)
    fail_shuf = sum(1 for t in shuffled if not well_formed(t, model)) / len(shuffled)
    controls = {
        "real_pass_rate": round(pass_real, 4),
        "shuffled_fail_rate": round(fail_shuf, 4),
        "real_median_nll": round(statistics.median(mean_nll(t, model) for t in good), 4),
        "shuffled_median_nll": round(statistics.median(mean_nll(t, model) for t in shuffled), 4),
        "threshold": NLL_THRESHOLD,
        "n": len(good),
    }
    if pass_real < 0.5 or fail_shuf < NEGATIVE_CONTROL_FAIL_RATE:
        raise SystemExit(
            "判据无判别力，拒跑："
            f"真中文通过率={controls['real_pass_rate']} 打乱拒绝率={controls['shuffled_fail_rate']}"
            "（阈值不改，改的是判据本身）"
        )
    return controls


ARMS = {
    "greedy": {},
    "no_repeat": {"ban_consecutive_same": True},
    "topk8": {"top_k": 8},
    "temp07": {"temperature": 0.7},
}


def _serialize_prompt(runtime, prompt: str) -> bytes:
    from api.seed_runtime import SeedRuntime

    return SeedRuntime._serialize(prompt, []).encode("utf-8")


def _advance(remaining: int, symbol: int) -> tuple[int, int]:
    if remaining == 0:
        if symbol < 0x80:
            return 0, 0
        if symbol < 0xE0:
            return 1, symbol
        if symbol < 0xF0:
            return 2, symbol
        return 3, symbol
    return remaining - 1, 0


def generate(taiji, prompt: bytes, arm: str) -> bytes:
    opts = ARMS[arm]
    temperature = float(opts.get("temperature", 0.0))
    top_k = int(opts.get("top_k", 0))
    ban_repeat = bool(opts.get("ban_consecutive_same", False))

    taiji.reset_dynamics(episode_id="r2-surface-p1")
    step = taiji.observe(
        taiji.config.boundary_symbol,
        learn=False,
        readout="predictive",
        use_memory=False,
        use_identity=False,
    )
    for symbol in prompt:
        step = taiji.observe(
            int(symbol), learn=False, readout="predictive", use_memory=False, use_identity=False
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(TEMPERATURE_SEED)
    out: list[int] = []
    remaining, lead = 0, 0
    for _ in range(GEN_LENGTH):
        allowed = _utf8_allowed(remaining, lead)
        probabilities = step.probabilities.detach().cpu().clone()
        mask = torch.full_like(probabilities, -1.0)
        mask[torch.tensor(allowed, dtype=torch.long)] = 0.0
        if ban_repeat and out:
            mask[out[-1]] = -1.0
        logits = probabilities + mask
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            symbol = int(torch.multinomial(probs, 1, generator=generator).item())
        elif top_k:
            k = min(top_k, int((mask == 0.0).sum().item()))
            if k <= 0:
                symbol = int(logits.argmax().item())
            else:
                top = torch.topk(logits, k)
                picked = torch.multinomial(
                    torch.softmax(top.values / 1.0, dim=-1), 1, generator=generator
                ).item()
                symbol = int(top.indices[picked].item())
        else:
            symbol = int(logits.argmax().item())
        if symbol == taiji.config.boundary_symbol:
            break
        out.append(symbol)
        remaining, _ = _advance(remaining, symbol)
        step = taiji.observe(
            symbol, learn=False, readout="predictive", use_memory=False, use_identity=False
        )
    raw = bytes(out)
    for cut in range(len(raw), max(0, len(raw) - 4), -1):
        try:
            raw[:cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
        return raw[:cut]
    return raw


def _bigram_repetition(s: str) -> float:
    if len(s) < 2:
        return 0.0
    bigrams = [s[i : i + 2] for i in range(len(s) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams)


def corpus_byte_distribution(limit: int | None = None) -> Counter:
    hist: Counter = Counter()
    with CORPUS.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            if limit is not None and index >= limit:
                break
            try:
                text = json.loads(raw).get("text") or ""
            except ValueError:
                continue
            hist.update(text.encode("utf-8"))
    return hist


def total_variation(sample_bytes: Counter, reference: Counter) -> float:
    s_total, r_total = sum(sample_bytes.values()), sum(reference.values())
    if not s_total or not r_total:
        return 1.0
    keys = set(sample_bytes) | set(reference)
    return 0.5 * sum(
        abs(sample_bytes.get(k, 0) / s_total - reference.get(k, 0) / r_total) for k in keys
    )


def sign_test(pairs: list[tuple[int, int]]) -> dict:
    """逐题配对：pairs = (现状臂成句, 替代臂成句)；只用不一致的对做双侧精确符号检验。"""
    b = sum(1 for cur, alt in pairs if alt > cur)
    c = sum(1 for cur, alt in pairs if alt < cur)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n": 0, "p_two_sided": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return {"b": b, "c": c, "n": n, "p_two_sided": round(min(1.0, 2 * tail), 6)}


def main() -> int:
    from api.seed_runtime import SeedRuntime

    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    def items_of(dim: str):
        return eval_set["dimensions"][dim]["items"]

    tasks: list[tuple[str, str]] = []
    for dim in ("B", "G"):
        for item in items_of(dim):
            for turn in item.get("turns") or []:
                prompt = turn.get("prompt") if isinstance(turn, dict) else turn
                if prompt:
                    tasks.append((str(item["id"]), str(prompt)))
    if not tasks:
        raise SystemExit("评价集里没取到任何题面：字段名与预期不符，拒跑而不是静默产出空报告")

    runtime = SeedRuntime.load(CHECKPOINT)
    taiji = runtime.model.substrate

    results = {arm: {} for arm in ARMS}
    for item_id, prompt in tasks:
        key = f"{item_id}|{prompt[:24]}"
        for arm in ARMS:
            text = generate(taiji, _serialize_prompt(runtime, prompt), arm).decode(
                "utf-8", errors="replace"
            )
            results[arm][key] = text

    reference = corpus_byte_distribution(limit=200000)
    ngram = build_ngram_model()
    controls = assert_criterion_discriminates(ngram)
    per_arm = {}
    for arm, rows in results.items():
        texts = list(rows.values())
        bytes_hist: Counter = Counter()
        for t in texts:
            bytes_hist.update(t.encode("utf-8"))
        chars = [c for t in texts for c in t]
        well = sum(1 for t in texts if well_formed(t, ngram))
        per_arm[arm] = {
            "n": len(texts),
            "well_formed_rate": round(well / len(texts), 4),
            "mean_len_chars": round(statistics.mean(len(t) for t in texts), 2),
            "distinct_chars": len(set(chars)),
            "single_char_share_median": round(
                statistics.median(
                    Counter("".join(t.split())).most_common(1)[0][1]
                    / max(len("".join(t.split())), 1)
                    for t in texts
                ),
                4,
            ),
            "bigram_repetition_median": round(
                statistics.median(_bigram_repetition("".join(t.split())) for t in texts), 4
            ),
            "tv_distance_to_corpus": round(total_variation(bytes_hist, reference), 4),
            "samples": texts[:3],
        }

    base = results["greedy"]
    verdicts = {}
    for arm in ARMS:
        if arm == "greedy":
            continue
        pairs = [
            (
                1 if well_formed(base[k], ngram) else 0,
                1 if well_formed(results[arm][k], ngram) else 0,
            )
            for k in base
        ]
        delta_pp = round(
            (per_arm[arm]["well_formed_rate"] - per_arm["greedy"]["well_formed_rate"]) * 100, 2
        )
        verdicts[arm] = {"delta_percentage_points": delta_pp, "sign_test": sign_test(pairs)}

    d1 = [
        a
        for a, v in verdicts.items()
        if v["delta_percentage_points"] >= 20 and v["sign_test"]["p_two_sided"] < 0.05
    ]
    #: D2 的原意是"与现状**相差不到** 5 个百分点"，必须取绝对值。
    #: 用有符号比较会把"显著更差"（delta = -23）误判成满足 D2 ⇒ 首跑就是这么出错一次。
    d2 = bool(verdicts) and all(abs(v["delta_percentage_points"]) < 5 for v in verdicts.values())
    worse = [
        a
        for a, v in verdicts.items()
        if v["delta_percentage_points"] <= -5 and v["sign_test"]["p_two_sided"] < 0.05
    ]
    e1 = all(
        v["tv_distance_to_corpus"] >= 0.5 and v["distinct_chars"] <= 20 for v in per_arm.values()
    )
    e2 = all(v["tv_distance_to_corpus"] < 0.2 for v in per_arm.values()) and all(
        v["well_formed_rate"] < 0.05 for v in per_arm.values()
    )

    payload = {
        "format": "taiji-r2-surface-decode-p1p2-v1",
        "contract": "plans/reference/M5_R2_SURFACE_DECODING_CONTRACT_DRAFT_20260920.md",
        "checkpoint": "checkpoints/seed_beta.pt",
        "chain": {"relax_legacy_guard": "n/a (直接走 Taiji.observe)", "readout": "predictive"},
        "frozen_thresholds": {
            "bigram_upper_bound": BIGRAM_UPPER_BOUND,
            "min_len_chars": MIN_LEN,
            "max_single_char_share": MAX_SINGLE_SHARE,
            "gen_length_bytes": GEN_LENGTH,
            "temperature_seed": TEMPERATURE_SEED,
            "seed_note": "冻结合同未规定采样种子；本件补固定种子以满足可复现，属披露而非改线",
        },
        "criterion_controls": {
            **controls,
            "note": "反向对照门：真中文须过、打乱中文须挂；不过则整件拒跑。首跑的"
            "四条件合取判据正是被这道对照否证的（打乱样本 100% 通过）",
        },
        "supersedes_run": "首跑（四条件合取判据）的 D1/D2 判定已撤回，不得引用",
        "arms_run": list(ARMS),
        "arms_not_run": {"beam2": "未实现（需快照分支状态），不假装跑过"},
        "task_count": len(tasks),
        "per_arm": per_arm,
        "paired_vs_greedy": verdicts,
        "judgment": {
            "D1_decoding_improves": {"arms": d1, "holds": bool(d1)},
            "D2_decoding_exonerated": d2,
            "arms_significantly_worse_than_greedy": worse,
            "E1_collapse": e1,
            "E2_distribution_ok_structure_wrong": e2,
        },
        "claim_limits": [
            "成句率提升只说明表层解码形态改善，不构成任何语言回答能力主张",
            "该读出面是从未独立训练的运动面副本（P0-a′），本件不为其可训性辩护",
            "beam-2 未跑 ⇒ 不得写成解码策略空间已被穷尽",
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "judgment": payload["judgment"],
                "rates": {a: per_arm[a]["well_formed_rate"] for a in ARMS},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
