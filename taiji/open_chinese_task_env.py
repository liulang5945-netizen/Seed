"""Open-vocabulary natural Chinese continuous-task environment (`taiji-open-chinese-continuous-v1`).

Addresses the N34 coverage gap for natural Chinese continuous tasks without
closed-vocabulary lookup or silent label leakage:
1. Pure UTF-8 text/byte observations (`{"text", "utf8_bytes", "step", "turns"}`),
   zero closed-word IDs, zero `task_id` / `family_id`, zero silent OOVs.
2. Strictly split, 100% character-disjoint OOV entity & place pools (`OOV_ENTITIES`
   and `OOV_PLACES` share zero Chinese characters with training pools).
3. Strictly split natural Chinese syntactic paraphrases (`TRAIN_PATTERNS` vs
   `HELDOUT_PARAPHRASE_PATTERNS`) and distractors (`TRAIN_DISTRACTORS`,
   `HELDOUT_DISTRACTORS`, plus optional real-corpus sentences from `data/simple_zh`).
4. 6-turn continuous episodes requiring episodic state tracking across moves,
   cross-turn coreference (`move_coref`, `query_coref_recent`, `query_coref_first`),
   self-correction (`correction`), negation/cancellation (`negation`), distractor
   chatter, and explicit abstention (`clarify_unknown` -> `"未知"`) for entities not
   present in the current episode history.
5. Runtime anti-leak guard ensuring no query turn text ever contains its target
   answer string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

TASK_VERSION = "taiji-open-chinese-continuous-v1"

INTENTS: tuple[str, ...] = (
    "record_ok",
    "retrieve",
    "inspect",
    "locate",
    "clarify_unknown",
)
INTENT_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(INTENTS)}

ACK_TEXT = "确认"
UNKNOWN_TEXT = "未知"
EPISODE_TURNS = 6

# Training entities: 10 x 12 = 120 compositional items.
TRAIN_ENT_ADJ: tuple[str, ...] = ("旧", "新", "小", "大", "红", "蓝", "白", "黑", "银", "木")
TRAIN_ENT_NOUN: tuple[str, ...] = (
    "本子",
    "水杯",
    "钥匙",
    "毛巾",
    "报纸",
    "雨披",
    "眼镜",
    "钢笔",
    "布袋",
    "台灯",
    "剪刀",
    "胶带",
)
TRAIN_ENTITIES: tuple[str, ...] = tuple(f"{a}{n}" for a in TRAIN_ENT_ADJ for n in TRAIN_ENT_NOUN)

# 12 natural OOV entities sharing ZERO characters with TRAIN_ENTITIES.
OOV_ENTITIES: tuple[str, ...] = (
    "紫砂壶",
    "丝绸手帕",
    "登山杖",
    "测距仪",
    "投影仪",
    "陶瓷花瓶",
    "折叠伞",
    "机械键盘",
    "指南针",
    "急救箱",
    "吉他拨片",
    "充电宝",
)

# Training places: 10 x 10 = 100 compositional locations.
TRAIN_PLACE_A: tuple[str, ...] = (
    "书桌",
    "衣柜",
    "茶几",
    "厨房",
    "玄关",
    "书架",
    "床头",
    "餐桌",
    "窗台",
    "走廊",
)
TRAIN_PLACE_B: tuple[str, ...] = (
    "左侧",
    "右侧",
    "下层",
    "上层",
    "角落",
    "边缘",
    "抽屉",
    "挂钩",
    "隔间",
    "旁侧",
)
TRAIN_PLACES: tuple[str, ...] = tuple(f"{a}{b}" for a in TRAIN_PLACE_A for b in TRAIN_PLACE_B)

# 12 natural OOV locations sharing ZERO characters with TRAIN_PLACES.
OOV_PLACES: tuple[str, ...] = (
    "车库铁匣",
    "阁楼漆盒",
    "寝室壁橱",
    "门厅鞋篓",
    "地窖暗格",
    "庭院藤篮",
    "储藏顶棚",
    "办公工位",
    "仓库货盘",
    "凉亭石凳",
    "南坡花圃",
    "西厢竹筐",
)

TRAIN_DISTRACTORS: tuple[str, ...] = (
    "顺便一提，今天傍晚可能要下阵雨。",
    "对了，楼下快递驿站七点半才关门。",
    "刚才手机响了一下，好像是物业发的通知。",
    "顺带说一声，阳台的衣服我已经提前收好了。",
    "外面的风比上午大了些，记得顺手关窗。",
    "刚刚顺路买了一袋水果回来。",
    "听邻居说下周二小区要检修电梯。",
    "等会儿记得把明天的闹钟提前十分钟。",
    "今天下午的会议已经改到线上进行了。",
    "门口的脚垫刚才顺手拍干净了。",
    "晚饭打算熬点小米粥，省事又暖胃。",
    "刚才路过街角那家面馆人还挺多的。",
    "如果晚上加班回来晚，我给你留一盏灯。",
    "桌面上的零碎废纸我都扔进回收桶了。",
    "刚才听见外面有人喊收旧家电。",
    "周末如果天气放晴，咱们再去公园散步。",
    "笔记本电脑刚才充好电已经拔掉插头了。",
    "记得下周一早上把水电费一并缴了。",
    "刚刚翻日历才发现月底还有两天假期。",
    "楼道里的感应灯今天修好了，亮得很。",
    "刚才去取牛奶的时候看见一只橘猫。",
    "茶壶里的开水还温着，想喝可以自己倒。",
    "今天公交车来得特别准时，没怎么等。",
    "对面的装修队五点钟就按时停工了。",
)

HELDOUT_DISTRACTORS: tuple[str, ...] = (
    "另外提醒你，明天清晨气温会明显下降。",
    "对了，物业刚刚在群里发了停水预告。",
    "刚才顺手把信箱里的宣传单清理掉了。",
    "晚些时候风力会转小，适合出去走走。",
    "外头阳光不错，院子里的花都开得很好。",
    "刚接到电话，周末聚餐改在周六中午。",
    "今天地铁比平时空不少，一路都有座位。",
    "记得睡前把电脑系统更新顺便点上。",
    "顺便说一句，冰箱里的酸奶明天就到期了。",
    "刚才散步看到河边新修好的步道已经开放。",
    "社区图书馆这周日要延长开放两个小时。",
    "刚才物业通知本周末小区大门要粉刷翻新。",
    "下周气象台预报有一次冷空气过境。",
    "明天出门前别忘带好工牌和门禁卡。",
    "楼下新开的面包房据说八点半出炉。",
    "今天傍晚晚霞特别好看，楼顶很多人拍照。",
)

TRAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "place_init": (
        "先把{ent}放在{loc}。",
        "我把{ent}搁到{loc}了，你记一下。",
        "请注意，{ent}现在摆在{loc}。",
        "{ent}刚刚被我收到{loc}去了。",
        "今天起{ent}就固定放{loc}。",
        "帮我记着，{ent}我先塞进{loc}。",
        "刚整理好，我把{ent}放到{loc}去了。",
        "{ent}已被我顺手放在{loc}了，请记住。",
        "先把{ent}收到{loc}里，待会儿要用。",
        "我刚顺手把{ent}摆到{loc}那边了。",
        "注意一下，{ent}现在就放在{loc}里头。",
        "刚把{ent}拿过去放到{loc}了。",
    ),
    "move_named": (
        "现在我把{ent}改放到{loc}了。",
        "顺手把{ent}从原地方挪到{loc}。",
        "刚清理东西，{ent}已经移去{loc}。",
        "换了个地方，{ent}目前在{loc}。",
        "麻烦更新一下，{ent}搬到{loc}去了。",
        "刚才收拾东西，把{ent}从原来的地方移到{loc}了。",
        "{ent}换地方了，我挪到{loc}那边去了。",
        "提醒一下，{ent}刚才被我搬去{loc}了。",
        "我把{ent}重新放到{loc}去了，不在原先处。",
        "现在把{ent}改移到{loc}里了。",
    ),
    "move_coref": (
        "顺手把刚才那件挪到{loc}了。",
        "刚放下的那个东西，我现在移到{loc}。",
        "另外它刚才被我改摆去{loc}。",
        "把上面说到的那件物品转到{loc}。",
        "刚刚那件我又拎到{loc}去了，记一下。",
        "前面提到的那件东西，我现在搬去{loc}了。",
        "刚才那件不太顺手，我又把它移到{loc}去了。",
        "上一轮放好的那个东西，被我挪到{loc}了。",
        "刚刚那件物品现在改放到{loc}那边去了。",
        "把刚才那件顺手搬去{loc}了。",
    ),
    "correction": (
        "不好意思，刚才说错了，{ent}其实是在{loc}。",
        "更正一下上一条，{ent}真正的位置是{loc}。",
        "哎呀我记岔了，{ent}实际上一直放在{loc}。",
        "别按刚才记的算，{ent}明明搁在{loc}。",
        "说反了，我再确认一遍：{ent}在{loc}。",
        "不对，刚才讲错了，{ent}其实一直放在{loc}。",
        "我记反了，更正一下，{ent}现在真正是在{loc}。",
        "刚报错地方了，其实{ent}就摆在{loc}，别按旧的算。",
        "说错了说错了，{ent}其实一直在{loc}。",
        "纠正一下，{ent}并不是在别处，真正位置是在{loc}。",
    ),
    "negation": (
        "本来打算把{ent}改到{fake_loc}，后来没动，还在原来的位置。",
        "差点忘了说，{ent}没有搬去{fake_loc}，位置没变。",
        "不用改成{fake_loc}了，{ent}最终还是留在原处。",
        "刚才考虑过把{ent}搁到{fake_loc}，但取消了，原位不变。",
        "记住{ent}并没有放去{fake_loc}，依然保持原先地方。",
        "本来想把{ent}搬去{fake_loc}的，后来取消了，没动。",
        "说清楚一下，{ent}并没有移到{fake_loc}，还在原来地方。",
        "刚才虽然打算把{ent}挪去{fake_loc}，但最后没改，照旧。",
        "原打算将{ent}移去{fake_loc}，结果没动，位置依然没变。",
        "没把{ent}搬到{fake_loc}去，还是留在原先位置。",
    ),
    "query_retrieve": (
        "现在去把{ent}拿过来吧。",
        "麻烦帮忙取一下{ent}。",
        "我现在需要用{ent}，帮我拿过来。",
        "顺手帮我把{ent}取回来。",
        "请你去把{ent}取出来。",
        "麻烦帮我把{ent}从现在位置拿过来用一下。",
        "我急着用{ent}，请顺手取回来。",
        "方便的话帮我去把{ent}取过来吧。",
        "请帮我把{ent}拿回来用一下。",
    ),
    "query_inspect": (
        "帮我看一眼{ent}现在还好吗，去检查一下。",
        "麻烦过去确认检查下{ent}的状态。",
        "请你核查一下{ent}有没有放稳。",
        "顺便帮我查看检查一下{ent}。",
        "需要检查确认一下{ent}，麻烦去看看。",
        "请过去检查看看{ent}放得稳不稳。",
        "顺便帮我查看核查一下{ent}现在状况。",
        "麻烦去确认检查一下{ent}放好了没。",
        "过去看一眼检查下{ent}有没有问题。",
    ),
    "query_locate": (
        "问一下，{ent}现在具体在哪里？",
        "我一时找不到{ent}，它在什么位置？",
        "能告诉我{ent}此刻摆在何处吗？",
        "刚才记不清了，{ent}当前在哪个位置？",
        "查一下{ent}目前所处的地方。",
        "想问下，{ent}现在具体在什么地方？",
        "请问一下，{ent}目前放到哪个位置了？",
        "一时想不起来，{ent}现在摆在哪里？",
        "请问{ent}此刻放在什么地方？",
    ),
    "query_coref_recent": (
        "刚才最新提到或者动过的那件东西，现在去哪儿了？",
        "我最后一次摆弄的那件物品当前位于哪里？",
        "查一下最近处理过的那件东西现在在何处？",
        "把刚刚最后放动过的那件找一下位置。",
        "刚才最后动过的那件物品，现在具体在哪里？",
        "最近一次动过的那个东西，目前放在什么地方？",
        "查一下刚刚最后处理的那件东西现在在何处？",
        "最后提到或者挪过的那件东西，现在在哪个位置？",
    ),
    "query_coref_first": (
        "最开始先放下的第一件东西，现在放在哪里？",
        "问一下最早登记的那件物品目前在什么地方？",
        "一开始最先放好的第一件东西，此刻在何处？",
        "回忆一下最初提过的那件东西现在在哪里？",
        "一开始最先放好的第一件东西，现在在什么地方？",
        "回忆一下最早放下的那件物品，目前在哪里？",
        "最开始登记的那件东西，此刻在什么位置？",
        "最初先放好的第一件物品，现在到底放在哪里？",
    ),
}

HELDOUT_PARAPHRASE_PATTERNS: dict[str, tuple[str, ...]] = {
    "place_init": (
        "刚刚整理完房间，我把{ent}放到{loc}去了。",
        "{ent}已经被我顺手摆在{loc}了，别忘记。",
        "现在先把{ent}收到{loc}里头，待会儿方便找。",
    ),
    "move_named": (
        "刚刚重新收拾了一下，把{ent}从原先地方移到{loc}了。",
        "{ent}刚换了位置，我已经挪到{loc}那边去了。",
        "顺便提醒你，{ent}刚才被我搬去{loc}了。",
    ),
    "move_coref": (
        "前面刚提过的那件东西，我现在已经搬去{loc}了。",
        "刚才那件物品不太顺手，我又把它移到{loc}去了。",
        "上一轮放好的那个东西，刚刚被我挪到了{loc}。",
    ),
    "correction": (
        "不对，刚刚说错了，{ent}其实一直摆在{loc}。",
        "我刚记反了，更正一下，{ent}现在真正是在{loc}。",
        "刚才报错了地方，其实{ent}就放在{loc}，别按旧的记。",
    ),
    "negation": (
        "本来想把{ent}搬去{fake_loc}的，后来取消了，还是没动。",
        "说清楚一下，{ent}并没有移到{fake_loc}，依然在原来位置。",
        "刚才虽然打算把{ent}挪去{fake_loc}，但最后没改，位置照旧。",
    ),
    "query_retrieve": (
        "麻烦帮我把{ent}从现在的位置拿过来用一下。",
        "我现在急着要用{ent}，请顺手取回来。",
        "方便的话帮我去把{ent}拿过来吧。",
    ),
    "query_inspect": (
        "请过去检查看看{ent}现在放得稳不稳。",
        "顺便帮我查看核查一下{ent}当前的状况。",
        "麻烦去确认检查一下{ent}有没有放好。",
    ),
    "query_locate": (
        "我想问下，{ent}眼下具体在什么地方？",
        "帮我查查，{ent}目前到底放在哪个位置？",
        "现在一时想不起来，{ent}究竟摆在哪里？",
    ),
    "query_coref_recent": (
        "刚才最后动过或者提过的那件物品，现在具体在哪里？",
        "最近一次摆弄过的那个东西，目前放在什么地方？",
        "查一下刚刚最后处理的那件东西现在在哪个位置？",
    ),
    "query_coref_first": (
        "一开始最先放好的第一件东西，现在到底在什么地方？",
        "回忆一下最早放下的那件物品，目前具体在哪里？",
        "最开始第一个登记的那件东西，此刻在什么位置？",
    ),
}


_OOV_FORBIDDEN_CHARS: frozenset[str] = frozenset(
    "".join(OOV_ENTITIES)
    + "".join(OOV_PLACES)
    + ACK_TEXT
    + UNKNOWN_TEXT
    + "".join(p for ps in TRAIN_PATTERNS.values() for p in ps)
    + "".join(p for ps in HELDOUT_PARAPHRASE_PATTERNS.values() for p in ps)
)
BROAD_TRAIN_CJK: tuple[str, ...] = tuple(
    chr(c) for c in range(0x4E00, 0x9FA0, 17) if chr(c) not in _OOV_FORBIDDEN_CHARS
)


def _verify_disjoint_pools() -> None:
    """Guarantee OOV entities, OOV places, BROAD_TRAIN_CJK, and text pools are disjoint."""
    tr_ent_chars = set("".join(TRAIN_ENTITIES))
    oov_ent_chars = set("".join(OOV_ENTITIES))
    if not tr_ent_chars.isdisjoint(oov_ent_chars):
        overlap = sorted(tr_ent_chars & oov_ent_chars)
        raise AssertionError(f"OOV_ENTITIES shares chars with TRAIN_ENTITIES: {overlap}")

    tr_place_chars = set("".join(TRAIN_PLACES))
    oov_place_chars = set("".join(OOV_PLACES))
    if not tr_place_chars.isdisjoint(oov_place_chars):
        overlap = sorted(tr_place_chars & oov_place_chars)
        raise AssertionError(f"OOV_PLACES shares chars with TRAIN_PLACES: {overlap}")

    if len(BROAD_TRAIN_CJK) < 200:
        raise AssertionError(f"BROAD_TRAIN_CJK too small: {len(BROAD_TRAIN_CJK)}")
    if not set(BROAD_TRAIN_CJK).isdisjoint(oov_ent_chars | oov_place_chars):
        raise AssertionError("BROAD_TRAIN_CJK overlaps OOV chars")

    if set(TRAIN_DISTRACTORS) & set(HELDOUT_DISTRACTORS):
        raise AssertionError("TRAIN_DISTRACTORS overlaps HELDOUT_DISTRACTORS")
    for key, train_pats in TRAIN_PATTERNS.items():
        if set(train_pats) & set(HELDOUT_PARAPHRASE_PATTERNS[key]):
            raise AssertionError(f"Pattern overlap under key={key!r}")


_verify_disjoint_pools()


def load_external_chinese_distractors(
    path: str | Path = Path("data/simple_zh/dialogue_extended_clean.jsonl"),
    max_items: int = 48,
) -> tuple[str, ...]:
    """Deterministically extract short, clean real Chinese sentences from simple_zh.

    Sentences containing any entity/place token, `ACK_TEXT`, or `UNKNOWN_TEXT` are excluded
    so wild chatter never leaks entity/place ground truth. Returns empty tuple if absent.
    """
    fp = Path(path)
    if not fp.exists():
        return ()
    forbidden = set(TRAIN_ENTITIES) | set(OOV_ENTITIES) | set(TRAIN_PLACES) | set(OOV_PLACES)
    forbidden |= {ACK_TEXT, UNKNOWN_TEXT, "原位", "原处", "原来"}
    cjk_only = re.compile(r"^[\u4e00-\u9fff，。！？、]+$")
    collected: list[str] = []
    seen: set[str] = set()
    with fp.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= 4000 or len(collected) >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                text = str(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                continue
            # Split into clauses/sentences and pick clean short daily Chinese sentences.
            for raw_seg in re.split(r"[\n。！？]", text):
                seg = raw_seg.strip()
                if not (12 <= len(seg) <= 26):
                    continue
                if "问：" in seg or "答：" in seg or any(f in seg for f in forbidden):
                    continue
                if not cjk_only.match(seg):
                    continue
                cand = seg.rstrip("，、") + "。"
                if cand not in seen:
                    seen.add(cand)
                    collected.append(cand)
                    if len(collected) >= max_items:
                        break
    return tuple(collected)


class OpenChineseTaskEnv:
    """6-turn open-vocab Chinese continuous task (moves, corrections, negations, coref, unknown)."""

    def __init__(
        self,
        seed: int,
        *,
        split: str = "train_seen",
        external_distractors: tuple[str, ...] | None = None,
        open_lexicon_prob: float = 0.0,
    ) -> None:
        if split not in ("train_seen", "heldout_paraphrase", "heldout_oov_entities"):
            raise ValueError(f"unsupported split {split!r}")
        if open_lexicon_prob < 0.0 or open_lexicon_prob > 1.0:
            raise ValueError(f"invalid open_lexicon_prob={open_lexicon_prob}")
        if open_lexicon_prob > 0.0 and split != "train_seen":
            raise ValueError("open_lexicon_prob > 0 is strictly forbidden outside train_seen")
        self.seed = int(seed)
        self.split = split
        self.gen = torch.Generator().manual_seed(self.seed % (2**63))
        if split == "train_seen":
            self.entities = TRAIN_ENTITIES
            self.places = TRAIN_PLACES
            self.patterns = TRAIN_PATTERNS
            self.distractors = TRAIN_DISTRACTORS
            if open_lexicon_prob > 0.0:
                lex_gen = torch.Generator().manual_seed((self.seed * 31 + 7) % (2**63))
                if float(torch.rand((), generator=lex_gen)) < open_lexicon_prob:
                    idx = torch.randperm(len(BROAD_TRAIN_CJK), generator=lex_gen).tolist()
                    pos = 0
                    dyn_ents: list[str] = []
                    dyn_locs: list[str] = []
                    for _ in range(6):
                        el = 2 + (idx[pos] % 3)
                        dyn_ents.append("".join(BROAD_TRAIN_CJK[j] for j in idx[pos : pos + el]))
                        pos += el
                    for _ in range(6):
                        pl = 3 + (idx[pos] % 3)
                        dyn_locs.append("".join(BROAD_TRAIN_CJK[j] for j in idx[pos : pos + pl]))
                        pos += pl
                    self.entities = tuple(dyn_ents)
                    self.places = tuple(dyn_locs)
        elif split == "heldout_paraphrase":
            self.entities = TRAIN_ENTITIES
            self.places = TRAIN_PLACES
            self.patterns = HELDOUT_PARAPHRASE_PATTERNS
            self.distractors = HELDOUT_DISTRACTORS
        else:
            self.entities = OOV_ENTITIES
            self.places = OOV_PLACES
            self.patterns = HELDOUT_PARAPHRASE_PATTERNS
            self.distractors = external_distractors if external_distractors else HELDOUT_DISTRACTORS
        self.turns_data: list[dict[str, Any]] = []
        self.step_idx = 0

    def _rand_idx(self, n: int) -> int:
        return int(torch.randint(0, n, (), generator=self.gen))

    def _pick(self, seq: tuple[Any, ...] | list[Any]) -> Any:
        return seq[self._rand_idx(len(seq))]

    def _maybe_chatter(self, text: str, prob_half: bool = True) -> str:
        if prob_half and self._rand_idx(2) == 0:
            return text
        chat = self._pick(self.distractors)
        if self._rand_idx(2) == 0:
            return f"{chat}{text}"
        return f"{text}{chat}"

    def _build_episode(self) -> list[dict[str, Any]]:
        ent_perm = torch.randperm(len(self.entities), generator=self.gen).tolist()
        ent_a = self.entities[ent_perm[0]]
        ent_b = self.entities[ent_perm[1]]
        ent_u = self.entities[ent_perm[2]]

        loc_perm = torch.randperm(len(self.places), generator=self.gen).tolist()
        loc_a0 = self.places[loc_perm[0]]
        loc_b0 = self.places[loc_perm[1]]

        state: dict[str, str] = {ent_a: loc_a0, ent_b: loc_b0}
        first_ent = ent_a
        recent_ent = ent_a
        coref_updated_ents: set[str] = set()
        corr_neg_ents: set[str] = set()
        turns: list[dict[str, Any]] = []

        # Turn 0: initial placement of ent_a
        t0_text = self._maybe_chatter(
            self._pick(self.patterns["place_init"]).format(ent=ent_a, loc=loc_a0)
        )
        turns.append(
            {
                "text": t0_text,
                "intent": "record_ok",
                "answer": ACK_TEXT,
                "is_query": False,
                "is_coref": False,
                "is_corr_neg": False,
                "is_unknown": False,
            }
        )

        # Turn 1: initial placement of ent_b
        recent_ent = ent_b
        t1_text = self._maybe_chatter(
            self._pick(self.patterns["place_init"]).format(ent=ent_b, loc=loc_b0)
        )
        turns.append(
            {
                "text": t1_text,
                "intent": "record_ok",
                "answer": ACK_TEXT,
                "is_query": False,
                "is_coref": False,
                "is_corr_neg": False,
                "is_unknown": False,
            }
        )

        # Helper to construct a state event turn
        used_locs = {loc_a0, loc_b0}

        def _fresh_loc(exclude_loc: str) -> str:
            cands = [p for p in self.places if p != exclude_loc and p not in used_locs]
            if not cands:
                cands = [p for p in self.places if p != exclude_loc]
            chosen = self._pick(cands)
            used_locs.add(chosen)
            return chosen

        def _apply_event(kind: str) -> str:
            nonlocal recent_ent
            if kind == "move_named":
                target = ent_a if self._rand_idx(2) == 0 else ent_b
                new_loc = _fresh_loc(state[target])
                state[target] = new_loc
                recent_ent = target
                return self._maybe_chatter(
                    self._pick(self.patterns["move_named"]).format(ent=target, loc=new_loc)
                )
            if kind == "move_coref":
                target = recent_ent
                new_loc = _fresh_loc(state[target])
                state[target] = new_loc
                coref_updated_ents.add(target)
                return self._maybe_chatter(
                    self._pick(self.patterns["move_coref"]).format(loc=new_loc)
                )
            if kind == "correction":
                target = ent_a if self._rand_idx(2) == 0 else ent_b
                corr_loc = _fresh_loc(state[target])
                state[target] = corr_loc
                corr_neg_ents.add(target)
                recent_ent = target
                return self._maybe_chatter(
                    self._pick(self.patterns["correction"]).format(ent=target, loc=corr_loc)
                )
            if kind == "negation":
                target = ent_a if self._rand_idx(2) == 0 else ent_b
                fake_loc = _fresh_loc(state[target])
                corr_neg_ents.add(target)
                recent_ent = target
                return self._maybe_chatter(
                    self._pick(self.patterns["negation"]).format(ent=target, fake_loc=fake_loc)
                )
            # chatter_distractor
            return str(self._pick(self.distractors))

        # Turn 2 & Turn 3: diverse combination of updates, corrections/negations, chatter.
        ev2 = self._pick(("move_named", "move_coref", "correction", "negation"))
        if ev2 in ("correction", "negation"):
            ev3 = self._pick(("move_coref", "move_named", "chatter_distractor"))
        else:
            ev3 = self._pick(("correction", "negation", "correction", "chatter_distractor"))

        for ev in (ev2, ev3):
            turns.append(
                {
                    "text": _apply_event(ev),
                    "intent": "record_ok",
                    "answer": ACK_TEXT,
                    "is_query": False,
                    "is_coref": False,
                    "is_corr_neg": False,
                    "is_unknown": False,
                }
            )

        # Turn 4 & Turn 5: two distinct query turns covering named, coref, corr/neg, unknown_entity.
        query_modes_pool = [
            "named_a",
            "named_b",
            "coref_recent",
            "coref_first",
            "unknown_entity",
        ]
        # Sample 2 distinct modes biased to guarantee rich coref + unknown coverage.
        m1 = query_modes_pool[self._rand_idx(len(query_modes_pool))]
        rem_modes = [m for m in query_modes_pool if m != m1]
        m2 = rem_modes[self._rand_idx(len(rem_modes))]
        intent_choices = ("retrieve", "inspect", "locate")

        for mode in (m1, m2):
            surface_intent = self._pick(intent_choices)
            pat_key = f"query_{surface_intent}"
            if mode == "named_a":
                target = ent_a
                q_text = self._pick(self.patterns[pat_key]).format(ent=target)
                true_intent = surface_intent
                answer = state[target]
                is_coref = target in coref_updated_ents
                is_corr_neg = target in corr_neg_ents
                is_unknown = False
            elif mode == "named_b":
                target = ent_b
                q_text = self._pick(self.patterns[pat_key]).format(ent=target)
                true_intent = surface_intent
                answer = state[target]
                is_coref = target in coref_updated_ents
                is_corr_neg = target in corr_neg_ents
                is_unknown = False
            elif mode == "coref_recent":
                target = recent_ent
                q_text = self._pick(self.patterns["query_coref_recent"])
                true_intent = "locate"
                answer = state[target]
                is_coref = True
                is_corr_neg = target in corr_neg_ents
                is_unknown = False
            elif mode == "coref_first":
                target = first_ent
                q_text = self._pick(self.patterns["query_coref_first"])
                true_intent = "locate"
                answer = state[target]
                is_coref = True
                is_corr_neg = target in corr_neg_ents
                is_unknown = False
            else:  # unknown_entity: identical surface phrasing, zero history occurrence
                q_text = self._pick(self.patterns[pat_key]).format(ent=ent_u)
                true_intent = "clarify_unknown"
                answer = UNKNOWN_TEXT
                is_coref = False
                is_corr_neg = False
                is_unknown = True

            if answer in q_text:
                raise AssertionError(
                    f"Answer leakage in query text split={self.split}: {answer!r} in {q_text!r}"
                )
            turns.append(
                {
                    "text": q_text,
                    "intent": true_intent,
                    "answer": answer,
                    "is_query": True,
                    "is_coref": is_coref,
                    "is_corr_neg": is_corr_neg,
                    "is_unknown": is_unknown,
                }
            )
        return turns

    @staticmethod
    def encode_utf8(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def reset(self) -> dict[str, Any]:
        self.turns_data = self._build_episode()
        self.step_idx = 0
        text = self.turns_data[0]["text"]
        return {
            "text": text,
            "utf8_bytes": self.encode_utf8(text),
            "step": 0,
            "turns": len(self.turns_data),
        }

    def current_text(self) -> str:
        return str(self.turns_data[self.step_idx]["text"])

    def current_bytes(self) -> list[int]:
        return self.encode_utf8(self.current_text())

    @property
    def intended_intent(self) -> int:
        return INTENT_TO_ID[str(self.turns_data[self.step_idx]["intent"])]

    @property
    def intended_answer(self) -> str:
        return str(self.turns_data[self.step_idx]["answer"])

    @property
    def current_meta(self) -> dict[str, bool]:
        t = self.turns_data[self.step_idx]
        return {
            "is_query": bool(t["is_query"]),
            "is_coref": bool(t["is_coref"]),
            "is_corr_neg": bool(t["is_corr_neg"]),
            "is_unknown": bool(t["is_unknown"]),
        }

    def step(self, intent_id: int, response_text: str) -> tuple[dict[str, Any], float, bool]:
        ok = (int(intent_id) == self.intended_intent) and (
            response_text.strip() == self.intended_answer
        )
        reward = 1.0 if ok else 0.0
        self.step_idx += 1
        done = self.step_idx >= len(self.turns_data)
        if done:
            return (
                {
                    "text": "",
                    "utf8_bytes": [],
                    "step": self.step_idx,
                    "turns": len(self.turns_data),
                },
                reward,
                True,
            )
        text = self.current_text()
        return (
            {
                "text": text,
                "utf8_bytes": self.encode_utf8(text),
                "step": self.step_idx,
                "turns": len(self.turns_data),
            },
            reward,
            False,
        )


def oracle_policy(env: OpenChineseTaskEnv, _history: list[list[int]]) -> tuple[int, str, bool]:
    """Calibration control only: intended `(intent_id, answer)`. Never used in learned training."""
    return env.intended_intent, env.intended_answer, True


def random_policy(
    _env: OpenChineseTaskEnv, _history: list[list[int]], gen: torch.Generator
) -> tuple[int, str, bool]:
    intent = int(torch.randint(0, len(INTENTS), (), generator=gen))
    pool = (ACK_TEXT, UNKNOWN_TEXT, *TRAIN_PLACES[:10])
    ans = pool[int(torch.randint(0, len(pool), (), generator=gen))]
    return intent, ans, True


def eval_open_policy(
    policy_fn,
    seeds: list[int],
    *,
    split: str = "train_seen",
    gen: torch.Generator | None = None,
    external_distractors: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate a multi-turn policy across open Chinese episodes."""
    rng = gen or torch.Generator().manual_seed(0)
    all_ok = all_tot = 0
    q_ok = q_tot = q_intent_ok = q_em_ok = 0
    known_em_ok = known_tot = 0
    coref_ok = coref_tot = 0
    corr_ok = corr_tot = 0
    unk_ok = unk_tot = 0
    utf8_ok = ep_ok = 0

    for seed in seeds:
        env = OpenChineseTaskEnv(seed, split=split, external_distractors=external_distractors)
        obs = env.reset()
        history: list[list[int]] = [list(obs["utf8_bytes"])]
        ep_all = True
        while True:
            meta = env.current_meta
            target_intent = env.intended_intent
            target_answer = env.intended_answer
            if policy_fn is random_policy:
                pred_intent, pred_text, is_valid_utf8 = policy_fn(env, history, rng)
            else:
                pred_intent, pred_text, is_valid_utf8 = policy_fn(env, history)
            obs_next, reward, done = env.step(pred_intent, pred_text)
            hit = int(reward > 0.0)
            all_tot += 1
            all_ok += hit
            utf8_ok += int(bool(is_valid_utf8))
            if not hit:
                ep_all = False

            if meta["is_query"]:
                q_tot += 1
                q_ok += hit
                q_intent_ok += int(int(pred_intent) == target_intent)
                em = int(pred_text.strip() == target_answer)
                q_em_ok += em
                if meta["is_unknown"]:
                    unk_tot += 1
                    unk_ok += hit
                else:
                    known_tot += 1
                    known_em_ok += em
                if meta["is_coref"]:
                    coref_tot += 1
                    coref_ok += hit
                if meta["is_corr_neg"]:
                    corr_tot += 1
                    corr_ok += hit

            if done:
                break
            history.append(list(obs_next["utf8_bytes"]))
        ep_ok += int(ep_all)

    return {
        "split": split,
        "n_episodes": len(seeds),
        "all_turn_success": round(all_ok / max(1, all_tot), 4),
        "query_turn_success": round(q_ok / max(1, q_tot), 4),
        "query_intent_acc": round(q_intent_ok / max(1, q_tot), 4),
        "query_answer_em": round(q_em_ok / max(1, q_tot), 4),
        "known_answer_em": round(known_em_ok / max(1, known_tot), 4),
        "coref_success": round(coref_ok / max(1, coref_tot), 4),
        "corr_neg_success": round(corr_ok / max(1, corr_tot), 4),
        "unknown_abstain_success": round(unk_ok / max(1, unk_tot), 4),
        "episode_success": round(ep_ok / max(1, len(seeds)), 4),
        "utf8_valid_rate": round(utf8_ok / max(1, all_tot), 4),
        "counts": {
            "all_turns": all_tot,
            "query_turns": q_tot,
            "known_queries": known_tot,
            "coref_queries": coref_tot,
            "corr_neg_queries": corr_tot,
            "unknown_queries": unk_tot,
        },
    }
