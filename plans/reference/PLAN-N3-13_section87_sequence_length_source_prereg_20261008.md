# PLAN-N3-13 · §8.7 第四项"序列长度"的取值面升版＝同源复算件（零训练跑量，判据先冻、不代签）

日期：2026-10-08 · 线别：N3 甲（×10 单点 scaling 探针） · 母件：[PLAN-N3-02](PLAN-N3-02_scaling_probe_prereg_20261007.md)
仪器：[recompute_taiji_n3a_sequence_length.py](../../scripts/training/recompute_taiji_n3a_sequence_length.py)
＋[adjudicate_taiji_n3a_scaling_probe.py](../../scripts/training/adjudicate_taiji_n3a_scaling_probe.py)（新增 `--seq-a/--seq-b`）
契约测：[test_n3a_sequence_length_recompute_contract.py](../../tests/taiji_native/test_n3a_sequence_length_recompute_contract.py)
＋[test_n3a_scaling_probe_adjudication_contract.py](../../tests/taiji_native/test_n3a_scaling_probe_adjudication_contract.py)（追加四支）

## 0. 为什么要有这一件

㊵-545 把 `sequence_length` 这一列落进了训练器的收尾自述，但**两臂在盘的产物是 ㊵-545 之前的代码写的**：
`output/n3a_armA/progress_exit.json` 与 `output/n3a_armB/progress_exit.json` 的键集里都没有这一列
（实测：两份都 `sequence_length=<absent>`）⇒ 判读器对它们仍按 PLAN-N3-02 §2 那句"缺任一项即整档只算跑了、
不算测了"锁在 `ran_not_measured`（㊵-543 取证：五项里唯一 `False` 的就是这一项）。

解锁只有两条路：

1. **重跑两臂**（各 ≈250,000 符号、≈2.3 h 墙钟，改权重训练 ⇒ 要批）；
2. **同源复算**：这一项是**取数层的性质**（哪些篇、每篇多少符号），与模型权重无关，
   可以用同一条 `iter_corpus_symbols` 原样重放补出来——㊵-494 立过这个先例
   （"事后只读仪器与当时算是同一口径"）。

本件走第 2 条，并把"什么样的一件才算同源"冻在这里。

## 1. 判据（冻）：三条并立守卫 ＋一条形状守卫

一条复算件**被认可是在场**当且仅当它自己出版 `status="ok"`，而 `status` 由这四条决定：

| 编号 | 守卫 | 取法 | 不齐时 |
| --- | --- | --- | --- |
| C-1 | 同语料 | 给定 `--corpus` 的 `corpus_fingerprint`（名＋字节，与训练器同一个函数）＝收尾件里那串 | `guard_failed` |
| C-2 | 同预算 | `--max-symbols` ＝收尾件的 `budget_max_symbols` | `guard_failed` |
| C-3 | 同一条流 | 重放出来的 `document_visits` ＝收尾件自述的 `document_visits`（计数器**只有那条生成器**会写，所以这条是"走的是同一条流"的直接证据） | `guard_failed` |
| C-4 | 吃满了 | 重放取到的符号数＝`--max-symbols` | `stream_shorter_than_budget` |

判读器侧（PLAN-N3-02 §2 的第四项）：**收尾件自述优先**；自述缺席时只认 `status=="ok"` 的复算件，
并把取值面写进 `sequence_length_source`（`exit_record` / `recomputed_same_stream` /
`sidecar_guard_failed:<status>` / `missing_sidecar` / `absent`）如实披露；缺件或守卫不齐**都不算在场**。

## 2. 这次动的与不动的

* **动的只有第四项的取值来源**（多认一件由 §1 四条守卫钉住的同源复算件）。
* **不动**：§2 的 `J-N3a` 阈值与合取形状、§0 的对照锚、§4quater 的取法式子
  （"由 `iter_corpus_symbols` 在边界符处累加，不许用 `max_symbols` 参数值冒充实测分布"——
  本件正是照这句实现并照它复算的）、PLAN-N3-10 §3 的读数边界、DEBT-G64 之后加的同源必检。
* 要动上面任一项仍走**升版＋索引标作废**那条流程，不许借本件顺手改。

## 3. 已知不可判别项（写进件里，不藏）

`end_boundary_after_newline` 这条旗标**不在收尾件的自述里**（收尾件只自述指纹／预算／篇数）。
本件默认按训练器 CLI 的默认值 `True` 复算：该旗标只有否定形式
`--no-end-boundary-after-newline`（`action="store_false"` 且未给 `default` ⇒ argparse 默认 True，
`train_seed_corpus.py:1058-1066`），而两臂冻过的命令串（PLAN-N3-10 §9）里没有它 ⇒ 取默认。
**代价**：每篇 ±1 符号；而 C-3 那条篇数守卫**分不出**这个 ±1（315 篇只挪约一篇的切点），
所以这里如实标为"不可判别"，不许把复算件说成逐位等于当时那条流。

## 4. 本轮实测（两臂各一条，零训练跑量）

| 臂 | C-1／C-2／C-3／吃满 | 重放篇数 vs 收尾件自述 | 序列长度（min／median／max／mean） |
| --- | --- | --- | --- |
| 甲 `--max-unique-documents` 不给 | 三条全 True，取到 250,000 符号 | 315 对 315（`unique_documents` 315） | 84／626.0／2858／795.888889 |
| 乙 `--max-unique-documents 32` | 三条全 True，取到 250,000 符号 | 324 对 324（`unique_documents` 32） | 114／655.0／2643／771.91358 |

第三条独立核对：乙臂重放篇数 324 ÷ 池 32＝**10.125**，与收尾件自述的 `mean_revisits=10.125` 同值。

件：`reports/taiji_n3a_sequence_length_arm_a_20261008.json`／`reports/taiji_n3a_sequence_length_arm_b_20261008.json`。

## 5. 一条口径提醒：这一列与 §4quater 那句"全语料分布"不是同一个量

PLAN-N3-02 §4quater 记的是**整库** 123,090 篇的分布（min 53／median 671／max 8,214／mean 861.201），
而 §8.7 第四项要的是**本轮喂进去那些篇**的分布：甲只有前 315 篇（min 84／median 626.0／max 2858／
mean 795.888889）、乙是 32 篇池循环 324 次（114／655.0／2643／771.91358）。
**两个数不许互换、也不许拿整库那组冒充本轮那组**——这正是 §4quater 禁"拿参数值冒充实测分布"的同族。

## 6. 末尾三行（发表资格前置）

* **预期为绿的成员**：C-1…C-4 四条、两臂 `section_8_7_completeness` 五项齐、`measurement_complete` 为真。
* **仍红时收回哪句**：若任一臂仍 `ran_not_measured`（§8.7 其余四项缺任一项、或复算件守卫不齐），
  就收回"§8.7 这把锁已解开"这句并点名本件与那次读数，不许改用别的次要指标代答。
* **发表资格**：`J-N3a` 从 `ran_not_measured` 变成**判级读数**这件事＝已冻判据的**前置条件**换面，
  按 09 §3.2 的口径要 **owner 在本件签字**。签字之前只可说"取数面补齐、判读器已能复算"，
  **不可**说"×16.8 容量在等符号暴露下把斜率抬出了噪声带"。

## 7. 不开跑声明

本件零训练跑量、不动产品码、不动常量；两臂重跑（若要逐位自述那一列）仍各 ≈2.3 h 且要批。
## 签字（owner 第八次弹窗 · 2026-10-09 01:30）

owner 裁"两件都签"（与 [PLAN-N3-12](PLAN-N3-12_j_n3a_anchor_upgrade_prereg_20261008.md) 同日）⇒ §1 的 C-1…C-4
与判读器的 `sequence_length_source` 口径生效，§6 第三行那句"发表资格前置"就此满足：
**同源复算供数下的 `J-N3a` 读数可以入库、可以说出口**。仍受两条约束：

* ㊵-546⑤ 那两支 `not_holds` 用的是**旧锚**（`0.594120`，跨链/跨语料），随 PLAN-N3-12 换锚**降为历史读数**，
  不参与 J-N3a′，旧件不删不追改；
* 新第二合取项（甲臂末段均值 `0.352` > 对照臂末段均值 +0.02）**现在没有值**——`×2` 同语料对照臂已批、
  已于 `2026-10-09T01:32:59Z` 起跑（`output/n3a_control_x2/`，证据见 08 ㊵-547），到位之前
  `J-N3a′` 整条只能记"待对照"，不许拿甲臂自身的数或 §0 那条旧基线代答。
* §3 那句"已知不可判别"（`end_boundary_after_newline` 的 ±1）签字不改变它——它是披露，不是被批准的误差。
