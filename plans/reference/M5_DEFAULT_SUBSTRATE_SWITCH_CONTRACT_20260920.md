# 默认基座换底合同：产品默认入口改服务 16M-tick 训练态（2026-09-20）

授权：项目所有者 2026-09-20 弹窗裁决"换默认基座为 16M-tick 训练态，并同时满足它的来源要求"。同轮另一裁决是 H 阈值按 max×2 草案冻结（见 [冻结件](M5_CAP_H_THRESHOLD_FREEZE_20260920.md)）。本件冻结**做法、判据、影响面与停止线**；冻结后按 §1→§5 顺序执行，不在执行中调判据。

## §0 为什么这是缺陷修复而不是研究

`api/seed_runtime.py:29` 的 `DEFAULT_CHECKPOINT` 现指向 `checkpoints/seed_corpus.pt`，而该文件的来源登记件自述："这份文件是测试套件重初始化的产物，不是随仓库分发的出厂基座…原 tick=2 基座已不可还原（DEBT-I9）"。也就是说**产品默认入口当前服务的是一台测试跑出来的空壳**。把它换成真实训练态属于 R1（可信底座）范围，不是 R2 能力主张，也不构成任何语言能力声明。

## §1 前置判停 V0：默认 loader 能否无补丁加载 `seed_beta.pt`

必须先测、且结果决定后续走法。测法：子进程内 `SeedRuntime.load(checkpoints/seed_beta.pt)`，**不装任何进程内补丁**，记 `tick`、一次 `chat(..., learn=False)` 的原始输出与异常类型。

| 结果 | 走法 |
|---|---|
| 加载成功且产出原始输出 | 直接进 §2。此时 `REQUIRED_CHAIN` 里的 `relax_legacy_guard` 对该文件已是**冗余**（M2-2i 已在 `taiji/model.py:3333-3345` 落了"旧格式缺 organ 载荷 ⇒ 保留新初始化器官"的迁移，只放宽缺失、载荷在场仍走 lineage 校验），本件把这一点写成实测结论而非推论。 |
| 加载失败，失败点仍是 organ/legacy 一环 | **先补产品源码的迁移，再换底**。按本仓既有纪律：是迁移不是放宽；配一支"按设计会失败"的测试（改坏迁移它必须红）；lineage 校验对错误父本必须继续响亮失败。 |
| 加载失败，失败点在别处（信封格式/权重键/provider 装配） | 停下呈报，不换底。把 16M-tick 设成默认但入口靠补丁才打得开，等于把测试期 monkeypatch 搬进产品路径——那是新缺陷，不是修 DEBT-I9。 |

## §1.1 V0 实测结果（2026-09-20，已执行）

**结论：走第一条路。** 裸 `SeedRuntime.load(checkpoints/seed_beta.pt)`（不装任何进程内补丁）成功，
`tick=16000000`，`chat(..., learn=False)` 返回 41 字符 `str`，换题面输出改变；装上
`relax_legacy_guard` 后**两次读数逐位相同**（答案摘要同为 `a15059f7b819a665`，换题面为
`43ae955b161d21cb`）。⇒ 该补丁对这份 v8 文件已是冗余，真正让它可加载的是产品源码里的
M2-2i 迁移（`taiji/model.py:3333-3345`，只放宽"旧格式缺 organ 载荷"，载荷在场仍走 lineage 校验）。
换底因此**不需要**把测试期 monkeypatch 搬进产品路径。

同一份信封给出的来源三元组（写进 `plans/manifests/product_default_checkpoint_provenance.json` v2）：
`trainer=train_seed_corpus`、`tick=16000000`、`saved_at_utc=2026-08-24T02:23:26Z`（与文件时间戳
10:23:26 同一时刻）、`corpus_fingerprint=simple_zh_texts.jsonl / 1,394,775,610 B`；旁证是
`reports/seed_beta_progress.jsonl`（同一条长训的 tick 流，记到 16,800,000）与
`plans/archive/history/SEED_PUBLIC_BETA_ROADMAP.md` 里 2026-08-24 启动长训的完整命令行
（`--resume/--checkpoint` 均指这份文件，`--checkpoint-every 2000000` 与 16M 恰在存档边界相符）。
**边界照写不误**：当时的 `logs/long_train_20260824.log` 已不在盘上，所以这三者只是**互相吻合的一致性证据**，
不是可复算的生成性凭据 ⇒ 清单里 `provenance_limits` 三条必须在，缺了就红。

## §2 来源登记（DEBT-I9 的闭合半，不许提前宣布结项）

`plans/manifests/product_default_checkpoint_provenance.json` 改指新基座，字段如实填：路径 `checkpoints/seed_beta.pt`、sha256（实测，现算）、字节数、`envelope_tick`、`envelope_trainer`、`envelope_saved_at_utc`、文件时间戳（现记 8-23，早于任何套件写入时刻）。

**关键区分**：DEBT-I9 要的是"来源可判定"。"不是套件写的"只否证了一半；另一半"它是哪次训练产出、由哪个脚本/配置产出"要有凭据（训练日志/报告里的 sha 对应关系）。若查不到凭据，manifest 的 `provenance` 只能记 `non_test_artifact_origin_unverified`，**不得记 `known`，DEBT-I9 不得宣布结项**——这条要写在件里，防止用"文件更老了"冒充来源证明。

## §3 换底影响面（一处改，处处要跟）

| 面 | 必须同步的动作 |
|---|---|
| `api/seed_runtime.py` `DEFAULT_CHECKPOINT` | 改指新基座；相关合同测试跟着改。 |
| 来源 manifest 与其合同测试 | `test_product_default_checkpoint_provenance_contract.py` 现钉 sha/变化守卫 ⇒ 换文件必红，须同批改指并复跑。 |
| CAP 机器维度 | B–G 评价**整体在新基座重跑**并落**新文件名**（封存件不覆写）；C/D/E 读数预期仍贴地——这是能力事实，不因换底而变。 |
| A/H/F 健康支 | 在新基座重跑 `--health --f-live-evidence`；A05/A05b 读数重取（本轮实测：默认基座裸/required 两链 A05b 均 false，`seed_beta` 上 required 链曾测 true，新底是否复现要重测不外推）。 |
| H 标定 | 阈值绑定 `(设备,链路,checkpoint)` ⇒ 新基座上重采 5 次，再把 §4 冻结件生效。 |
| F04 两子句 | "默认入口服务的确实是训练态"这条复判会跟着变；F04 是否仍 fail 取决于模板回显那条是否仍成立。 |
| P3b/战役配对 | 债册记过一类失效正是"健康报告与它旁边的评价报告不同检查点"；换底后 `P3A_HEALTH`/`P3A_BASELINE` 配对关系要重新核对，别把 mismatch 静默留在链路上。 |

## §3.1 重跑范围（所有者裁决 2026-09-20：CAP 全套＋战役基线配对重取）

| 件 | 为什么必须重取 |
|---|---|
| CAP B–G 评价（裸链与约束链各一份） | 分数描述的是"默认入口服务的模型"，换底即作废 |
| 健康支 A/H/F（带 `--f-live-evidence`） | A05/A05b 的读数按 checkpoint 而定；旧退出材料取的是换底前的基座 |
| H 标定 5 次 + 阈值生效 | 冻结件 §1 把阈值绑定 `(设备,链路,checkpoint)` |
| 战役基线对 `P3A_BASELINE` / `P3A_HEALTH` | 配对守卫要求"健康报告与它旁边的评价报告同检查点"；两常量现指 09-15/09-18 的旧件，须在新默认上重出同名配对，否则每场战役的 A/H 支会静默退化成 `source_mismatch` |
| CAP runner 的默认基座常量 | 与产品常量分写两份正是本轮刚清掉的那类漂移（F 维旧手抄件同型）⇒ 改为直接引用 `api.seed_runtime.DEFAULT_CHECKPOINT` |

## §4 本件不授权的事

不把 `constrained_decode` 变成产品默认（那是表达层/R2 侧，属另一项裁决）；不改 A05b 的必过地位（换底正是为了让它在真实训练态上判）；不动任何已冻结阈值与合同；不读 sealed/final 集；不覆写封存报告与既有标定件。

## §5 停止线与如实记录

任一 §1/§2/§3 步骤出现不可解释的偏差（sha 不符、加载出例外、复跑读数与新底不自洽）⇒ 停在该步并呈报，不绕过、不"先合了再说"。收口口径沿用本仓既有对照基线：全量套零新增失败、`checkpoints/*.pt` 跑前跑后 sha 不变、canary 条目数不涨。
