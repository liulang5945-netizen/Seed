# M5-P5.1h 知识内化与采用：包合同草案 v1（静态盘点＋准入设计）

日期：2026-09-19。依据：用户在内容绑定线结案后的路线裁决中选定「知识内化 P5.1h」作为下一个 M5 缺口包（问答记录「1」）。定位：[01 §3](../active/roadmap/01_SCOPE_AND_PHASES.md#3-m5-四轴与晋级缺口) 知识轴必达项——**真实语料 child 采用前的硬依赖**。本文件是待审查草案：**本轮只做静态盘点与准入设计，不启动任何训练**；数值线（retention/admission 阈值）在静态只读探针后预注册冻结。

## §1 目的、范围与非目标

**目的**：把 P5.1g 已证明的"真实发布语料内容因果收益"（九门全过，sourced a-gate 0.6514，超频率基线 margin 0.284）推进为**可通过准入的 child**：retention/admission 线可达、旧能力保持、独立测试成立、来源与收益可分账。

**范围内**：独立测试 split 设计、旧能力保持协议、admission 线的求取与预注册、parent-child 恢复测试、来源许可清单、收益分账 schema。

**非目标**：默认产品采用（M6）、R2 语言能力、跨内容结构泛化（协作轴）、任何 sealed/历史 final 的读取、P5.1g 判据的追溯修改。

## §2 静态盘点（已核对）

**可继承资产**：
- 证据链：P5.1 内容迁移 0.5 → 对比辨别 0.7314 → encoder 注入 0.8700 → 构造同预算 0.25 → 真实语料同预算 0.6514（九门全过，`reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json`，预注册 `303f5d51`）；
- 机制模块：`taiji/internalization.py`（grounded 知识原语、ledger、六态含 admitted/rolled_back/tombstoned）、`taiji/internalization_learner.py`（`consolidate`：train/holdout/retention 三分区 + 双 lesion + parent→child 原子 trial）、`taiji/internalization_longitudinal.py`（外部描述移除候选、稳定性门，无文件系统/删除能力）；
- 冻结构造常数：hidden 64 + 250 epochs 的 trial-learner 测量路径（与 consolidate 内部逐位同构）、calls 配额采样口径（两臂 989/385/253 逐位相等）。

**已知缺口（P5.1g 遗留，正是本包要闭合的）**：
- 两臂 consolidate `admitted=false`、`rolled_back=true`——admission 线不可达已测量（hidden16 retention 天花板 ~0.38；hidden64 全面更优但仍未达线）；
- 独立测试（超出 P5.1g 所用词表/语料切片的 holdout）、旧能力保持协议、来源许可清单、收益分账 schema 均未建立；
- P5.1g 的 `4 failed/1199 passed` 与 mypy 61 为历史局部基线，不作为本包静态门口径。

## §3 准入设计框架（数值线后置冻结）

1. **独立测试**：与 P5.1g 语料切片词表不相交的独立测试 split（来源同一真实发布语料的后续分区或独立来源，选定后登记许可与 digest）；独立测试上收益方向与幅度为 admission 的必要项。
2. **旧能力保持**：retention 集 = 父代既有能力代表样本（K 轴五类合成载体限定范围 + P5.1g retention 分区），保持线 = retention 损失退化 ≤ 预注册比例；parent 份额逐位保留（checkpoint digest 链已有机件）。
3. **admission 线求取**：静态只读探针（无训练）先测量在不同 passes/数据配额/分区比例下 retention-holdout 联合可行域，据此**预注册** admission 阈值（新 child 独立测试收益 ≥ 线 A、retention 退化 ≤ 线 B、双 lesion 显著），随后才允许训练运行；不看训练结果调线。
4. **来源/仪器与原生收益分账**：报告须分开记录语料来源与许可、仪器路径（trial-learner 测量 vs consolidate 内部）、原生收益（超频率基线 margin 沿用 P5.1g 口径的独立测试版）。
5. **parent-child 恢复**：零步保存→新进程加载→继续一步→digest 对应（沿用本仓 atomic/digest 纪律）；ledger 六态流转（shadow→internalized 或 rejected/rolled_back）有测试。

## §4 条件推进与边界

草案（本文件）→ 用户批准 → 静态只读探针（无训练；求取并预注册 §3 各线）→ 实现门（零训练测试）→ 训练预算审批（单独决定）→ 标定/正式运行 → admission 判定（adopted 或 rolled_back，两态都可接受并如实记录）。**边界**：不碰默认产品与 M6 采用；不读历史 final；不改 P5.1g 已冻结判据（新线全部另版预注册）；训练预算未批准前不跑任何 consolidate。

## §5 结果去向

采纳（admitted）→ 报告 + ledger 状态 + parent-child 恢复证据 → M5 知识轴缺口闭合一项；判停（rolled_back）→ 负结果与可行域记录入账，缺口保留。两态都不自动晋级 M5 其他轴、不触发 R2 重启。

## §6 静态探针结果与预注册线（2026-09-19，零训练，纯报告考古）

**P5.1g admission 不可达的精确分解**（`reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json` + runner `eval_taiji_p5_1g_real_corpus_quota_budget_gate.py`）：

- 判据：`procedural retention accuracy ≥ 0.5`（ADMISSION_REVISION `p51g:admission`）；实测 **0.4189**，差距 0.081；
- 新知识侧无碍：procedural holdout 0.4512、semantic holdout 1.0→0.000597（零退化）、a-gate 0.6514；
- retention 上读数并非零学习：0.4189 比冻结 per-tick 多数基线 0.367 高 0.052——是"学到但不到线"；
- **配方旋钮未动**：consolidate 的 `replay_digest` 为空、`passes=4` 默认、sourced 臂 ranking_pairs = 0——replay 与排序对两个现成机制在 P5.1g 完全未使用；
- runner 未持久化任何 checkpoint（全 in-process），故 parent 侧对照只能靠仪器复跑（冻结 1200s 口径）获得，列入实现门而非本探针。

**P5.1h 预注册线（冻结，不得看训练结果调整）**：
1. procedural retention accuracy ≥ **0.5**（沿用 p51g 线，不放宽）；
2. 独立测试内容迁移 margin ≥ **0.15**（a-gate 切片与 P5.1g 的 Tool_Use_000328–344 及 train 词表不相交）；
3. semantic retention 损失退化 ≤ **0**（不劣化，沿用 P5.1g 实测 1.0→0.000369 的量级守恒）；
4. 双 lesion 纪律与 checkpoint 往返不变。

**配方变量（唯一允许变动面，标定阶段用）**：replay 比例 × passes × ranking pairs——三者机制现成、P5.1g 均未使用；数据配额与词表口径沿用冻结值。**边界**：门槛不动、语料身份不动（UltraData-SFT-Agent-2609, Apache-2.0）、默认产品采用仍非目标。

