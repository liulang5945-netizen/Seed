# M5.K1 Formal 预注册：技能组合课程多 seed 稳健性（canary 通过后的形式化）

> 起草日期：2026-09-09。前置：[K1 canary 预注册及执行记录](M5_K1_SKILL_COMPOSITION_PREREGISTRATION_20260909.md)（§7：K1.1 类型化绑定后 canary Gate 通过，单 seed）。本文在看任何 formal 结果之前预注册 9-cell 矩阵、判据与停止线；实现与判据冻结后运行。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> K1 canary 的组合成功率（A 臂未见三元组真实执行成功率 1.0、对照分离 1.0）**不是单 seed 抽样运气**：跨 task_seed（工作区文件内容变体）与 learner_seed（runtime/config seed）矩阵，A 臂未见组合成功率保持 ≥0.8/cell，且 A−B、A−C 分离保持 ≥0.3/cell。若任一 cell 失败，则 canary 结论不稳健，需按失败行归因（fact 组合？planner？执行？）而非调参。

## 2. 矩阵与臂（全部复用 canary 实现，不改 learner/掩码/判据）

| 维度 | 取值 | 变异来源 |
|---|---|---|
| task_seed | 0, 1, 2 | 工作区文件内容变体（`_build_workspace` 的 variant 位移）；语言解析/任务结构不变 |
| learner_seed | 17, 23, 31 | `TaijiConfig(seed=…)`、runtime episode、`InternalizationConverter(seed=…)` |
| 臂（每 cell 内） | A full-chain / B frozen / C semantic-lesion | 与 canary 完全相同（K1.1 类型化绑定 + 状态只读出） |

实现：新脚本 `scripts/training/eval_taiji_m5_k1_skill_composition_formal.py`，逐 cell 调用 canary 的 `run_cell(task_seed, learner_seed)`（import 复用，零逻辑复制），串行 9 cell，单 CPU。

## 3. 量尺合同（沿 M4.V2.R0 语义）

- **主量尺是绝对任务成功率**（unseen 三元组真实执行成功行数 / unseen 行数），**对比量尺是臂间 delta**（A−B、A−C），两者分字段报告，不聚合混算。
- **无 parent baseline，如实缺省**：K1 链不写任何 F1/记忆/结构 owner（runtime `learn=False`、语义 learner 为独立小网络），既无父代 checkpoint 可比较，也无旧能力损害面。parent-retention 字段以 `null` 显式缺省（fail-closed 语义：缺基线不得声称「遗忘」或「保持」），不伪造。
- **无跨域 BPB 聚合**：K 的量尺是任务成功率，不是 BPB。

## 4. 判据（全部预注册，看结果前冻结）

**技术门（每 cell 必须全过，任一失败即停）：**

1. canary 全部 4 项技术检查通过（含 `all_arms_produced_rows`）；
2. Stage-2 接口 6/6 良构；
3. A 臂全部行真实执行并产出 outcome 记录。

**主判据（稳健性）：**

4. **A unseen 成功率**：`9/9` cell ≥ 0.8，且 9-cell 均值 ≥ 0.9；
5. **对照分离**：`9/9` cell A−B ≥ 0.3 **且** A−C ≥ 0.3；
6. **S6B 准入**：`9/9` cell 中 A 臂全部成功 read outcome 的准入率 = 1.0（resolve 类成功 outcome 的 stale 投影限制为已知次要项，不计入判据、逐 cell 报告）；
7. **reward_variance > 0**：`9/9` cell 的 A 臂全行 reward 方差 > 0（成功/失败混合，真实 outcome 非常量）。

**报告义务（非 Gate）：**

8. B/C 臂逐 cell 成功率如实呈现（预期接近 0；不设上界判据，分离由判据 5 承担）；
9. 逐 cell 的失败行明细（path/status/accepted/intent/success/admitted）。

**Aggregate 与定位**：报告 format `taiji-m5-k1-skill-composition-formal-v1`，逐 cell canary 结构 + aggregate（每指标 min/mean/max、通过计数 n/9、`cells_passed`、`robust` 布尔）；`can_promote=false` 固定。**K1 formal 通过 = K1 证据线闭合（组合成功率跨 seed 稳健），授权进入 K2 设计或 A8 K 轴 formal 讨论；不是 A8 晋级，不是结构成长晋级。**

## 5. 停止线

- 任一技术门失败 → 停，归因 harness/learner，修复走新预注册，不放宽不换 seed；
- 判据 4/5 失败 → 按失败行逐行归因（fact 组合 status？planner reason_code？执行 outcome？），不调掩码派生、不调阈值、不挑 seed 重跑；
- 判据 6/7 失败 → 回到 S6B 投影/回报派生设计，属内化管线问题，不在 K1 内修；
- 不引入 provider/联网/真实客户端写入；执行全部在进程私有临时工作区，运行后清理。

## 6. 明确不做

- 不改 `taiji/semantic_training.py` 的掩码语义与 `SEMANTIC_TRAINING_VERSION`；
- 不改 canary 的 `run_cell` 判据与臂结构（formal 只做矩阵化与 aggregate）；
- 不做 K2（更深组合维度）、不做 checkpoint 持久化、不做跨 formal 数据混用；
- 不因 formal 通过自动解冻 M5 第二项或 A8 晋级讨论——晋级只由 aggregate scorecard 按 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) §5.3 计算。

## 7. 资源与验收产物

- 单 CPU、9 cell 串行（每 cell 与 canary 同量级，总时长 < 10 分钟量级）；
- 产物：`reports/taiji_m5_k1_skill_composition_formal_20260909.json`（versioned、逐 cell、aggregate、`can_promote=false`）+ 路线图执行记录更新 + 独立提交。
