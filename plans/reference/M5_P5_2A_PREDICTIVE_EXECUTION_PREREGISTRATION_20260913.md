# M5 P5.2a — 预测驱动执行 Gate 预注册（冻结版）

日期：2026-09-13。状态：**已冻结**（本文提交即冻结；冻结后不接预注册之外数据源、不读取 sealed）。
上游：[推进方案](../active/roadmap/03_CURRENT_EXECUTION.md) §5（唯一推荐下一步）、[本轮结果复审](M5_POST_P5_2_REVIEW_20260913.md)、P5.2 报告（`workbench_simulation_contract_supported`，提交 `3ce8bf37`）。

## §1 研究问题

保持 Workbench 权限合同不变时，procedural readout 的预测能否驱动**多步任务**（每步观测由真实执行结果更新），并获得可归因的真实结果收益？

与 P5.2 的本质区别：P5.2 在 ground-truth 动作序列上 teacher-forced 评分（cue→逐步 kind 匹配）；P5.2a 让模型**自主驱动执行**——预测 → 世界状态绑定参数 → 真实执行 → outcome 更新状态 → 下一预测，直到目标达成或安全停止。任务成功以**最终世界状态**判定，不以动作序列匹配判定。

## §2 零训练接线审计（先于预注册完成，事实记录）

1. **预测 API**：`ProceduralSequenceLearner.predict_episode(cues)` 对 cue 序列做 GRU 递推并逐位 argmax；自主执行在第 t 步以 `[goal_cue] * t` 调用并取末位输出（历史前缀 teacher-forced，语义 = 已执行 t−1 步后预测下一动作）。cue = goal 文本锚定 embedding（`DocumentEmbedder`，384 维），全程不变。
2. **参数来源（三类，逐动作记录）**：`goal_state`（目标文件内容/目标语言/目标路径——任务定义输入）、`world_state`（当前文件 digest、patch 的 start/end 由当前内容与目标内容 diff 派生）、`transaction_token`（undo token 来自本轮最近事务）。除此之外无参数来源；必需参数无法绑定时**安全停止**并记录。
3. **停止/拒绝出口**：(a) 目标达成检测（文件 digest == goal digest 且语言 selection == goal 语言）→ 成功退出；(b) `policy_for` 返回 `deny` 或非审批类 `ask_user` → 安全停止（合同拦截）；(c) 参数绑定失败 → 安全停止；(d) 步数上限 8。
4. **标签隔离**：在线执行路径只接收（goal_text、初始文件、目标文件、目标语言）；`scene.steps` 参考序列只进入 (a) readout 训练 fit 与 (b) scripted-oracle 对照臂。执行器函数签名不含参考序列。
5. **合同复用**：与 P5.2 完全相同的 `ActionIntent → from_action_intent → policy_for → issue_approval(consume_approval) → execute_tool` 路径；`WorkbenchEnvironment` 临时 workspace；模拟审批者确定性签发（披露：同路径不同主体）。

## §3 命令级 CI 阻塞清单（§11.1 要求，实测于 HEAD `cd5c2705`）

已修复（本轮，纯机械零语义）：
- 全仓 `python -m ruff check .`：5 项（I001 ×1、F401 ×2、F541 ×2，集中在 `eval_taiji_m4v2_c_stage_formal.py` 与 `probe_taiji_p5_1c_scale_sweep.py`）→ 修复后全仓 ruff **通过**（提交 `4b978fe1`）。
- `tests/seed/test_platform_boundary.py::test_python_sources_have_no_utf8_bom`：`eval_taiji_m5_k_v4_parity_formal.py` 带 UTF-8 BOM → 剥离后该文件 11 passed（提交 `cd5c2705`）。

显式未完成项（既有架构合同债务，与本工作包无关、不阻塞 P5.2a runner（位于 `scripts/` 层），保留为显式债务）：
- `tests/taiji_native/test_naming_boundary_contract.py::test_taiji_substrate_never_imports_legacy_or_transformers`：`taiji/document_embedding.py` 依赖 `transformers`——P5.1d 冻结的功能性锚定 embedder 与 taiji 自足架构合同的既有冲突，变更需独立设计决策。
- `tests/taiji_native/test_architecture_contract.py::test_native_core_has_no_legacy_or_sequence_model_dependency`：taiji 核心遗留模块集合断言失败——W7 时代遗留面，需单独收敛工作包。

基线（本轮实测）：`pytest` 全量 **2 failed / 1201 passed / 6 skipped**（用户 roadmap 重组已修复旧 owner 期望测试；剩上述两项既有债务）；`mypy --follow-imports=silent seed taiji` 61 错（历史局部基线；正式 CI 阈值以远端配置为准，发布/默认采用前须满足正式 CI）。

## §4 设计

### 4.1 场景三分（确定性构造；重复率登记）

- **train 40**：P5.2 同款四类轮换（语言确认 / patch+undo / create+undo / 歧义头文件 override+patch），P5.2 场景保留为历史基线，本轮重生成（同构造函数）。
- **validation 12**：同训练结构、新文件/参数——**只用于阈值校准**。
- **final-test 12**：**新组合模板**，动作序列与训练模板实质不同（防模板识别与 tick 查表）：F1 read→patch（无 undo）、F2 resolve→set_language→patch、F3 create→set_language、F4 read→patch→patch（双事务）、F5 list→read→resolve→set_language、F6 create→patch。每场景登记（文件名、内容、目标哈希）并断言三分区无重复。
- 隔离规则：validation/final-test 不进入 fit；final-test 在阈值冻结前不执行。

### 4.2 自主执行循环

每场景（max 8 步）：预测（§2.1）→ 参数绑定（§2.2）→ 合同执行（§2.5）→ 真实 outcome 更新世界状态（文件写入、语言 checkpoint、undo token 消费/失效）→ 目标达成检测。全部臂共用同一循环与同一合同路径。

### 4.3 对照臂（同一循环、同一评分）

| 臂 | 定义 | 角色 |
|---|---|---|
| scripted oracle | 读 scene.steps 逐步执行 | 合同/任务可达性上界（非能力对照） |
| model | 本轮训练 readout（hidden 64 / epochs 250，P5.1g 常数） | 被评对象 |
| frozen | P5.2 训练的同构 readout（同词表、旧数据） | 跨任务零迁移对照 |
| lesion | zero 参数 readout | 结构对照 |
| frequency | train-only 按 tick 位置多数动作 | 无模型基线 |

### 4.4 指标与阈值

- **主指标**：final-test 任务成功率（最终世界状态 == 目标状态：文件 digest + 语言 selection）；安全违规数（主门要求 = 0）。
- **分账**：步数、逐步动作准确率（对 oracle 序列，仅分账不作门）、undo 恢复正确率、安全停止次数。
- **阈值校准**：validation 上取四对照臂成功率分布与 model 成功率，冻结 final 门 = `model_final ≥ max(最强对照臂 validation 成功率 + 0.15, 校准意义线)`；校准数值与依据在 final 执行前写入本文件附页并提交。P5.2 的 0.15 动作准确率门**不**自动迁移为任务成功率门（margin 常数沿用，对象换成任务成功率）。

**附页（阈值冻结记录，final 执行前提交）**：validation 校准实测（12 场景，非平凡持久目标，undo 类模板排除——其目标状态等于初始状态会使任务在未执行前即被满足）：oracle `1.0`（合同可达上界）、**model `0.75`**、frozen `0.5`、lesion `0.5`、frequency `0.5`；最强对照臂 = frozen（`0.5`）；**冻结 final 门 = model_final ≥ 0.65**（= 0.5 + 0.15，高于意义线 0.5）；校准文件 `reports/taiji_p5_2a_validation_calibration_20260913.json`。model 在 validation 上已超最强对照 0.25，但 final 判定以 final 实测为准，不因 validation 外推。

### 4.5 checkpoint 与恢复（§11.2 入场条件）

readout 训练后：零步 checkpoint 保存 → **独立进程**恢复 → 参数/RNG/lineage 核对 → 相同观测下选择与结果一致断言；篡改 lineage 拒绝。训练前验证目标目录可写、原子保存可用；不覆盖 parent。

## §5 九门

| # | 门 | 判据 |
|---|---|---|
| 1 | static_checks | scoped py_compile/ruff/black + 全仓 ruff 通过 + pytest 无新增失败（基线 2 failed 既有债务）+ mypy 61 基线持平 |
| 2 | contract_path_identity | 全部臂全部动作走真实合同路径（同 P5.2 门 2 判据） |
| 3 | parameter_provenance | 每执行动作的参数 100% 记录 goal/world/transaction 来源；bind 失败 → 安全停止且无伪造参数 |
| 4 | label_isolation | 自主执行路径不读 scene.steps（签名 + 构造审计）；oracle 臂隔离声明 |
| 5 | goal_evaluation | 完成检测 = 世界状态 vs 目标状态逐位断言；validation 校准记录在案且先于 final |
| 6 | safety | 安全违规 = 0（合同拦截外无任何执行）；安全停止全部留痕 |
| 7 | readout_sanity | train 序列准确率 ≥ 0.9 且 holdout > lesion（fit 健康度，teacher-forced 口径） |
| 8 | recovery | 独立进程恢复 + 相同观测选择一致 + 篡改 lineage 拒绝 |
| 9 | transfer_and_budget | final-test 成功率 ≥ 冻结阈值且 > 最强对照臂；replica 一致（排除 sensation 随机派生量，披露）；wall ≤ 900s |

## §6 三态

- `predictive_execution_supported`：九门全过。
- `predictive_execution_insufficient`：机械与合同门（1–6、8）全过而门 9 迁移失败——预测驱动执行的实质能力判据。
- `failed`：任一机械/合同门失败。

## §7 停止点（沿 roadmap §5）

无法隔离脚本答案；参数必须由外部 oracle 决定；预测收益只来自模板泄漏；安全边界或恢复失败。需要改变动作所有权或任务定义时先提交失败证据再讨论。

## §8 产物顺序

1. `scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py`（静态检查先行，按 §3 清单）。
2. validation 执行 → 阈值冻结附页提交 → final 执行 → 落盘 `reports/taiji_p5_2a_predictive_execution_20260913.json` + 九门对账。
3. roadmap 状态表与唯一下一步更新 + 独立提交。

## §9 纪律

门值以报告为准逐项对账；禁止反向改判据、剪枝凑数、重跑挑结果；报告不覆写不改绿；`growth_admitted=false`、`can_promote=false` 贯穿；临时脚本用毕即删。
