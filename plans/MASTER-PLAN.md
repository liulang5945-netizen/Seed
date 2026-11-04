# MASTER-PLAN（唯一权威主线）

> 文档状态：[当前唯一权威主线·2026-11-04 收敛版]。它取代 2026-11-03 的 CONSOLIDATION 与 17 时代入口；读取顺序：本文件 → [账本](reference/INVARIANT-LEDGER_20261103.md)（判据/定理）→ [能力卡](reference/CAPABILITY-CARD_20261103.md)（读数边界）→ [03 执行流水](active/roadmap/03_CURRENT_EXECUTION.md)（逐日回执）。

## 0. 一句话与队首

**唯一主线**：以原生字节为主体、同一主体内逐字节表达；按“数据前提 → 图验 → 定价 → 恢复预检 → 正式比较 → 条件确认”的窗口制推进，一个窗口判负就归因、修图、再验，不用训练掩盖结构缺口。

**当前窗口**：18 原文自主绑定（18.1 正式比较已判负；18.2 修复包 = 轨道包 **19.1–19.2**，验证达标）。
**队首（下一条要做的唯一事）**：**19.3 定价**（原 18.2 定价）——在新图上实测每更新 CPU，重算 `candidate_sha_v2`，出具六格裸价/保护价。未出价前不得启动 19.5 的任何正式比较。轨道与判据见 [ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md)。

## 1. 状态表

状态取值：**活**（有调度效力）/ **死**（已判负或永久否定）/ **待裁定**（owner 未决）/ **历史**（只作证据，不调度）。

| 线名 | 状态 | 最近读数 | 入口判据 | 出口判据 | 证据指针 |
|---|---|---|---|---|---|
| 09–13 生长/种子/剂量方向 | 死 | 旧剂量臂均未过任务门；仅保留工程资产 | 旧合同 | 旧合同 | [05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md) |
| 14.6 完整动作 | 死 | 0.5844（门 0.80） | 旧合同 | 任务门 | `reports/*14_6*` |
| 15.1 内容转移 R1 | 死 | 0.5625（2400 Adam） | R1 合同 | 任务门 | [N15 规范](reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md) |
| 16.1 选择—解码绑定 | 死 | 0.0500 / 0.0167（1200 Adam） | 16 合同 | 任务门 | `reports/*16_1_compare*` |
| **17 选择—锁定绑定窗口** | **死** | 六格 overall 0.000–0.200（门 0.80），任务门全未过；`n17_negative_closeout`；结构消解被 **M1** 判为不可解读 | 17.2R 图验通过 | 整句 ≥0.80、复制 ≥0.60、结构 A−C>0 | `reports/taiji_selection_routing_17_7_closeout_20261104.json` |
| OP2 数学 **M1** | 生效（永久） | ∂η/∂α ≠ 0：内容保真与记录选择不可结构性分离 | 数学审查 | — | [OP2 规范](reference/SPEC-COPY_OPERATOR_OP2_MATH_20261104.md) |
| OP2 数学 **M2** | **待裁定** | “任意记录切换 + 逐位恢复”不可同时要求 | — | owner 选 ①/② | [账本 §13](reference/INVARIANT-LEDGER_20261103.md#13-m2-的两个可裁定选项建议-待-owner-裁定实现按--设计未裁定) |
| OP2 数学 **M3** | 生效（永久） | 前向混合权重 ≠ 训练信用；不得据 γ 推断信用饥饿 | 数学审查 | — | 同上 §12 |
| 18.0 数据前提/定价/预注册 | 活（已交付） | 220 名字池、训练/确认不相交、六格裸价 9,109.4 / 保护价 19,220.8 CPU（18.0 口径） | 三条数据前提 | 预注册 10 门冻结 | `reports/taiji_data_premise_18_0_pricing_20261104.json`＋`plans/manifests/…18_0_pricing…json` |
| **18.1 六格正式比较** | **死** | overall 0.000、`new_name_copied` 0.000、unseen 0.000；16,196.3 CPU / 1,800 Adam；`release_confirm_n18_2=false` | 18.0 预注册＋图验 | 10 门 | `reports/taiji_selection_routing_18_1_compare_20261104.json` |
| 18.2 归因（γ 扫描/UTF-8/漂移） | 完成（结论生效） | 复制命中 0.090；b 0.290；b_pure 0.755；c 0.390；c_γ0 1.000；γ 6 档整句全 0 | 18.1 判负 | 四条归因落账 | 账本 [§11](reference/INVARIANT-LEDGER_20261103.md)＋[§14](reference/INVARIANT-LEDGER_20261103.md) |
| **包 19 位置计数＋UTF-8 掩码**（原 18.2 修复包 = 19.1–19.2） | **活**（19.1/19.2 达标，19.3 待定价） | b_pure 内容 1.000、物件/地点 1.000/1.000、首末差 0.000；自由解码无非法字节率 1.0；恢复四项 0；100.1 CPU / 4 Adam | 诊断归因（transport 指针＋解码掩码） | b_pure ≥0.95、首末差 ≤0.02、UTF-8 ≥0.99、恢复 0 | `reports/taiji_cursor_utf8_18_2_20261104.json`＋manifest；轨道见 [ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md) |
| 旧能力保持 | 不可判 | `retention_not_admissible`（首个合格父为空） | 首个合格父 | 两层判据：硬不变量 0 容忍＋留存率 bootstrap 下界 ≥0.85 | [账本 §10](reference/INVARIANT-LEDGER_20261103.md) |
| 经验写入面 | 不存在 | `experience_surface_absent` | 跨 session 写入合同 | 经验面验收 | 账本 T11；前提见 ROADMAP §3 |
| 18 窗口剩余（条件确认/跨轮状态/自主绑定） | 未启动 | — | 包 19 结案＋owner 批准 | 系统计划 §5 各阶段门 | [系统计划](reference/PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md) |
| **S19 跨 session 经验与事实修订** | 未调度 | — | 包 19/20 结案＋写入合同冻结 | overall ≥0.80、时间/冲突 ≥0.75、增益 ≥0.10（CI>0） | 系统计划 §6 |
| **S20 反馈塑形/保持/首次任务可塑扩容** | 未调度 | — | S19 结案 | 系统计划 §7 | 系统计划 |
| **S21 世界模型/条件价值/无新提示活动** | 未调度 | — | S20 结案 | 系统计划 §8 | 系统计划 |
| **S22 继承式规模生长/终审** | 未调度 | — | S21 结案＋硬不变量全绿 | 系统计划 §9 | 系统计划 |

## 2. 依赖顺序（谁先做、谁必须等谁）

1. **现在唯一可做**：19.3 定价（工程测量，0 训练、0 confirm）。
2. 定价之后、且 owner 批准 + M2 裁定之后：19.4 预检 → 19.5 正式比较（新合同、新证据文件名）。
3. 包 19 结案 → 包 20（逐字节任务门落地）→ 包 21（经验写入面立项前提）。
4. S19 → S20 → S21 → S22，逐级解锁；每级结案才可开工下一级，**当前一律未调度**。
5. **被否定定理阻断**：
   - **M1** 阻断一切“内容通道与绑定通道结构分离/独立可控”的架构主张与相应消融签收；A−B / B−C 只能作为**整条选择支路的代理信用**解读。
   - **M2** 阻断“任意记录切换 + 逐位恢复”并存；T8 与 17.4 的跨记录续推声明必须限定到同一控制范围（待裁定 ①/②）。
   - **M3** 阻断由 γ 推断训练信用饥饿；γ 相关实验只能报告前向混合效应。
6. **被前置条件阻断**：旧能力保持（无合格父）、经验写入（无合同）、S19–S22（包 19/20 未结案、未定价）。

## 3. 单一权威源（每条事实只允许一处正文）

| 事实类别 | 唯一权威文件 | 其他文件只保留 |
|---|---|---|
| 当前状态、队首、待裁定 | 本文件 | 一行指针 |
| 推进轨道（步骤、判据、预算、分支） | [ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md) | 一行指针 |
| 判据阈值、硬不变量、否定定理 M1/M2/M3 | [账本](reference/INVARIANT-LEDGER_20261103.md) | 一行指针 |
| 读数解释边界（数字该怎么读） | [能力卡](reference/CAPABILITY-CARD_20261103.md) | 一行指针 |
| 逐日执行流水与回执 | [03](active/roadmap/03_CURRENT_EXECUTION.md) | 一行指针 |
| OP2 形式化 | [OP2 规范](reference/SPEC-COPY_OPERATOR_OP2_MATH_20261104.md) | 一行指针 |
| 系统阶段目标（18–22） | [系统计划](reference/PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md) | 一行指针 |
| 工作包出口/失败回流细则 | [10](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) | 一行指针 |
| 17 机制细则（已结案） | [17 设计](reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md)＋[SPEC-N16-01](reference/SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md) | 标 `historical` |
| 数据前提定义 | `taiji/data_premise_18_0.py`＋18.0 定价报告 | 一行指针 |
| 证据数值 | `reports/*.json`（只读，不改字节） | 一行指针 |
| 文档索引 | [PLAN_INDEX](active/PLAN_INDEX.md) | 只做指针，不写结论 |

## 4. 冲突消解（唯一正确版本）

| # | 冲突 | 旧版本（标 superseded） | 唯一正确版本 | 证据 |
|---|---|---|---|---|
| 1 | 17 结构分离是否成立 | “内容保真通道独立可控/可分离” | **不成立**（M1：∂η/∂α≠0）；只可作整条选择支路代理信用 | 账本 §12 M1、L4 已改 |
| 2 | γ 门归因 | “门是最大压制/主因是 γ 门” | **错误**：门只放大**已对准**的指针；γ=0 只说明前向质量压在生成侧；按 M3 不得推断信用饥饿 | 账本 §14.4、γ 扫描、18.2 |
| 3 | 硬不变量表述 | “任意记录切换 + 逐位恢复，无条件” | **同一控制范围内**逐位一致；两者不可同时要求（M2 待裁定） | 账本 §10/§13 |
| 4 | 逐字节门是否采纳 | “未采纳/只是诊断” | **已作为下一包正式任务判据预注册**：物件/地点 ≥0.95、抄回 ≥0.60、整句 ≥0.80；不追溯 18.1；18.2 的 ≥0.95 是**诊断门**，不是任务门 | 账本 §11/§14 |
| 5 | `prefix_cursor` 是否应非零 | “恒 0 是正常状态面” | **错误**：恒 0 是缺陷；18.2 起必须随接受字节递推并进 bundle；读数 A 18/18、B 16/16 | 能力卡 §8、18.2 报告 |
| 6 | 17.4 “任意记录切换中断续推一致” | 无条件跨记录续推 | 限定为同一可达控制范围内的切换一致 | 账本 §13 表 ① |
| 7 | 18 是否已达标 | “18 已接近达标/复制只是没被用上” | **18.1 判负**；18.2 只证明复制通路的结构可达性（gold 记录+gold 起点），不是任务能力 | 18.1 报告、18.2 §14.2 边界 |
| 8 | 旧能力保持是否完成 | “保持已通过/漂移小即保持” | **不可判**：无合格父，`retention_not_admissible`；不得用 NLL 漂移冒充保持 | 账本 §10、T10 |
| 9 | 19 号编号撞车（跨 session 经验 vs 位置计数包） | “19 = 跨 session 经验” | **包 19 = 位置计数＋UTF-8（含 19.1–19.7）**；系统计划长期阶段改用 **S19–S22** 并冻结调度 | [ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md) §0 |

## 5. 待裁定（只列必须 owner 决定的）

1. **M2 选 ① 还是 ②**。建议 **①**（逐位继承，记录控制限定同一控制范围）：已交付工程证据都按 ① 验证；改 ② 需重签全部逐位恢复证据。若改判 ②，只换判据文字，不动图结构。
2. **逐字节门是否正式采纳为下一包任务判据**。建议**采纳**（0.95/0.95/0.60/0.80，不追溯 18.1）：它把“复制是否承重”变成可机检，且与 18.0 独立门相容。
3. **19 是否在位置计数达标后立即启动**。建议**不立即**：位置计数只在 gold 条件下达标，自由路由/任务尚未验证；先定价 + 18.2 修复版正式比较，再启 19。
4. **“任意记录切换”是否永久放弃为能力声明**。建议：若 M2 选 ①，则**永久放弃**该能力声明（保留为上限研究方向）；若选 ②，则保留声明但放弃逐位恢复判据。

## 6. 盘点（计划/判据/结论文档）

统计（2026-11-04 收敛后实测）：`plans/` 共 **1,063** 个文件（827 md + 232 json + 4 其他）；其中 `reference/` 672（进入当前决策集 8 份 + 本次旁挂 4 份 + 轨道 1 份，其余 659 只作历史证据）、`active/` 20、`manifests/` 218、`archive/` 150；`reports/` 共 **118,077** 个文件，其中 16–18 命名决策报告 **44** 份。

| 路径 | 用途 | 最后更新 | 调度效力 |
|---|---|---|---|
| `plans/MAINLINE.md` | 单入口 | 2026-11-04 | **active（唯一入口）** |
| `plans/MASTER-PLAN.md` | 唯一权威主线（本文件） | 2026-11-04 | **active** |
| `plans/active/roadmap/03_CURRENT_EXECUTION.md` | 执行流水 | 2026-11-04 | active |
| `plans/active/roadmap/02_GATES_AND_CI.md` | CI/门禁规则 | 2026-10-30 | active |
| `plans/active/roadmap/01_SCOPE_AND_PHASES.md` | 范围/阶段 | 2026-10-30 | active（范围） |
| `plans/active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md` | 工作包出口/回流 | 2026-11-01 | active |
| `plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md` | 旧主线计划 | 2026-10-30 | **superseded → 本文件** |
| `plans/active/roadmap/04_EXECUTION_PLAN.md` | 旧路径兼容入口 | 2026-10-22 | active（别名，不发布队首） |
| `plans/active/roadmap/05_TECH_DEBT_REGISTER.md` | 技术债登记 | 2026-10-30 | active（登记） |
| `plans/active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md` | 旧决策论证 | 2026-09-15 | historical |
| `plans/active/roadmap/07_MINI_MODEL_DELIVERY.md` | 交付计划 | 2026-10-30 | active（长期） |
| `plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md` | 上游同步 | 2026-10-30 | active（工具） |
| `plans/active/roadmap/11_CI_FLOAT_EVIDENCE_FOR_OWNER_20261025.md` | CI 浮点证据 | 2026-10-25 | historical |
| `plans/active/roadmap/12_LINE2_EVIDENCE_CLOSEOUT_20261025.md` | Line2 结案 | 2026-10-25 | historical |
| `plans/active/PLAN_INDEX.md` | 文档索引 | 2026-11-04 | active（只做指针） |
| `plans/active/TAIJI_CORE_REQUIREMENTS.md` | L1–L14 需求 | 2026-10-31 | active |
| `plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md` | 原生架构 | 2026-10-31 | active |
| `plans/active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md` | 长期发展（约束） | 2026-11-01 | historical（长期约束，不调度） |
| `plans/active/ARCHITECTURE_DIRECTION_2026_08.md` | 旧方向 | 2026-08 | historical |
| `plans/active/SEED_ARCHITECTURE.md`、`SEED_DEVELOPMENT_ROADMAP_2026_08.md` | 项目背景 | 2026-08 | historical |
| `plans/reference/INVARIANT-LEDGER_20261103.md` | 判据/定理 | 2026-11-04 | **active（判据唯一源）** |
| `plans/reference/ROADMAP-NEXT_20261104.md` | 推进轨道（19/20/21 包） | 2026-11-04 | **active（轨道唯一源）** |
| `plans/reference/CAPABILITY-CARD_20261103.md` | 读数边界 | 2026-11-04 | **active（解释唯一源）** |
| `plans/reference/SPEC-COPY_OPERATOR_OP2_MATH_20261104.md` | OP2 形式化 | 2026-11-04 | active |
| `plans/reference/PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md` | 18–22 阶段 | 2026-11-03 | active |
| `plans/reference/REVIEW-PACKET_20261102.md` | 17 时代评审 | 2026-11-02 | **superseded**（17 已结案） |
| `plans/reference/CONSOLIDATION_20261103.md` | 上次收敛 | 2026-11-03 | **superseded → 本文件** |
| `plans/reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md`、`SPEC-N16-01_…DRAFT_20261103.md` | 17 机制细则 | 2026-11-03 | historical（17 死） |
| `plans/reference/PRICING-17_3_SELECTION_ROUTING_20261103.md` | 17 定价 | 2026-11-03 | historical（价随图改作废） |
| `plans/reference/ATTRIBUTION-ARCH-VS-ALG_20261103.md`、`MATH-REVIEW_…_20261103.md`、`DESIGN-CANDIDATE-COMPOSITION_20261101.md` | 归因/数学/14–16 设计 | 2026-11-03 前 | historical |
| `plans/reference/` 其余 612 个文件 | N1 及更早 | ≤2026-11-02 | historical（不调度） |
| `plans/manifests/` 16–18 预注册（18.0 定价、18.1 preflight/compare、18.2、17.2R identity 等） | 冻结判据/价格 | 2026-11-04 | **active（冻结，不得改字节）** |
| `plans/manifests/` 其余约 200 个 | 旧窗口 | ≤2026-11-02 | historical |
| `plans/archive/` 150 文件 | 历史快照 | ≤2026-11-03 | archived |
| `reports/taiji_selection_routing_18_1_compare_20261104.json` | 18.1 正式判决 | 2026-11-04 | **active（判负证据）** |
| `reports/taiji_cursor_utf8_18_2_20261104.json`（＋A/B 逐例） | 18.2 验证 | 2026-11-04 | **active** |
| `reports/taiji_data_premise_18_0_pricing_20261104.json` | 18.0 定价 | 2026-11-04 | **active** |
| `reports/taiji_preflight_18_1_20261104.json` | 18.1 预检 | 2026-11-04 | active |
| 其余 16–18 报告（44 份，按 `_16/17/18_` 命名） | 探针/预检/结案 | 2026-11-04 | 证据（只读） |

## 7. 收敛与清理记录

- 原 `MAINLINE.md` 正文（405 行 / **177,021 字节**）**逐字节归档**至 [archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md](archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md)，`MAINLINE.md` 只保留单入口与锚点桩。
- 新增旁挂（不改旧字节，共 **1,356 字节**）：`CONSOLIDATION_20261103.md.superseded.json`（317）、`REVIEW-PACKET_20261102.md.superseded.json`（373）、`09_NEXT_MAINLINE_PLAN.md.superseded.json`（282）、`PRICING-17_3_SELECTION_ROUTING_20261103.md.superseded.json`（384）。
- 删除：**0 个文件 / 0 字节**。现有“死草案”均为被引用的兼容入口或历史证据，按“保留全部判定/判负报告与证据 bundle”处理；无引用且方向已死的独立草案本次未发现。
- 新增唯一轨道 [ROADMAP-NEXT_20261104.md](reference/ROADMAP-NEXT_20261104.md)（**11,395 字节**）：19/20/21 包步骤表、六项字段、预算上限、决策树与本轮不做清单。
- 合计新增字节：177,021 + 1,356 + 11,395 = **189,772 字节**（1 个归档文件 + 4 个旁挂 + 本文件 + 轨道文件）。
