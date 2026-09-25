# Seed 开发计划索引（2026-09-25 v3，编号体系规范化）

> 用途：**唯一的计划入口**。v3 新增**编号体系**（一个前缀一个意义，§0）；
> v2 修正层级：**主线是 M 系列（里程碑）**，A/B 是**支线**（欠账修复 / 架构债），与主线**并行、互不阻塞**。
> 历史文档**冻结、不重命名**（改名会断日志与提交的链接），用 §4 的**新编号 ↔ 旧文件名**映射检索。
> 冻结判据/预注册/冻结证据只追加；本索引随状态**就地修订**。

## 0. 编号体系（**一个前缀一个意义**；新增文档强制，存量不动只映射）

**格式**：`<类别>-<归属>-<序号>[_<主题>][.<日期>]`
—— **类别**说"这是什么性质的文件"，**归属**说"属于哪条线/哪个里程碑"，**序号**是该归属内的顺序。

| 类别前缀 | 意义（只放这一类东西） | 旧名对应 |
|---|---|---|
| `PLAN` | 计划/提案/路线（**非判决**，可修订） | *_PROPOSAL、*_BRIEF、*_MECHANISM_DESIGN |
| `SPEC` | 判据/预注册/合同（**冻结件，只追加**） | *_PREREG、*_CONTRACT、*_BUDGET |
| `EXP` | 实验执行与判决（训练臂/评测/计分的**落盘件**） | eval_taiji_*、train_taiji_*、reports/*.json 判决件 |
| `DIAG` | 诊断与归因（探针/审计，**非判决**） | probe_taiji_*、audit_taiji_*、diag_* |
| `FIX` | 缺陷修复与守卫 | *_fixes、tests/*_contract |
| `MS` | 里程碑判定/批准 | M5_EXIT_APPROVAL |
| `CONV` | 口径/约定 | metrics_conventions、FOLDER_STRUCTURE_RULES、REPO_HYGIENE_RULES |

**里程碑**：`M1..M6` 不变（大阶段主线，见 §1）。
**归属**：`R2`（M5 排除项调查，已收束）/ `A`（修复线，见 §2）/ `B`（架构债线，见 §3）/ `M6`（产品交付，见 §1）。
**脚本前缀**（代码件，另有约定）：`train_/eval_/verify_/build_/probe_/diag_/audit_/score_` 说"脚本干什么"，
与文档类别前缀**互不混用**。

**存量处置**：历史文件**冻结、不重命名**（改名断日志与提交链接）；§4 给**旧名 → 新编号**的映射；
**新增文档一律按本表**（例：T8 若今天新建应为 `PLAN-A-08_mechanism-design`）。

## 1. 主线：M 系列（里程碑）

| M | 名称 | 状态 |
|---|---|---|
| M1–M4 | 早期里程碑 | ✅ 已完成（详见 01_SCOPE_AND_PHASES） |
| **M5** | 限定退出 | ✅ **2026-09-20 批准落盘**（[批准书](../reference/M5_EXIT_APPROVAL_20260920.md)） |
| **M6** | **desktop / 产品交付（当前大阶段）** | 🔄 **并行会话推进中**（2026-09-21 起 7 提交：Electron 壳迁移→Python 载荷→安装-运行-卸全链→GPU 降级→logo 候选） |

### M6 未闭合五项（2026-09-21 活动卡 §5.7）

1. logo 判定（两轮候选后**待所有者输入**）
2. 托盘退不出复现（**待所有者输入**）
3. UI 交互其余项
4. 跨壳孤儿回收（技术债）
5. 单实例锁裁定

## 2. 支线 A：R2 欠账修复线（语言读出/答对）

> 来历：M5 退出时 R2 语言能力列为**显式排除项**（欠账保留在账）。调查已收束（归因定案），
> 修复由**并行会话**以 A2 复制回路推进。**与 M6 并行、互不阻塞**（负载边界：R2 训练等待期
> 不并行 ≈30 min 级重构建/重评测）。

| 步 | 名称 | 状态 | 判据 |
|---|---|---|---|
| **A-1** | R2 归因 | ✅ **定案** | CAP D+E=0/36 是「先告知→后提问」与"无归纳机制"的结构性错配 |
| **A-2** | 复制回路（写入门 + F1 直读，gate 零初始化位级不变） | 🔄 并行会话推进 | §4.1 存在性判据已通过（greedy 首字节=0xE9=「阿」） |
| **A-3** | M-3：问答结构语料 + 召回条件发射训练 | ⏳ **待 A-2 判决**（A-2 若 CAP>0 则跳过） | CAP D+E 机器计分 > 0；详细设计见 §5 |
| **A-4** | 可分离读法推广到主训练线 | ⏳ 待 A-2 成立 | `predictive_context_region0_only`（默认关、掩码式、守卫 4 绿） |

**支线 A 唯一下一步**：**等 A-2 的 CAP D+E 读数** —— **>0 ⇒ 走 A-4**；**仍 0 ⇒ 启 A-3**。

## 3. 支线 B：架构债线（审计修复）

| 步 | 名称 | 状态 | 判据 |
|---|---|---|---|
| **B-0** | P0+P1+P2 三轮修复 | ✅ 完成（116 测试） | `project_audit_fixes{,_p2}_2026-08-23.md` |
| **B-1** | `detect_modality` 抽为纯函数（无 self） | ✅ 完成 | 等价守卫 3 绿（`a6895b6a`） |
| **B-2** | `_infer_domain` 迁移完成（neurons 提升为参数；黄金向量 330 格等价）✅；`_reencode_domain_generation_context` 挂起（需 neuroplex tokenizer 夹具） | ⏳ 留 B-3 | 见 §6 |
| **B-3** | Cortex 神对象完整拆分（路由/域推断/生成三簇） | ⏳ **需负责人签字** | 每迁一簇配等价测试；冻结基线，禁静默漂移 |
| **B-4** | 核心推理路径覆盖率 ≥60% | ⏳ 待 B-3 后 | — |

## 4. 存量映射（**新编号 ↔ 旧文件名**；旧文件冻结不改名）

| 新编号 | 旧文件（`plans/reference/` 与 `scripts/training/`） |
|---|---|
| `MS-M5-01` | `M5_EXIT_APPROVAL_20260920.md` |
| `EXP-A-01` | `M5_R2_READOUT_RETRAIN_*`（A/B/C 合同、判决与 runner） |
| `PLAN-A-02` | `M5_R2_COMPOSITION_BINDING_BUDGET_PROPOSAL_20260923.md` |
| `SPEC-A-03` / `EXP-A-04` | `M5_R2_T1_T2_PREREG_20260923.md`（预注册）与其轨迹/判决件 |
| `SPEC-A-05` / `EXP-A-06` | `M5_R2_T3_FABRIC_WRITE_PREREG_20260923.md` |
| `SPEC-A-07` / `EXP-A-08` | `M5_R2_T4_SLOW_ALL_PREREG_20260924.md` |
| `SPEC-A-09` / `EXP-A-10` | `M5_R2_T5_REGION0_CUE_PREREG_20260924.md` |
| `SPEC-A-11` / `EXP-A-12` | `M5_R2_T6_CAP_PREREG_20260924.md`（判决件 `reports/taiji_r2_cap_checkpoint_scores_20260924.json`） |
| `SPEC-A-13` / `EXP-A-14` | `M5_R2_T7_MASKED_A_PREREG_20260924.md`（判决件 `reports/taiji_r2_masked_arm_a_20260924.json`） |
| `PLAN-A-15` | `M5_R2_T8_MECHANISM_DESIGN_20260925.md` |
| `PLAN-A-16` | `M5_R2_ARCH_LEVEL_PROPOSAL_20260925.md` |
| `FIX-B-01` | `project_audit_fixes_20260823.md` / `project_audit_fixes_p2_20260823.md` |
| `CONV-B-02` | `docs/metrics_conventions.md` / `docs/REPO_HYGIENE_RULES.md` / `docs/FOLDER_STRUCTURE_RULES.md` |

## 5. A-3（M-3）详细设计：**问答结构语料 + 召回条件发射训练**

**目标**：让 F1 读出在"问题条件"下学会**发射答案字节**（T8 已判 (c)：读出没学过"由召回内容发射答案"）。

* **语料来源**：`data/simple_zh/dialogue_extended_clean.jsonl`（已有 问：/答： 结构）。
* **语料格式（设计）**：每行 JSON
  `{"text": "<告知>。<提问>？<答案>。"}`
  例：`{"text": "我叫阿岩。我的名字是什么？阿岩。"}`。
  由 dialogue 的 问：/答： 对改写成此形，**再注入合成的 given-then-ask 对**（不同名字/地点/物品），
  避免只学表面问答形状。
* **⚠️ 防泄漏（沿用指标白皮书）**：合成对的"告知内容"**必须与 CAP-43 评价集的答案不相交**
  （否则 CAP>0 只是测试集记忆）。评价集 `cap0_eval_set_v2.json` 仍作**持有外**基准，不进训练。
* **训练制度**：F1 读出（`readout="predictive"`）全程 `learn=True`；
  `use_memory=True, use_identity=True`（召回条件存在）；写入面开放（Fabric 写入允许）。
* **判读线（冻结）**：CAP D+E 机器计分 **> 0**（对照 `t1@5M=0`、`base=0`）；成句率只报不判；
  0 ⇒ 记"读出对问答条件也无通路"（升级 B/架构侧）。
* **预算**：5M 判读 ≈ 5.5 h → 16M 定论 ≈ 22 h（视判读线定）。

## 6. B-2 / B-3 详细设计：Cortex 神对象拆分

**模式**（已建立样板 `B-1`）：纯函数抽离 + staticmethod 委托 + 黄金向量等价测试。

* **成员**：`detect_modality`（已迁，无 self）✓；`_infer_domain`（依赖 `self.neurons`）；
  `_reencode_domain_generation_context`（依赖 `self.neurons`）。
* **方法**：把 `self.neurons` 提升为参数 ⇒ **签名变 ⇒ 真重构 ⇒ 需负责人签字**。
  每迁一簇，先在 `tests/` 立**黄金向量等价测试**：同一输入下 `Cortex.方法(x)` == `helpers.函数(neurons, x)`
  **逐位相同**，绿后才迁；迁后立刻跑全量测试。
* **顺序**：B-1 `detect_modality` ✓ → B-2 `_infer_domain` → B-2 `_reencode_domain_generation_context`
  → B-3 完整拆分（路由集群 / 域推断集群 / 生成集群）。
* **风险**：冻结基线的静默数值漂移 ⇒ **等价测试先立、不绿不迁**；不签字不动。

## 7. 里程碑索引

| M | 名称 | 状态 |
|---|---|---|
| M5 | 限定退出 | ✅ 2026-09-20 批准落盘 |
| M6 | desktop / 产品交付 | 🔄 并行会话推进中 |
