# 09.6.1：五项工程补齐合同（不授予正式训练）

> 文档状态：[合同] — owner 已于 2026-11-01 在 [09.6 §7](SPEC-TAIJI-CORE_PRICING_20261101.md#7-出口与交付) 批准 `approved_for_r029_implementation`。本合同只冻结五项工程缺口的实现与验收；**不批准 09.7 正式训练或有界能力比较**。调度见[MAINLINE](../MAINLINE.md)。

日期：2026-11-01。对象为 R021 参考图、R023 出生、R024/R027 主体恢复与 R025 活动循环。`taiji/living_core.py` 字节保持不变；新增能力只能通过独立模块或显式扩展接入，并在报告中绑定源码 SHA。

## 1. 范围与边界

- 只使用合成/公开中文夹具；不读 sealed、不改产品默认、不启动持续服务。
- 允许工程性优化器更新与参数变化，但更新后的参数只用于验收，不发布为模型。
- 所有新增门必须能失败；不得用“非零梯度”“有字段”“journal 有条目”代替实际学习、保持或贡献。
- 若任一实现使长生涯单步更新成本偏离 09.6 超过 20%，必须重新定价并修订 09.7 预算后再请 owner 裁定。

## 2. 五项交付与工程门

### 2.1 功能约束更新（算法 §10.2）

交付一个受约束反馈更新器：对新任务梯度 `g_new` 与一组旧能力梯度 `g_old,j` 求

\[
\Delta^*=\arg\min_\Delta \frac12\|\Delta+\eta g_{new}\|^2,
\quad g_{old,j}^\top\Delta\le\epsilon_j,\quad \|\Delta\|\le\rho.
\]

工程门：

1. KKT/可行性残差 ≤1e-8；约束违反 ≤1e-12。
2. 在 09.6 中文夹具上，8 步受约束更新满足 `fit NLL` 下降 ≥0.010 且 `retention drift` ≤0.030。
3. 同夹具无约束 Adam(lr=1e-3) 必须复现超过 0.05 的漂移，作为负对照；不得删除或放宽该对照。
4. 记录每步 CPU/wall、约束激活集、投影前后范数与 Δθ/ULP。

### 2.2 活动策略学习

交付活动价值头与策略更新：同一活动特征驱动 policy/critic，P_R 探针奖励来自活动前后合法探针 NLL、计算成本与旧能力惩罚；不使用 P_A/P_C，不读取 gold。

工程门：

1. 策略与价值参数 hash 前后不同，Adam/eligibility 状态可保存恢复。
2. 在同经历、同活动预算下，训练后模型臂的累计 P_R 奖励优于未训练模型臂与固定阈值对照臂；若无优势，如实记 `negative_activity_learning`。
3. wait/maintain 不得伪造微步；imagined 事件不写 store；每轮 journal 记录 chosen_by、触发量、奖励、TD/eligibility 范数与成本。
4. P_R 暴露计数逐轮入账，超过声明上限即停止。

### 2.3 lesion 验收

交付训练后移除新生支路的独立贡献检查（`lesion_trained`：只删新生单元，保留已训练父行）。

工程门（两面分开，不得合并）：

1. **恢复面**（未训练子模型）：coverage_birth 出生时函数对父逐位中性；对同一子模型的未训练副本执行 `lesion_trained` 后，其函数必须逐位回到父输出。
2. **贡献面**（已训练子模型）：训练后 `lesion_trained` 必须逐位保留已训练的父行、只移除新生单元。若训练后 holdout 收益 ≥1e-4，lesion 后收益必须减少 ≥1e-4，记 `lesion_removes_benefit`；若收益 <1e-4，记 `no_benefit_to_lesion`，**不得反签结构有效**；两者皆不满足记 `lesion_does_not_remove_benefit` 并判失败。
3. 已训练子模型在 lesion 后**不**声称逐位回到未训练父输出——父行已被训练改动，该声称与 `lesion_trained` 的定义矛盾。原措辞在实现前更正，更正落 03 台账。
4. lesion 成本与新增参数/单元数一起入账。

### 2.4 自然中文任务夹具

交付确定性中文连续任务生成器，覆盖当前值、历史值、更正、未知/缺资料与改写指称；runtime 只见原文字节、来源与时间，不见 gold 槽或答案标签。

工程门：

1. 每个任务实例固定应答轮分母；生成器自检配对数、答案唯一性或合法未知集合。
2. 对调/置换探针证明不能靠固定位置或最新记录捷径满分。
3. 夹具不读 sealed；来源、版本与许可字段完整。

### 2.5 子进程 CPU 计量

恢复 worker 自报 `worker_cpu_seconds`、`worker_wall_seconds` 与父进程等待墙钟；定价报告将子进程 CPU 并入恢复成本。

工程门：

1. 新字段不得改变 R024/R027 的 h/probabilities/参数逐位一致性。
2. 旧测试全部通过；新增回归证明字段确实来自子进程，而不是父进程复制。
3. 重新测量八边界恢复的总 CPU/wall。

## 3. 预算、出口与交付

- 09.6.1 工程验证预算：≤1800 CPU 秒 / 3600 墙钟秒；超限 `budget_stop`。
- 出口：`passed_r029_engineering_only`／`failed_constraint`／`failed_activity_learning`／`failed_lesion`／`failed_fixture`／`failed_worker_accounting`／`budget_stop`／`engineering_failure`。
- 交付：本合同、实现模块、测试、`scripts/training/verify_taiji_r029_engineering_gaps.py`、`reports/taiji_core_r029_engineering_gaps_20261101.json`、重新定价读数及 MAINLINE/03 同步。
- 通过只表示五项工程缺口按声明夹具成立；不表示 09.7 正式比较已准入，也不表示中文任务能力、成长收益或自主活动有效性成立。

## 4. 重新定价与 09.7 预算修订

§1.4 的 20% 偏离条款已触发，但触发方式需要分清：**09.6 表内的无约束声明更新单价未变**（`declared_feedback_updates` 与 `run_career` 默认 `retain_outputs=True` 的行为字节未改），偏离来自新增的约束更新规则与新增面。

实测单价（原理形状、09.6 中文长生涯、单线程 CPU float64）：

| 面 | 09.6 单价 | 09.6.1 实测 | 偏离 |
|---|---|---|---|
| 长生涯无约束声明更新 | 2.500 CPU 秒/步 | 未改，仍为 2.500（本轮 control 段 28.5 CPU 秒与 09.6 单臂 28.406 互证） | 0% |
| 长生涯功能约束更新（含实际探针与回退） | 未实现 | **8.830 CPU 秒/接受步**（8 步用了 11 次尝试，即每次尝试 6.42） | 新面，为无约束的 3.53 倍 |
| 单臂（3 前评＋8 步＋3 后评） | 28.406 CPU 秒 | 无约束臂不变；约束臂 **77.3 CPU 秒** | 约束臂为 2.72 倍 |
| 新生恢复 worker | `cpu_seconds` 记为 **0.0** | 子进程自报 **0.8125 CPU 秒**（bundle 77,304,316 字节与 09.6 逐字节同面，optimizer state 123 项）；父侧等待 CPU 为 0.0 | 09.6 把恢复成本记为零，非低估而是缺席 |
| 八边界恢复序列 | 只见父侧 | 父侧 2.391 ＋ 子进程合计 1.328 CPU 秒、子进程墙钟合计 1.634 秒、序列墙钟 16.117 秒 | 新增子进程账 |
| 活动学习（256 训练轮＋3×32 评估轮，小形状） | 未实现 | **164.3 CPU 秒**（其中 256 训练轮 119.484） | 新面；原理形状未定价 |
| lesion 面（信用构建＋出生＋8 步训练＋holdout 评估） | 未实现 | **33.2 CPU 秒** | 新面 |

按此重算 09.7（六臂×3 seed，含每 seed 一次 lesion 面与一次活动学习面）：

| 配置 | 臂成本×3 seed | ＋lesion 99.6＋活动 492.9 | 对 G7 的 1800 CPU 秒 |
|---|---|---|---|
| 全无约束（09.6 基线） | 511.3 | **1104** | 余 696 |
| 仅候选臂用约束更新，5 对照臂无约束 | 658.0 | **1250** | 余 550 |
| 六臂全用约束更新 | 1391.4 | **1984** | **超 184，破 G7** |

墙钟按 09.6.1 实测比值 1.067 折算（全约束档约 2117 秒），三档均远低于 3600 秒上限；**CPU 是唯一绑定约束**。全约束档若把 seed 从 3 降到 2 则为 **1323 CPU 秒**，落回 G7 内。

因此 09.7 只有两条可执行路径，必须由 owner 选定，不得由实现方自行放宽：

1. **仅候选臂使用约束更新**（1250 CPU 秒，落在 G7 内）。代价：五个对照臂与候选臂使用不同更新规则，"约束更新的净贡献"这一问只能由候选臂对 `replay_only` 的差回答，不能由六臂同规则比较回答。
2. **六臂全用约束更新**（1984 CPU 秒，破 G7 184 秒）。需要 owner 批准把 G7 的 CPU 上限从 1800 抬到 ≥2100，或把 seed 数从 3 降到 2（1323 CPU 秒，落在 G7 内但削弱种子重复）。

另需同时裁定的两条面限制：活动学习面**只在小形状定价**，原理形状未测；Windows `time.process_time` 粒度约 15.6 毫秒，轻量 worker 可自报 0.0（本轮八边界中有一次 `activity_round` 与最后一次 `resolve` 即为 0.0），因此子进程 CPU 账只在该粒度以上有意义。

## 5. 执行读数（2026-11-01）

实现与交付：[constrained_learning.py](../../taiji/constrained_learning.py)、[activity_learning.py](../../taiji/activity_learning.py)、[native_task_fixtures.py](../../taiji/native_task_fixtures.py)、[growth_credit.py 的 `lesion_trained`](../../taiji/growth_credit.py)、[native_activity.py 的 `model_action`](../../taiji/native_activity.py)、[state_migration.py 的 `retain_outputs` 与 worker 自报成本](../../taiji/state_migration.py)、[测试](../../tests/taiji_native/test_r029_engineering_gaps.py)、[验证器](../../scripts/training/verify_taiji_r029_engineering_gaps.py)、[报告](../../reports/taiji_core_r029_engineering_gaps_20261101.json)。`taiji/living_core.py` 字节未改。计时数字以报告为准，本节只记判据读数与结论，避免副本过期。

§2.1 功能约束更新（原理形状，09.6 中文长生涯为 fit、旧能力约束只挂 retention 生涯、holdout 仅作评估不进约束）：8 步全部接受、共 11 次尝试（3 次实际探针回退、ρ 逐次减半）；fit NLL 降 **0.0307817**（门 ≥0.010）、保持漂移 **0.0162904**（门 ≤0.030）、ΔNLL_holdout **−0.0172613**；KKT 残差 **1.94e-16**（门 ≤1e-8）、约束违反 **0.0**（门 ≤1e-12）、互补松弛 5.37e-14。负对照 Adam(lr=1e-3) 同夹具漂移 **0.1111525**（门 >0.05，复现 R026 的 0.1105–0.1126 区间）。

**对照同时暴露代价，必须与门一起读**：无约束臂 fit 降 1.9725041、ΔNLL_holdout −0.3056635，其 fit 收益是约束臂的 64 倍、holdout 收益是 18 倍，但漂移突破 09.6 G1。因此本次能声称的只是"约束更新是当前唯一同时满足 G1 与 G2 的配置"，不能声称它学得更好。

§2.2 活动策略学习（小形状，256 训练轮＋3×32 评估轮，三臂核心参数摘要逐位相同）：policy/critic 摘要前后不同、learner 状态经 `torch.save`/`load` 逐位往返、暴露计数 256 未越上限、训练与评估逐轮账含 `chosen_by`/触发量/奖励/TD/eligibility 范数/成本。累计 P_R：训练臂 **0.0**、未训练臂 −0.0319197、固定阈值臂 −0.0198944，优势 **+0.0198944**，判 `activity_learning_advantage`。

**该优势的机制已查明，不是任务收益**：训练臂 32 轮全选 `maintain`，累计恰为 0.0，优势数值上等于对照臂烧掉的计算成本。成因是声明夹具只在队列里放一条外部消息，`interact` 在 32 个评估轮中最多合法一次，空转因而是结构性最优。报告以 `advantage_is_idle_cost_avoidance` 与 `advantage_disclosure` 显式披露；09.7 若要就活动学习或自主性下任何结论，必须先造一个触发反复出现的活动夹具。

§2.3 lesion（原理形状，coverage_birth 臂、09.6 冻结 lr=1e-5、8 步）：出生对父函数逐位中性 ✓、未训练副本 `lesion_trained` 后逐位回到父输出 ✓、训练后 `lesion_trained` 逐位保留已训练父行且只移除新生单元（258→256，8 单元/4 组）✓。但训练收益 **0.0201929** 中移除新生单元只减少 **4.33e-7**（占比 2.1e-5），远低于 1e-4 门 ⇒ 判 `lesion_does_not_remove_benefit`，本合同按预注册出口结案为 **`failed_lesion`**。

这是第三次独立指向同一结论的读数：R026 六臂预注册 negative、09.6 的 coverage−replay = −4.72e-7、09.6.1 的新生单元贡献占比 2.1e-5。**在当前夹具与 lr 下，覆盖出生这一结构生长机制没有可测贡献；观测到的全部收益来自继承父行的参数更新。** 该结论不得被读作仪器故障，也不得据此反签结构有效。

§2.4 自然中文任务夹具：5 类（当前值/更正值/未知缺资料/改写指称/来源冲突）；每实例应答轮分母 25/28/10/25/16，逐实例等于 `career_step_specs` 的 round 数；捷径尺有动态范围且无一满分——最新来源 2/5、最早来源 1/5、弃答 2/5；记录顺序反转后最新来源降为 1、最早来源升为 2，证明夹具对顺序敏感；两个运行时模块源码中 `expected_target`/`shortcut_label`/`category`/`gold` 令牌计数全为 0。

§2.5 子进程 CPU 计量：见 §4 表。新字段未改变 R024/R027 的逐位一致性（h 与概率逐位相同），R022–R027 回归 33 passed，09.6.1 新增测试 15 passed。

回归与门禁面：整树 `ruff check .` 与 `black --check .` 全绿（2371 文件）、`audit_frozen_marker_formatter_conflict.py --no-write` passed。mypy 核心面（`mypy --follow-imports=silent seed taiji`，CI 基线 0）本机实测 44 处错误分布在 11 个文件，其中 **29 处在本轮未改动的文件**（`living_core.py` 7、`native_content_core_v2.py` 8、`native_content_training_v3.py` 7、`native_content_training_v2.py` 2、`native_content_core.py` 2、`native_development.py` 2、`corrected_old_baseline.py` 1），另 14 处在本轮改动文件的**未改动行**上（`native_activity.py:41/277` 为 R025 策略头与 imagine 分支、`growth_credit.py:408-417` 为 R023 `allocate`、`state_migration.py:301` 为 R024 `load_subject`）。09.6.1 三个新模块单跑 mypy 为 `Success: no issues found in 3 source files`，即**本轮新增 0 处类型错误**；该面在本轮之前已红，属既有债，不在 09.6.1 范围内结清。
