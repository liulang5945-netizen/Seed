# PLAN-N1-05 · 逐步再激活的持久载体预注册（2026-10-14 起草；**本件不动码、不开跑**）

冻结时刻：2026-10-14（台账 ㊵-835）。批准出处＝十六裁⑴甲（owner 已批"开另案"，[PLAN-N1-04](PLAN-N1-04_s1b_reactivation_implementation_prereg_20261014.md) §5 批⑵ 的替代路）；登记＝**DEBT-G142①**。
**效力**：把 N1 阶段1（H-S1b）缺的"逐步且持久的记忆载体"落成**可执行规格＋先冻判据**；它本身不构成开跑授权、不构成动 `taiji/` 的授权——两件事要 owner 在本件入库后单独批。
**与 [PLAN-N1-04](PLAN-N1-04_s1b_reactivation_implementation_prereg_20261014.md) 的关系**：本件是**载体**（写入/入档/恢复/读取），PLAN-N1-04 是**消费侧**（生成期再激活）；载体绿 ≠ S1b 成立，**两案不许互相冒充**（DEBT-G142③）；S1 阶段1 继续顺延至本件全绿＋PLAN-N1-04 训练臂另批。
**09 §1.3 对账消化的条款**：V-11（M 跨时经历的答题路径实现）、V-19（回答计算链"会话和可见记忆"环的持久化）为核心；V-08（连续表示＋原始引用双通道——载体存投影，引用仍在证据层）、V-13（时间状态定"保留什么"——本件定义保留粒度＝逐步）、V-24（方案 A 证据记忆形态）为形状约束。

## §0 现读的代码位（否证链，全部读码所得；行号为 2026-10-14 HEAD 现读）

| 位置 | 现读事实 |
| --- | --- |
| `taiji/memory.py:1041-1066` | `EpisodicField.to_payload()` 的记忆侧只存**单枚** `last_event`（`:1063-1065`）——逐步轨迹不入档 |
| `taiji/contracts.py:1942-1961` | `EpisodicMemoryRecord`（frozen dataclass，`memory_id/episode_id/tick/cue/…`）＝仓内唯一"逐步且持久"的记录类型 |
| `taiji/adapter.py:11662-11699` | 该记录**只在 `settle_action`（行动路径）被写**；cue 取 `percept.features` 或 `world.latent` |
| `api/seed_runtime.py:449` 与 `:443-447` | 答题路径 `chat()` 不经 `settle_action`；生成前只调 `_record_told_history` ⇒ **每轮一枚池化向量的剪贴板**，无逐步轨迹 |
| `taiji/config.py:348` | `episodic_memory_default_mount: bool = False`（PLAN-N4-04 J-N4d-1）⇒ 默认装机连情节场都不挂 |
| `taiji/config.py:149` 对 `taiji/fabric.py:271-274` | 维度不符：`PerceptionConfig.feature_dim = 32`，而 `fabric.step` 的 `episodic_feedback` 硬要求 `(cortical_context_dim,)`，否则 `ValueError: episodic feedback does not match cortical context` ⇒ 现有记录类型**即便写入也不能直接喂再激活通道** |
| `taiji/copy_circuit.py:263-273` | 运行时属性先例：`endpoint_yield_override`／`successor_restricted_addressing` 均"不在参数面/payload 里，checkpoint 载入不受影响" |
| `scripts/training/probe_taiji_a30_stop_failure.py:1231` | 面脚本赋值先例：`substrate.copy_circuit.successor_restricted_addressing = True` 由旗标显式置位 |
| `taiji/config.py:135-141` | 入档兼容先例：`values.setdefault(...)` ⇒ 旧 payload 缺新键仍能载入 |

⇒ 否证结论（㊵-833 ⑨）：**答题路径没有逐步且持久的记忆载体**——S1b 要"沿存储轨迹重放"，而轨迹在答题路径上从未被存下、存下的（行动路径）维度不对、默认还不挂载。PLAN-N1-04 §1.4 的阻塞项由此解法承接。

## §1 载体形状与最小格（开关化，默认关）

1. **新增运行时属性开关** `persistent_stepwise_carrier`（默认 `False`）：照 `taiji/copy_circuit.py:270` 那条形做——**不在参数面/TaijiConfig 字段里**（`taiji/model.py` 的 `restore()` 要求配置与在档逐位相等 ⇒ 任何 `TaijiConfig` 新字段都会把旧档拒载）；面脚本照 `probe_taiji_a30_stop_failure.py:1231` 那条形显式置位。
2. **载体本体**＝按 `episode_id` 组织、每生成步一条、形状 `(cortical_context_dim,)` 的逐步投影列表：
   - 写入点＝**生成环内每步**（条件于当步实际参与再激活的皮层上下文；具体挂点＝substrate 顶层 payload 新键或同层在册结构，实施附件在"运行时属性开关＋setdefault 入档"两约束的交集中选定并给行号）；
   - **入档**＝新键进 checkpoint payload，照 `taiji/config.py:135-141` 的 `setdefault` 先例——**旧档缺此键必须仍能载入**；
   - **读取口**＝按 `episode_id` 取逐步列表，条目可直接作 `fabric.step` 的 `episodic_feedback` 实参（形状硬校验实测，见 J-N1e-5）。
3. **不改** `recall()` 池化语义、**不改** `memory_confidence_decay`、**不改** `consolidate()` 三处注入、**不改** `settle_action` 的行动路径写入 ⇒ 本件只加答题路径一条载体；行动路径与它互不冒充。
4. 维度基准确认：`cortical_context_dim` 的现值以在档基座（`checkpoints/a31_chunked_self/checkpoint.pt`）payload 现读为准，实施附件出版该值；**不许**把 `feature_dim=32` 的旧记录形状混进本载体。

## §2 判据（先冻五条，每条写"什么情况下为假"）

* **J-N1e-1 逐步性**：开关开的一轮答复结束后，载体内该 `episode_id` 的条目数 **== 该轮实际生成步数**（由面自述的步数计）且 **≥ 2**。为假＝条目数只有 1（那还是池化摘要换了个名字）或与步数不符（漏写/多写）。
* **J-N1e-2 持久性**：写完载体的进程做 save→restore 后，载体内该 `episode_id` 的全部条目**逐位相同**（逐条张量 `equal`）。为假＝restore 后丢失/变形 ⇒ "持久"不成立。
* **J-N1e-3 旧档兼容**：**不携带本载体键**的旧档（在档基座 `checkpoints/a31_chunked_self/checkpoint.pt`）在带本件代码的树上载入 rc=0 且摘要不变。为假＝载入报错或摘要变 ⇒ setdefault 兼容失败。
* **J-N1e-4 来源真实性（反证支，缺此支不结项）**：把载体内该 `episode_id` 的条目**清空**后再走读取口 ⇒ 读取面必须返回**空**或响亮拒绝；若仍能返回同样的逐步序列 ⇒ 读的是别的来源（`last_event`／剪贴板／现场重算）⇒ 判假，"逐步来源真是从在册载体读出来的"不成立。
* **J-N1e-5 维度兼容**：从载体取出的任一条目直接作 `fabric.step(..., episodic_feedback=<条目>)` 实参，形状硬校验（`taiji/fabric.py:273-274`）**不抛 ValueError**。为假＝维度不符 ⇒ 载体喂不进再激活通道，S1b 依旧不可测。

## §3 守卫（先冻，与判据同为动码前置）

* **G-N1e-1 默认逐位不变**：不带开关的臂对同面件跑比较器 ⇒ `identical=true`、`behavior_diff=0`、`schema_diff=0`；为假＝改动从默认路径泄漏（写入了任何东西都算）。
* **G-N1e-2 开关不在配置面**：`taiji/config.py` 全文 grep 本开关名 **0 命中**（机检入契约测）；在档基座 restore 路径 rc=0 ⇒ 配置逐位相等守卫未被触碰。
* **G-N1e-3 旗标互斥**：与 `--copy-successor-restricted-addressing`（阶段0）、`--copy-evidence-endpoint-yield`（S5）、`--answer-reactivation-replay`（PLAN-N1-04，若已实施）任一同开必须**响亮拒绝**（代码级 `RuntimeError`，沿用 G-S1-3 条形）。
* **G-N1e-4 件自述**：面件必须出版载体键名、`episode_id` 清单、每 episode 条目数、条目形状、`cortical_context_dim` 现值、写入/读取调用点行号；缺键按未测处理，不许补零。
* **G-N1e-5 覆盖率自述**：面收尾必须带 `kind=tail` 行与覆盖率 ≥0.95（㊵-484④ 教训：无 tail 的面会把静默截断读成完整分布）。

## §4 面、预算与排期（现在不花）

1. **零训练**：本件全部判据/守卫都是写入-持久-读取链取证，无训练臂。三支面＋一枚载入检查：
   (a) 开关开写入面（96 题、`max_length` 256，同阶段0 面形，约 13 分钟本机 CPU）取 J-N1e-1/2/4/5；
   (b) 开关关对照面（同参）取 G-N1e-1；
   (c) 旧档载入检查（零跑）取 J-N1e-3＋G-N1e-2。
2. 报价纪律：总机时 **≤ 30 分钟本机 CPU、零训练、零外部配额**；修复 1 次仅限仪器机械修复；判据级不重跑。
3. **排期硬前置**：不在任何在跑臂的窗口内动 `taiji/`（沿用 PLAN-N1-04 §4.3 的窗口约束）；动码格须 owner 单独批（§5）。

## §5 owner 决策点（本件不自批）

* 批⑴＝允许按 §1 动产品码（运行时属性开关＋载体写入/入档/读取＋§3 守卫单测），零训练取证在先；
* 批⑵＝三支面的机时（≤30 分钟 CPU）；
* 本件全绿后，PLAN-N1-04 §1.4 的阻塞项**解除**，S1 阶段1 训练臂按 PLAN-N1-04 §5 批⑵另批——**那是另一格，不在本件内**。

## §6 落笔即钉的三行

* **预期绿**：G-N1e-1（默认逐位不变）、G-N1e-2（配置面 0 命中）、J-N1e-3（旧档载入 rc=0）。
* **仍红时收回哪句**：J-N1e-1 红 ⇒ 收回"载体是逐步的"，改登记"仍是池化近似"；J-N1e-2 红 ⇒ 收回"持久"，载体降级为进程内临时结构（S1b 继续阻塞）；J-N1e-4 红 ⇒ **收回"逐步来源从在册载体读出"**，本件判未达成且不许以"功能可用"结项——来源真实性是 DEBT-G142② 的硬要求；J-N1e-5 红 ⇒ 收回"维度兼容"，S1b 依旧不可测。
* **发表资格前置**：本件全绿之前，N1 泳道状态仍是"S1a 读取可修但不充分、S1b 阻塞于载体"；任何"答题路径已有逐步记忆"的句子都不许写。
