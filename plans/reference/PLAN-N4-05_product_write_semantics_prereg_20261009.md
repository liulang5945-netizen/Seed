# PLAN-N4-05 设计预注册：让寻址材料来自**产品自己的写入路径**（判据先冻，不开跑）

- 主线：N4 第二格。owner 第十三次弹窗 (甲) 批的是"开写入格（产品码）"。
- **本篇要改的是那条批文的前提**：我原以为产品链里没有情景写入、需要新加语义。
  查证结果不是——写入**早已在产品里**（见 §1）。所以我按纪律先把事实钉清楚，
  不把训练侧口径冒充成产品语义，也不去改一个已经存在的东西。
- 性质：设计预注册。不产读数、不改码；事后只许就地日期更正或升版。

## 1. 现状事实（本轮逐条查证，含我自己的一处错）

- 产品链**已有**情景写入：`taiji/adapter.py` 在 `settle_action`（函数起始 `:11317`）内部
  构造 `EpisodicMemoryRecord`（`:11668`）并调用 `self._episodic_memory.write(record)`（`:11699`）；
  cue 由产品状态导出，记录里带 `outcome`／`world_transition`／`prediction_error`／`action_intent`
  与 `event_ids`／`assembly_ids`／`object_ids`／`relation_ids`。
- 因此"挂载空库不改变观测路径"（㊵-601④）的成因不是产品缺写入，而是
  **N4 的语料训练链与寻址面只调 `observe`，不落定动作回合**（不 `act`＋`settle_action`）。
  回合驱动在**评测侧**早已存在并可写：`eval_taiji_p4_procedural_runtime.py:88` 的
  `model.settle_action(1.0, learn=False)`、`eval_taiji_p3_risk_sensitive_execution.py:165-166` 的
  `act`＋`settle_action`，以及 `eval_taiji_m4v2_r4_shadow.py`/P6 那条名为
  `episodic_outcome_write` 的病灶（P6 还直接读 `episodic_memory_count`）
  ⇒ 产品写入路径**可达且已被评测用过**，本格要做的不是"加写入"，而是"用哪一类回合造可寻址材料"。
- 我先前草稿里写过"产品链内没有任何 `store.write(` 调用点"——**那句是错的**，
  被 `grep` 直接否证（`taiji/adapter.py:11699`）。本行保留是为了留下"没查就写"的代价记录。
- 训练侧那套按边界符攒 32 字节写一条的口径（`train_seed_corpus.py` 的 `--episodic-mount`）
  是**仪器口径**：它产出的 `wrong_top1_rate=0.06` 与 S4 的 `0.33` 都属 `mount_layer="harness"`。

## 2. 判据（五条，先冻）

- **J-N4e-1（材料必须来自产品写入路径）**：产品级寻址面的全部记忆条目，只允许由
  `settle_action` 里那条既有写入产生；**禁止**评测件自己调 `store.write(...)`。
  **判对条件**：生成件里对 `store.write(` 的调用点数为 0，而 `store.count > 0`
  （计数由 `git grep` 或 AST 机械数，不靠眼看）。
- **J-N4e-2（回合驱动可复现）**：材料必须由**确定性的动作回合**生成（seed 固定、
  动作序列与观测序列同源自述），并出版 `episodic_writes`、`write_trigger="settle_action"`、
  `cue_dim` 三枚自述；缺任一 ⇒ 判读器 `rc=2` 记 `ran_not_measured`。
- **J-N4e-3（默认关逐位不变）**：`episodic_memory_default_mount=False` 时同一回合驱动的
  观测摘要必须仍等于仓外 HEAD 基线（当前值
  `71afabbc9b7dc0990da6b10bfc036bee049f4a18170596c1a427b41eb696a126`）且 `count == 0`；
  反面必须实走：默认开 ⇒ 同输入 `count > 0`。
- **J-N4e-4（两档不可比就不比）**：产品档（回合写入、cue 来自认知状态）与训练侧档
  （前 32 字节 cue）的 cue 形状不同 ⇒ **禁止**跨档比 `wrong_top1_rate` 数值或据此说
  "产品寻址更好/更差"；只许各自报自己的尺，并自述 `cue_source`。
- **J-N4e-5（"产品级"名号的前置）**：只有 `mount_layer="product"`、`product_default_mounted=true`、
  `episodic_writes > 0` 三者同真，读数才允许挂产品名号；否则写成
  "运行时挂载、内容来自仪器写入"这种分层表述（`evidence-layering` 的四层口径）。

## 3. 实施前置（顺序）

1. 本篇入库（㊵-604 登记）。
2. **向 owner 回报前提变了**：批文说的是"加产品写入"，而写入已在；
   真正缺的是**驱动回合**（`act`＋`settle_action`），那是评测/仪器侧改动，**不需要动产品码**。
   若 owner 仍要产品侧改动（例如让 `observe` 流自动收束回合），那是**另一个语义决定**，要重新裁。
3. 反例探针先跑（J-N4e-3 能为假、J-N4e-2 缺键两面都走），过才实施生成件与判读件。

## 4. 失败出口

- 若回合驱动在 tiny 配置下产生不了任何 `outcome` 落定（即写入分支实际不可达）⇒
  本格不产出"产品级寻址读数"，结论写成"产品写入路径在可用回合形态下不可达"，回 05 立债；
  **不许**退回用训练侧口径冒充产品档。
- 若 `count > 0` 但检索恒命中/恒失败（尺无动态范围）⇒ 报 `ruler_unusable`，不发表比较。
