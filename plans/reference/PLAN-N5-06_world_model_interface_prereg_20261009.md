# PLAN-N5-06：世界模型接口合同补齐（实施预注册，2026-10-09）

> **状态**：起草即冻结——§2/§3 的判据与守卫在任何产品码改动**之前**落盘；此后只许带日期就地更正或再次升版。
> **为什么是产品码**：09 §2 N5 的步骤③＝"世界学习器首次真实调用"。㊵-641 的普查仪把"接口齐不齐"
> 从散文变成读数，现读是 `contract_incomplete`。本件只做**设计**：补齐哪两法、"补齐了没有"的读数怎么取，
> 先冻；实施与任何长跑另批（动 `taiji/` ⇒ owner；跑训练 ⇒ owner 再批一次）。
> **本件明确不承诺**：不新增第三份"世界模型"实现，不重造 rollout/评测链，不动产品默认位。

## §0 已实测的前置事实（逐条带出处，全部来自本会话工具输出）

- **F-1 缺哪两法**：`scripts/training/audit_taiji_n5_world_model_contract.py` 现读
  `contract_verdict=contract_incomplete`、`missing_verbs=["propose","snapshot_restore"]`；
  被检对象是 `taiji/world_learning.py:1147` 的 `WorldDynamicsLearner`，公开方法 **8 枚**
  （`register_open_set`/`register_schema_alias`/`prune_schema`/`rollback_schema`/`record_schema_feedback`/
  `predict`/`fit`/`online_update`），快照只存在于**私有** `_snapshot_state_dict`（:1183）。
- **F-2 两个"命中"只到名字级**：`observe` 由别名 `online_update` 命中、`feedback` 由
  `record_schema_feedback` 命中 ⇒ 语义是否等价要读签名，普查器自己在件里写
  `reading_limit="method-name match only; semantic equivalence needs a signature read"`。
- **F-3 同类语义已有先例**：`taiji/world_evolution.py:235 def checkpoint()`／`:270 from_checkpoint()`
  是一对**公开**的存取（`NativeWorldPredictionTrainer`）；`taiji/world_learning.py:1748 def rollout_episode(...)`
  已在做多步展开。⇒ 补齐 `snapshot/restore` 应照这两处的**既有形状**，不新造第三种信封。
- **F-4 "零真实调用"只在语料链成立**：`scripts/training/train_seed_corpus.py` 对该类引用实测 **0**，
  但生产/训练面合计 **73 处、19 枚文件**（`taiji/foundation_training.py` 12 处、
  `scripts/training/train_taiji_world_action.py:129` 真的构造）。09 §2 N5 那句已就地打日期戳加范围限定（㊵-641）。
- **F-5 读数件在册**：`reports/taiji_n5_world_model_contract_census_20261009.json`（87 行）；
  钉子 `tests/taiji_native/test_n5_13_world_model_contract_census_contract.py`（5 passed）。

## §1 本件要买到的能力（一句话）

让 `WorldDynamicsLearner` 满足 09 §6 那份合同的**两条缺席项**——带预算的候选生成
`propose(goal, state, budget)` 与公开成对的 `snapshot()/restore(payload)`——并让"恢复之后能完整继续"
变成一条可红的读数，而不是一句设计意图。

## §2 判据（先冻；跑后不得改）

1. **J-N6a-1（`propose` 在场且预算是显式参数）**：普查器对 `propose` 的命中值必须从 `[]` 变成
  `["propose"]`，且其签名里 `budget` 是**第 3 位显式参数**（AST 取 `args`，不接受靠 `**kwargs` 或实例属性
  传预算）。判据式＝`matched_methods["propose"] == ["propose"] ∧ signature.index("budget") == 2`（0 基）。
2. **J-N6a-2（成对公开存取）**：`snapshot` 与 `restore` 两名必须**同时**公开在场
  （`missing_verbs` 由 `["propose","snapshot_restore"]` 变成 `[]`），且 `restore` 只接
  `snapshot()` 出版过的信封 ⇒ 信封外形状必须响亮拒绝，不许静默补齐。
3. **J-N6a-3（"完整继续能力"是读数不是意图）**：同一初始化学员（同 seed、同 schema）走两条路
  ——路 A 连续 `online_update` **8** 步；路 B 前 **4** 步后 `snapshot()`、`restore()`、再走 **4** 步。
  判据＝路 B 末态参数摘要与路 A **逐位相同**（`content_digest` 相等），且两次前向对同一
  `(state, action)` 的 `predict` 输出最大绝对差 `== 0.0`。任一项不等 ⇒ 判 `not_wired`，本件不得发表
  "可完整继续"这句。
4. **J-N6a-4（默认关逐位不变）**：不给任何新旗标时，语料训练链对**同一固定题面**的观测摘要必须与
  HEAD 逐位相同（尺＝㊵-636 那把：`git archive HEAD taiji seed` 解到仓外、同一 tiny 配置、
  同一 64 字节输入、一次 `content_digest`）。现读锚点：`9d1e06218e09b8dd3600800670a2debac2d1e881811a59f6a8dfd19a5442f547`
  （位置关）／`c0706e00ad9907621d985c585401a120d19177245aad72db34c1ff36a227d89d`（位置开）／
  `ae3177e00b4d937cddd571ebae9e2fc64dfca4567a2b49c42d2adb3d869ed3a0`（发育叠加层＋位置关）。
5. **J-N6a-5（不越权：默认位、参数量、存档摘要）**：① 产品默认位一律不动（`readout_utf8_position_input`
  默认仍 `False`、发育迁移默认仍关、世界学习钩子默认仍关）；② `propose` 必须是**只读**候选生成
  ⇒ 参数数量差 `== 0` 且既有存档摘要不变；若实现不得不新增张量，则触发"参数面加新张量的三件配套事"
  全套（窄口径钉子＋旧 payload 按零补＋`W=0` 回退连平手一起钉），不许静默过。
6. **J-N6a-6（"被走到"必须能在面件里读到）**：挂钩子之后，面件（`tail` 行或面头的发育/世界块）里
  必须出现世界侧的在场计数键，取值按 `'k' in block` 判在场、`.get()` 判值；缺键 ⇒ 判
  `ran_not_measured`，不许读成"喂了 0 次"。

## §3 守卫（fail-closed，逐条点名拒判形状）

- **G-N6a-1 普查器不许被改宽**：本件实施**不得**修改 `CONTRACT_VERBS` 的别名集来让读数变绿。
  若真要改取法，另立仪器债并配"改前必须为缺、改后必须为齐"的双向测（㊵-629 的教训：改判别式两侧都能为假才叫修好）。
- **G-N6a-2 复用而非重造**：`snapshot/restore` 的信封形状必须与 `NativeWorldPredictionTrainer.checkpoint()`
  同源或显式声明差异；`propose` 内部不得重抄 `rollout_episode` 的展开逻辑（重抄＝另起一条会腐化的生成链）。
- **G-N6a-3 预算语义要能红**：`budget` 取 `0` 时必须返回**空候选集**且**不改变任何参数**；
  取 `1` 与取 `4` 时必须能给出条数差（`len(candidates_4) >= 1 ∧ len(candidates_0) == 0`）。
  一条恒为真的预算参数不配当合同（本仓反复为"看起来接了其实不动"付过学费）。
- **G-N6a-4 三对照仍归评测面**：本件只补接口，不改 09 §6 的对照形状（无支持／打乱支持／冻结更新）；
  任何"接口补齐 ⇒ 能力提升"的推断一律禁发，能力结论要走 `WorldEpisodeEvaluator` 那批读数。
- **G-N5e-5 同族分层**：本件判绿只允许写"合同齐了＋默认位不变＋恢复后能逐位继续"；
  "世界模型真的在训练里起作用"必须等面件在场计数与评测读数（J-N6a-6）落地才能说。

## §4 失败出口（预先指定谁判据谁定位）

- J-N6a-3 的摘要差 ≠ 0 ⇒ 判 `not_wired`：`restore` 没把优化器状态/EMA/注册表版本一起带回，回退改动并登记缺哪一块。
- J-N6a-4 任一格摘要不同 ⇒ 本件**不是**零行为改动：点名是哪一格，回退，结论收回。
- J-N6a-5 的 `propose` 造成参数增量 ≠ 0 ⇒ 当场停手（说明它不是只读），回 §2 第 5 条走三件配套事或改设计。
- 普查器 rc=2（取法失败）⇒ 那是"没算"，本件不出任何结论，先修面再谈判据。

## §5 预算与前置（不预先声称跑得起）

- J-N6a-1／2／3／5：**零训练**，CPU，AST 与进程内小夹具（8 步/4 步量级），预计一轮内可判。
- J-N6a-4：HEAD 对表法两趟（同一 face 脚本），零训练。
- 实施动 `taiji/world_learning.py`（新增两个公开方法）＝产品码 ⇒ **owner 批**；
  接进语料训练链（`train_seed_corpus.py` 挂钩子）是**第二格**，需另交调用点全清单测＋默认关逐位不变证明。
- 长跑（步骤③ 的"真跑"）：不在本件，按 09 另批。

## §6 发表资格前置（末尾三行）

- 预期为绿的成员：J-N6a-1／J-N6a-2（两名在场）、J-N6a-5（默认位与参数量不动）、G-N6a-3 的 `budget=0` 支。
- 仍红时收回的句子：「世界模型接口已满足 §6 合同」「恢复之后能完整继续」；届时点名是 J-N6a-2 还是 J-N6a-3 拦的，
  并把缺陷登记进 05（不悄悄重试、不改普查器取法）。
- 发表资格前置：必须先有 §3 的三条能红的守卫实测（预算 0/1/4 的条数差、信封外形状被拒、逐位继续），
  才允许引用"合同齐"；引用时带范围限定——"名字在场"、"语义等价"、"训练链真走到"是三件事，各自要各自的证据。
