# PLAN-N5-05：位置输入接入发育 F1 学习通路（实施预注册，2026-10-09）

> **状态**：起草即冻结——§2/§3 的判据与守卫在任何产品码改动**之前**落盘；此后只许带日期就地更正或再次升版。
> **为什么是产品码**：owner 弹窗 #16 在 `DEBT-G71` 三条例路里裁了**甲**（改产品把位置输入接进发育 F1 学习通路），
> 而不是乙（再造基件）或丙（换形状）。本件只做**设计**：接通与"接通了没有"的读数先冻，训练跑另批。

## §0 已实测的前置事实（逐条带出处，全部来自工具输出）

- **F-1 拒绝点在哪儿**：`Taiji.migrate_f1_to_developmental_synapses`（`taiji/model.py:1284`）第一件事就是调
  `_reject_position_input_without_learning_path`（`taiji/model.py:1131`），抛
  `readout_utf8_position_input is not wired to the developmental F1 learning path; use the plain predictive-readout learning chain`
  （实测原话来自 `output/n5_618_off/stdout.log` 的真跑回溯）。
- **F-2 没有可用的替代基件**：`checkpoints/*.pt` 全集 **13 枚**每一枚都
  `readout_utf8_position_input=true`（自述 tick 含 0／92／273）⇒ 绕开甲去 resume 只能在
  `Seed.restore`（`seed/model.py:240`）撞第二道拒绝 `checkpoint configuration does not match Seed`。
  两道拒绝合起来＝"同一基件续训"在今天的仓里不可执行（㊵-616／DEBT-G71）。
- **F-3 影子链本身不是问题**：通电与在场计数已直证（㊵-618／㊵-628：治疗臂
  `shadow_forward_hits=11735`／`shadow_learn_hits=11735`，对照臂同一块内三枚全 0）；
  学习侧母量两对臂同向 `not_holds`。⇒ 缺的从来不是影子，是**能算代价的 before 面**。
- **F-4 before 面已在库**：`reports/taiji_n2_cap0_before_20261008.json` 与
  `reports/taiji_n2_replay24_before_20261008.json` 的 `checkpoint` 都是
  `checkpoints/seed_a31self_with_circuit.pt`（带位置输入那枚）⇒ 甲接通后，PLAN-N5-04 的续训双臂
  不需要再造 before，只需按 `G-N5d-1` 做字节级同源核对。

## §1 本件要买到的能力（一句话）

让「基件带 `readout_utf8_position_input`」**和**「F1 迁移到发育突触叠加层」这两件事**同时成立**，
使 N5 的保持侧（`J-N5b-5`）第一次有可评的 before／after 对；接不上就退回乙／丙，不许把 `raise` 删掉了事。

## §2 判据（先冻；跑后不得改）

- **J-N5e-1（默认逐位不变，硬前置）**：不开发育迁移时，改动前后对**同一固定题面**的读数必须逐位相同。
  尺沿用本仓已冻的那把（HEAD 对表 ＋ 内容摘要），新增的只有"改动前后"这一对；
  任一格不同 ⇒ 本件**不是**零行为改动，那句结论当场收回并点名是哪一格。
- **J-N5e-2（真接通，不是消音）**：给定「基件带位置输入 ＋ 请求迁移」这一形状：
  ① 不再抛；② **位置输入通路必须留下做功的证据**——一步 `learn` 之后，与位置输入相连的权重
  至少一枚的变化量 `> 0`（只断言"没抛异常"＝把 `raise` 删掉的假修，本件明确否证它）。
  判据写成 `position_path_delta > 0.0` 与 `n_changed_units >= 1` 的合取，两枚都在仪器里出数，不手算。
- **J-N5e-3（不越权，三条并列）**：① 产品默认位一律不动（`readout_position` 默认仍 True、发育迁移默认仍关）；
  ② `--no-readout-position` ＋ `--developmental-fast-slow` 的旧形状必须**照旧工作**（不新增拒绝路径，
  也不把旧路径变静默）；③ 参数量与存档 payload 摘要不变——若接通不得不新增张量，则触发
  「参数面加新张量的三件配套事」全套（窄口径钉子＋旧 payload 按零补＋W=0 回退连平手一起钉），不许静默过。
- **J-N5e-4（终判在保持侧，且需另批）**：接通之后才允许起 PLAN-N5-04 的续训双臂（改权重 ⇒ owner 再批一次），
  按 `J-N5d-1` 逐臂抄录 `retention_holds`／`cost_persists`；两臂同结论才算 `J-N5b-5` 有读数。

## §3 守卫（fail-closed，逐条点名拒判形状）

- **G-N5e-1 调用点全清单**：动 `:1131` 那道拒绝之前，先 `grep` 出
  `migrate_f1_to_developmental_synapses`／`_reject_position_input_without_learning_path` 的**全部**调用点与
  测试引用数并写进本件；引用数为 0 的分支必须补一支进程内测（否则改完没人能证它被走到）。
- **G-N5e-2 负对照**：构造"该抛仍抛"的对照形状（位置输入在场**且**学习通路未接通）⇒ 必须抛错；
  接通后 ⇒ 必须不抛。两支各跑一次，缺一支这条守卫就是恒真。
- **G-N5e-3 默认关逐位不变**：见 J-N5e-1；并核 HEAD 的摘要尺与本轮新跑**同取法**（分子分母同一集合）。
- **G-N5e-4 不新增静默**：任何"从抛错改成不抛"的路径都必须同时产生**可读出**的证据字段
  （接一个 `developmental_position_input_wired=true` 之类的自述到面/信封里，缺键按 `ran_not_measured`），
  静默成功与静默失败同样是债（㊵-590 那族的形状）。
- **G-N5e-5 结论分层**：本件接通成功只允许写"链路可用＋默认位不变"；
  "能力涨了吗"必须等 J-N5e-4 的保持侧读数，且按 J-N5b-6 的合取口径写，不许拿仪器绿当能力涨点。

## §4 失败出口（预先指定谁判据谁定位）

- J-N5e-1 红 ⇒ 改动带行为影响：点名是哪一格摘要不同，回退改动，本件降级为"仅登记设计约束"。
- J-N5e-2 的 `position_path_delta == 0` ⇒ **判 `not_wired`**：说明只是消音，必须回退，不得改判据凑绿。
- restore 仍不匹配（第二道拒绝没解）⇒ 说明 config 侧还要一处改动 ⇒ 停手并回 owner 复裁（本件不擅自扩大改动面）。
- J-N5e-4 两臂结论不一致 ⇒ 记 `arm_dependent`，不发表方向性结论，先查两臂是否只差那枚旗标。

## §5 预算与前置（不预先声称跑得起）

- J-N5e-1..3：**零训练**，CPU，只读＋单测；预计一轮内可判。
- J-N5e-4：两臂 60k 续训（单臂 60k 从头档墙钟实测 396.803 s／334.638 s 量级）＋四张 after 面件；
  **改权重 ⇒ 需 owner 再批**，本件只冻形状不起跑。

## §6 发表资格前置（末尾三行）

- 预期为绿的成员：G-N5e-1（调用点清单）、J-N5e-3（默认位与参数量不变）、G-N5e-2 的"该抛仍抛"支。
- 仍红时收回的句子：「位置输入已接入发育 F1 通路」「N5 保持侧可判」；届时点名是 J-N5e-1 还是 J-N5e-2 拦的，
  并把缺陷登记进 05（不悄悄重试）。
- 发表资格前置：必须先有 §3 的调用点清单实测与 §2 的两支判据读数入库，才允许引用"已接通"；
  引用时带范围限定——"默认关逐位不变"与"开启后可学习"是两件事，各自要各自的证据。
