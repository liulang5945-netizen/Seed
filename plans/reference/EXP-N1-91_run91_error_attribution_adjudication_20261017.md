# EXP-N1-91 · 2026-10-17 · 甲路判读：288 行的逐题错因归因出值，`selection_side=96/96` 且 `transition` 类零错

读数件＝`reports/taiji_n1_confirmation_error_attribution_run89.json`（单枚，`status=published`、`accepted=true`、`confirmation_queries=0`、`cpu_seconds=0.0156`）；仪器＝`scripts/training/attribute_taiji_n1_confirmation_errors_run89.py`（只读两枚在库件，`--readiness` 与 `--approved` 互斥，写靶已存在即拒）；预注册＝`plans/reference/PLAN-N1-91_per_item_error_attribution_run89_20261017.md`（`plan_sha256=88b938f59e16937e…` 由报告自述）。四条绑定守卫全真：B1 288 行＝9 终端 × 32 题且 id 集与题集相等；B2 题集摘要与封条相等（该枚件无 CRLF，raw 与归一两口径同值 `22ba37976bc6eaaa…`，两枚各自披露）；B3 逐行重算的三臂合计与封存聚合逐项相等（position 79／terminal 33／cut 80），九组 `summary` 同样逐组相等；B4 `boundary` 行的步数恒等式零反例、1 枚 `cap` 行停在 64 步（与 `protocol.max_generated_bytes=64` 同值，本件不猜该常量住在哪）。

## 1. 轴一（96 组三元组）与轴三（选择侧还是续跑侧）

`all_three_right=33`、`all_three_wrong=16`、`exactly_one_right=1`、`exactly_two_right=46`、`other=0`，五类之和 96（合格条件之一）。`exactly_one_right` 那一组缺的是 **position 与 terminal**、独自答对的是 **cut**；`exactly_two_right` 的 46 组里缺席的那一臂**全是 terminal**（`axis1_exactly_two_right_missing_arm={"terminal":46}`）。这两句合起来是"terminal 是能力底臂"的第三次独立复算，且**没有**一组是位置臂独自答对——与 EXP-N1-90 判否的臂差结论同源，不新增任何臂差主张。

轴三是本件的主要产出：**`selection_side=96`、`compounding_only=0`**，即 96 行自由生成失败的行，在教师强制下也全部没选中目标序列。`teacher_tokens_missing`（目标长度减教师命中数）分布为 1 枚缺 1＝15、缺 2＝10、缺 3＝13、缺 4＝6、缺 5＝21、缺 6＝22、缺 7＝9——缺 5 以上占 52/96。所以这不是"打分对、续跑崩"，而是**打分阶段就没把答案符号排到顶上**；§3 的第⑵条出口（续跑侧）在本件当场为假，第⑴条（读通路侧）为真。

## 2. 轴二（错形）与轴四（kind × seed × arm）

九十六行的形态只有两类非零：`first_symbol_wrong=69`、`prefix_then_diverge=27`；`empty_output`、`truncated_prefix`、`overshoot`、`equal_output_not_exact` 四枚恒为 0，`unclassified=0`，`no_boundary_cap=1`（单列，不与形态类相加）。按类分：`lookup` 25＋15、`revision` 29＋11、`unknown` 15＋1，而 **`transition` 类不在表里＝该类零错误**——72 行 transition（8 题 × 9 终端）全部 `free_exact` 且 `teacher_exact`。这条是本件最干净的一格：失败完全集中在 lookup／revision／unknown 三类，与读取臂无关。

`unknown` 的 16 行错里 15 行是首符号即错（§3 第⑷条出口同时为真：弃权表达那一格是短板，`answer="unknown"` 这个字面的首字符就没排上）。

按 seed 拆开看，失败不是均匀散布：seed11 与 seed12 **没有**一组三臂全错（`all_three_wrong=0`），seed13 有 16 组，且这 16 组恰好是 8 个 lookup 题与 8 个 revision 题全部（id 形如 000/001、004/005、…、028/029 的四格周期前两格）。`axis4` 里 seed13 的 terminal 与另两臂同形：`lookup 0/8`、`revision 0/8`、`transition 8/8`、`unknown 2/8` ⇒ **seed13 这一枚模型在 lookup/revision 两类上整体缺席**，而不是某一臂在它上面更差。这把 EXP-N1-90 §3 那句"失败画像按 seed 与类分布、与读取臂无关"从描述升级成可核对的格：三臂在同一种子上同进同退。

## 3. 轴五（同错时是不是同一个错）

16 组三臂全错的行里，两两 `output_hex` 相等的组数为 **0**（`axis5_identical_outputs_within_all_wrong={}`），三臂全同亦为 0。⇒ 三枚终端可以在同一题上同时错，但**错成三个不同的串**。这与 EXP-N1-90 §2 那句"79/96 逐题字节全同"不矛盾：那里的相同集中在共同答对区，本件说明共同答错区完全不重合。合起来读：读出臂的差异体现在"错的时候错什么"，而不是"能不能错在同一处"。

## 4. 本件不许声称什么，以及下一格要 owner 裁什么

§0 的四条禁令在本件全部生效：`selection_side=96/96` 是**选择侧的证据**，它不与"位置读出机制无用"等价——本件没有动过任何一条读通路，也没有做过候选注入的对照；把它读成机制结论就是无出处的因果。同样，`transition=0 错` 不许写成"迁移类能力已成立"，因为 8 题 × 9 终端全对在这三批配方下只说明这 32 题里的 transition 那一格没有区分度，动态范围不足（A 支线旧教训：预注册前要问"这把尺有没有动态范围"，本件正是那一条的镜像应用）。

下一格按 §3⑴ 的预注册指向**读通路侧**：要测的变量是"哪条通路为答案头提供了候选"，可做的最小题是把候选符号直接给定（oracle 注入）再看三臂是否分形——但 v5 面已消费且 `retry_allowed=false`，任何这类诊断若再用那 32 题就只能标"已知面上的机制探针"，不得再冠"未见"。要做成"未见"必须走 owner 定义分布的新题集（EXP-N1-90 §6 乙路）。本件不自选，只把两条可行动作连同各自的限制报给 owner。

## 5. 出口与前置

本件只有一格产出：上述报告与这份判读件。发表资格前置不变（保持集、C4 净增益、三器官各自行为证据三样都还没做）；本件不结 N1，不开 N2/N3/N5，也不撤销任何已登记的训练充分结论。相关更正与同族读数：EXP-N1-90（臂差判否）、DEBT-G147（口径混用）、DEBT-G148（状态钉死断言）、DEBT-G149（black 面与 PINNED 定值 4 枚＝执法型 2／披露型 2）。
