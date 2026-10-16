# EXP-N1-90 v5 未见确认判读：位置读出臂相对 cut 对照的收益被判否（2026-10-17）

判读对象＝`reports/taiji_n1_transition_factorial_confirmation_run89.json`（九枚同协议终端 × 既有 v5 32 题，288 行、`confirmation_queries=288`、`parameter_updates=optimizer_steps=signature_clone_updates=0`、CPU 34.34375 秒、`can_promote=false`）。判据一个数字未改，冻在 [PLAN-N1-90](PLAN-N1-90_three_family_binding_v5_confirmation_20261017.md) §3 并照抄 [PLAN-N1-85](PLAN-N1-85_transition_factorial_frozen_confirmation_20261016.md) §原14判据。本件是**否定式结论**，写法上把"被否证的是哪一句"点死。

## 1. 四门逐条出值

| 门 | 值 | 依据（现取） |
| --- | --- | --- |
| `each_position_seed`（每枚 ≥16/32） | 真 | position 三枚 31、32、16 |
| `each_position_kind`（每类 ≥2/8） | **假** | seed13 position 的 `lookup` 0/8、`revision` 0/8（seed11 仅 `unknown` 7/8，seed12 四类各 8/8） |
| `advantage_terminal`（≥ 对照＋12/96） | 真 | position 79 对 terminal 33，差 ＋46 |
| `advantage_cut`（≥ 对照＋12/96） | **假** | position 79 对 cut **80**，差 −1 |

`behavioral_pass=false`、`passed=false`。合取不成立的两条都指向同一件事：**在未见题集上，位置读出臂相对 cut 对照没有任何收益**。

## 2. 双臂在未见面上几乎不可区分（零查询，只读已封存 288 行）

逐字（`raw_hex`）对表 96 个 (seed,item) 三元组：position 与 cut 在 **79/96 上输出完全相同**；terminal 与二者各只相同 33/96。分类看：`transition` 24/24 全同、`unknown` 23/24、`lookup` 16/24、`revision` 16/24；17 处不同里 **16 处发生在 seed13**、按类集中在 lookup(8)＋revision(8)。

更关键的是结果交叉表：position 答对的 79 题**全部**也被 cut 答对（`both_right_and_equal=79`），"只有 position 对"= **0**，"只有 cut 对"= 1。⇒ 位置读出臂在未见证据上买不到任何东西：它没有一次做出 cut 做不到的事。教师端同向（median `teacher_ce`：position 9.6e-05、cut 3.3e-04、terminal 1.425035；三臂 `stop_reason` 除 terminal 一枚 `cap` 外全为 `boundary`）。

## 3. 失败画像不在"臂"，在 seed 与类

position 与 cut 的失分同处 `lookup`/`revision`（seed13 两臂都 0/8），而 `transition` 与 `unknown` 两臂都拿满 ⇒ v5 的未见难度是**按 seed 与按类**分布的，与读取臂无关。terminal 臂的失分则相反：`transition` 8/8、`lookup`/`revision` 0/8（三枚皆然）⇒ "末梢读出"能答迁移类但答不了查表/修订类。这是关联读数，不指认成因；本件不把它写成"绑定通道已定位"。

## 4. 工程完整性那一格是假阴性，但不改变结论，也不可补救

`status=complete`、288 行、9 次完整加载、`code_unchanged=true`、CPU 34.34375≤180、未见越界查询 0，但 `complete=false`，唯一失败格＝`sources_unchanged=false`。根因已定位到取法：90 绑定沿用了三族回执里按"JSON CRLF 归一"算的摘要，而 85 收尾用**原始字节** sha 复核 ⇒ 现读 149 枚守卫中 17 枚 `.json` 原始值不等、**逐枚按 CRLF 归一后全部相等**（真实改动 0 枚）。这是我自己新仪器的口径缺陷，入册 DEBT-G147；不改写已封存的报告字节，也不据此声称工程完整。**面已消费**（`confirmation_face_consumed=true`、`retry_allowed=false`）⇒ 无论哪一格，都不存在"修好仪器再跑一次"的选项。

## 5. 与训练面读数的关系（这正是 C3 仪器要抓的东西）

同一批九枚检查点在已知训练面上是：position 三枚 192/192/191 全过六门（㊵-1012）、cut 两枚过／seed11 `joint` 39 不过（㊵-1019）。到未见面，position 掉到 79/96 且与 cut 不可区分 ⇒ 六门是训练拟合度量的事实再次被证实，且第一次出现"训练面全绿、未见面判否"的**同批检查点对照**。因此：不得用训练面 192/96 支撑任何能力主张；本件支持的结论只有"该配方下两臂未见表现无差别"。

## 6. 收回的句子与后继出口（不自选）

- 收回：EXP-N1-88 §6 甲路里"九枚同协议可让 `领先 ≥12/96` 不混杂地声称"这一**期望**——不混杂是真的，声称不成立也是真的；以及任何"位置读出是同一实体迁移的杠杆"表述（本轮被未见面否证，非被训练面支持）。
- 保留：run88c/89a/89b 的批次级工程读数与"衰减修末周期"这一配方事实（它在两臂上都复现了：89b seed13 cut 由 180/joint24 → 191/joint47）。
- 出口需 owner 裁（都改变范围或要新配额）：⑴**归因换向**——问题不在读取臂，v5 已消费但 288 行原始输出已封存，可零查询做逐题错因分类（尤其 seed13 的 lookup/revision 全灭），据此决定 N1 下一件；⑵**转 C4** 保持集/净增益，但 C4 若仍以 position 为治疗臂，其前提已被本件削弱；⑶**另立未见题集**（需新分布定义＋owner，且不能再引 v5）。本件不自选，也不补训、不换题、不改线值。

## 7. 发表资格前置

未见能力类主张今天为**负**：v5 上无任何臂差。此后任何正向主张需要：另立的未见集 ∧ 保持集 ∧ C4 净增益 ∧ 三器官各自行为证据，且 `can_promote=false`、不替换默认模型。
