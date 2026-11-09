# 放权决策台账（DECISION-LEDGER）

> 用途：记录**由执行者自行决定**的越界决策，一笔一条，可逐条复核。授权来源统一为：**用户 2026-11-09 授权（放权，须留档）**。
> 覆盖范围：①预算超限；②动封存集或改写已冻结判据；③开新训练窗口；④方向二选一（按“上限更高的方案”取舍）。其余纯实现修复、同口径复测、文档同步不在此立账。
> 纪律：留档在决策当时写，不事后补；每笔独立提交（不推送）；索引行追加在里程碑简报末尾。
> 关联：判据与读数见 [SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md](SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md) §20–§21、[INVARIANT-LEDGER_20261103.md](INVARIANT-LEDGER_20261103.md) §47–§49、[CAPABILITY-CARD_20261103.md](CAPABILITY-CARD_20261103.md) U35–U37。

## 索引

| 编号 | 类型 | 一句话 | 提交 |
|---|---|---|---|
| D-20261109-01 | 判据改写（冻结记录事后修正，判据本身未变） | 仅格式重跑后改写 manifest 的 `runner_sha256`／`freeze_note`，判据与语料未动、读数逐项相同 | 7c8eb0646 |
| D-20261109-02 | 方向取舍（四选一） | 队首定为“表示层”，放弃“训练合流／更短地址／外部语料” | 2d88d0ddc |
| D-20261109-03 | 判据改写（读数前口径修正） | 无关对由文字“120”更正为完整枚举 1,920（120 个无关 goal 对 × 16 句对），首份未入库 manifest 删除重冻 | 59994d141 |
| D-20261109-04 | 判据改写（读数后夹具字面修正，重跑读数逐项相同） | 夹具 `native_goal_world_wide.states` 的 `zip` 补 `strict=True` 以满足仓库 B905 闸；夹具 sha 变更、重冻重跑，全部读数相同 | 59994d141 |
| D-20261109-05 | 外部依赖（R3 由“仅缓存”放宽为“允许下载”） | 为回答“夹具是否存在可分上界”，允许下载冻结句向量权重并记录 revision／config digest，R3 只作上界、不进总判定 | c23b1d9ca |
| D-20261109-06 | 读数后 runner 增补（R4a 输出逐对明细） | R4a 首跑缺逐对距离，无法做家族拆分；给 runner 加 `pairs` 输出后重冻重跑，判据/阈值/夹具未动，八项读数逐项相同 | 90426725e |
| D-20261109-07 | 预算超限＋读数后 runner 修正（CPU 线程与 wall 计时） | R5 首跑 845.0 CPU、次跑 702.0 CPU（上限 200）；归因为 BLAS 线程自旋＋wall 计时公式写错，限 1 线程后重冻重跑 0.516 CPU，判定与全部读数逐项相同 | 0061918d8 |
| D-20261109-08 | 判据改写（读数后发现 joint 列结构性无效，冻结新口径重跑） | R6 的 joint 用 20 类 one-hot，而按 goal 分组交叉验证时测试折的联合类在训练中无样本，20 个权重恒零、argmax 永不选中，joint 必然 0.0；改为“同一行 object ∧ place 都对”后 R6b 三臂 joint 1.0、判 `cross_family_readout_transfers` | 98263aba2 |
| D-20261109-09 | 读数后 runner 修正（名字域串用，分子错位） | R8 首跑三臂短语 object 全是 0.0、且预测里出现“左盒／窗台”等地点名：`detail()` 用 `slot` 同时决定名字表与目标字段，短语级传 `slot="target"` 导致名字表取成 `place_names`；改为显式传 `names` 后重冻重跑，判据／分母／夹具未动，verdict 仍 `control_failed` | e7b6b53df |
| D-20261109-10 | 判据缺陷（读数后发现汇总字段由控制臂构造性满足；冻结口径不改、字段改为不采信） | R9 的 `project_line` 定义为“任一臂两槽均定位成功”，而 R1 同时是**度量正控**（短语是句子子串，构造性满分）⇒ 该字段恒真；不动 runner／判据／读数，只在文档与能力卡中明确“项目侧只读 R2” | e16dfb52a |
| D-20261109-11 | 判定解释（读数后确认 0.00 为构造性零，不采信为“泛化失败”证据） | R10 留一后的 0.00 源于 1:1 词表删掉了该概念**唯一**训练目标，任何方法都必为 0；冻结口径与读数不动，只把判定含义收紧为“1:1 词表的桥只能靠写入”，并要求 R10b 多说法留一闸才可谈泛化 | f856e6acd |
| D-20261109-12 | 判定解释＋机制归因（读数后落中区，两个方向都不采信） | R10b 判 `phrase_bridge_inconclusive`：λ=1 地点 `sentence_after` 0.40 在 0.60–0.80 中区，网格最大 0.75 使 `memorised` 不成立；事后归因 λ=1 对 10 行词表**过正则**（预测被压向零、余弦退化成噪声），故小 λ 才有 0.70–0.75。不改判据、不重跑，只写明“背过／能泛化”两向禁引、信号只在地点槽、写入式桥仍是唯一可靠路径；并登记报告字段 `sentence_after_worst_lambda` 实为网格**最小**值 | 94cc2a15d |
| D-20261109-13 | 读数前口径修正（回滚证据＋未知短语筛选） | R11 首跑用含计数器的 `digest()` 当回滚证据（拒绝批次必然改计数器，证据自相矛盾）→ 改 `content_digest()`；未知短语手工挑选共享 token → 改“token 不相交”自动筛选；判据、语料、阈值未动 | 0c8cad59e |
| D-20261109-14 | 架构判断＋键定义扩展 | R12 并列 2 字节／码位／码位＋字符位三臂：字节碎片 12/18 被当地址 ⇒ 键单位确实选错；但码位版仍有 4 条单字共享碰撞（比 2 字节多 1），零线未达；offset 只治位置漂移，残留“同字同位”。判 `bridge_key_unit_still_leaks` | 31ef1850c |
| D-20261109-15 | 架构判断＋键身份改写 | R13 两段分开判：A 段键＝整串字符保序保重数、读＝连续包含才作答 ⇒ 未写候选 0／碎片 0／整句 0／能力 40/40；代价是 9 条同义改写全 `no_match`。B 段 9 条改写作附加键写入 ⇒ 9/9 读回、写后零线仍 0、键 18／实体 9。判 A `sequence_identity_clean`／B `alias_write_clean` | ff79e7094 |
| D-20261109-16 | 架构判断＋写冲突策略 | R14 三段分开判：C 键／实体 9/9＝1.0→18/9＝2.0→78/9＝8.667 被拒（60 句含实体名，“80 键”不可达；比值随说法数增长）；D 18 键表零线 0／0／0、能力 40/40；E 同一说法指向两实体 3 例全部 `rejected_batch_address_already_bound`、先写者保留、整批回滚，`replace` 对照会转主。判 C `cost_curve_measured`／D `large_table_zero_clean`／E `conflict_rejected_first_writer_kept` | edcf623ca（提交说明误写为“R14 test”，内容见下文本笔） |
| D-20261109-17 | 架构判断＋机制收敛（`replace` 路径删除／两段读新增） | R15 两段分开判：A `replace` 路径**不可达**（构造签名无参数、传入抛 `TypeError`、AST 扫描 5 个标识符 0 处、重绑批拒绝且摘要不变）＝`replace_path_absent`；B 20 句×2 槽 **40/40**，答出 2 个 20／只答 1 个 0／答错 0／弃答 0，遮蔽按首次出现同宽替换（5 字×10、6 字×10），三出口 `answered_single`／`answered_single_duplicate_key`／`ambiguous_same_start`；单遍直读基线 0/20＝`two_pass_clean`；零线 0／0＝`zeros_clean` | c1fdeaa7d |
| D-20261109-18 | 判据冻结（读数前新增两几何情形）＋记账要求（复跑锚） | R16 专测“说法位置重叠”与“一句三实体”：重叠 4 例 **4/4** 与冻结判据逐项相同（相交即在该层停并吞后一条、仅相邻不吞）、三实体不相交 **3/3** 停第 4 遍由 `no_match` 退出、跨接说法 **2/3** 停第 3 遍、同起点弃答、同址三现计 1 实体；R15 回归 40/40；R11–R15 逐个登记“最后可跑提交＋runner blob sha1＋报告 sha256”，措辞统一为“在当前 HEAD 不可跑，按提交 X 可复跑” | 93e292840 |
| D-20261109-19 | 判据冻结（读数前新增四个压缩算例）＋架构判断（键级压缩不进入默认路径） | R17 测“同值包含压缩”：值级去重 3 键→1 值但**键减少 0**；安全嵌套 2→1 并救回长说法；残差守卫挡下危险删除（无守卫会多答 1 个值）；边界泄漏证明守卫放行后会多答 1 个值；跨值包含禁止；默认 18 键表删除 0、摘要不变、40/40、零线 0/0、九项全等。总判 `compression_value_level_only` | f6b627fd8 |
| D-20261109-20 | 判据冻结（读数前新增纠正算例）＋机制新增（CAS 纠正与审计日志） | R18 测“显式纠正协议”：普通冲突写仍 `rejected_batch_address_already_bound`；`correct` 用当前值作凭据、一步改值并写日志；过期凭据/未写地址/批内撞键全部整批拒绝；纠正链 2 步有序；18 键表往返后摘要复原、40/40、零线 0/0、九项全等。总判 `explicit_correction_available`；新增 R16/R17 复跑锚 | 64a221f37 |
| D-20261109-21 | 判据冻结（读数前新增闭环算例）＋架构判断（单键闭环可用，核对器不可校验） | R19 测“读—核验—纠正”：一步纠正、读值作凭据、已一致不写、过期读被拒、普通冲突写仍拒、两轮收敛；核对器给错时协议照样接受（边界）；18 键表往返后摘要复原、40/40、零线 0/0、九项全等。总判 `read_verify_correct_loop_available` | dcda39ca9 |
| D-20261109-22 | 判据冻结（读数前新增双实体算例）＋架构判断（批量原子性可作能力） | R20 测“句级双实体纠正”：单槽错只动一槽；双槽同批 `corrected=2`；过期子集整批拒绝、有效项也不落盘；已一致不写；普通冲突写仍拒；两轮收敛；18 键表双键往返后摘要复原、40/40、零线 0/0、九项全等。总判 `sentence_pair_correction_available` | 56c87bbd0 |
| D-20261109-23 | 判据冻结（读数前新增来源算例）＋机制新增（`authority`／`reason` 与可选允许集合） | R21 测“纠正来源与授权”：空策略向后兼容；缺来源 `rejected_batch_missing_authority`、未知来源 `rejected_batch_unknown_authority`、允许来源可纠正并记账；来源不替代 CAS；普通冲突写仍拒；两步链可归因；默认表往返不变。总判 `correction_authority_attributed`；新增 R18/R19/R20 复跑锚 | 180ab0f1e |

---

## D-20261109-01 冻结记录事后修正（判据未变）

**1. 编号／时间／类型**：D-20261109-01｜2026-11-09（与决策同一会话，记录时间晚于决策约数分钟）｜判据改写类——实际是“冻结 manifest 的记账字段事后改写”，**判据与语料均未变**。

**2. 动了什么（逐字）**：文件 `plans/manifests/taiji_contig_20261109.json`，两处。

- `runner_sha256`
  - 原值：`3aca50ce…`（**完整摘要已不可得**：该字节版本从未入库，其报告已被覆盖；仅保留前缀）
  - 新值：`b804a5d460fb71ebb0187c89c872a661f120e0d4352f2e6d5bb9609b9f118da3`
- `freeze_note`
  - 原值（逐字）：`harness-only revision: fixture classification and the unknown-gate definition changed; address layer, corpus, ranking and thresholds unchanged`
  - 新值（逐字）：`harness-only revision: fixture classification and the unknown-gate definition changed; address layer, corpus, ranking and thresholds unchanged. Runner reformatted in-process (black Mode line_length=100, py312; formatting only) and readings re-taken under the same criteria.`
- 同时**新增**增量块 `post_reading_corrections`（不覆盖原判据，只并列保存原值与新值、原因、比对读数、回退路径）。

**3. 触发原因**：`cf030848a` 首次提交时 pre-commit 的 in-process black 闸判“would reformat”，因为本轮新增的 harness 代码未按 `line_length=100`／py312 排版。格式化**只改空白与换行**，但会改动文件摘要，导致冻结的 `runner_sha256` 不再描述将入库的文件——即“报告的双列 runner sha 与仓库字节不一致”。为保住“报告由恰是该 runner 产出”的可复核性，格式化后按同一判据整跑复测。读数出处：[taiji_contig_20261109.json](../../reports/taiji_contig_20261109.json) 的 `gates`／`write`／`length_buckets`／`fallback`。

**4. 代价**：

- 失去的保证：“冻结后 manifest 逐字节不动”的强保证被打破一次（两个记账字段被改写）。
- 变化的分母／分类：**无**。未知集分母仍 550，`unknown_answerable` 仍 1，回退分母仍 9，`answerable_total` 仍 1,101。
- 旧结论可比性：**全部可比**——重跑读数逐项相同：入表 542/550、`<64 B` 121/129＝0.937984、旧口径 1/550、新口径 0/550、回退 9 条、独立复核违反 0、判定 `continuity_restored`。
- 预算：复测 **46.235 CPU／7.131 wall**（上限 200），**未超限**；该次重跑属“作废重跑”性质，**不计入有效预算**，单列于此。

**5. 替代方案（及否掉理由）**：

- 用 `--no-verify` 直接提交未格式化 runner——**否决**：违反“不得跳过 hook”。
- `--freeze` 升 `freeze_revision 5`——**否决**：判据与语料未变，升版会制造“判据第二次变更”的假象，且会丢掉 `criteria_changed_revision=4` 的记录。
- 不提交 runner、只提交报告与文档——**否决**：报告将引用仓库中不存在的 sha，证据链断裂。

**6. 撤回路径**：

- 回到上一轮（rev 3、runner `3089e1fb…`）：`git revert cf030848a`。
- 回到 rev 4 格式化前字节：**不可行**（该版本从未入库）；可恢复的是读数与判据（本就不变），manifest 原值已逐字保存在 `post_reading_corrections.fields[].before`。
- 若要撤销本笔的台账登记：删除 `plans/manifests/taiji_contig_20261109.json` 的 `post_reading_corrections` 键并 revert 本笔提交。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）。

**8. 对已有结论的影响**：U35 与 §20、§47、能力卡的**所有旧读数仍然可比**（判据、语料、地址层、排序、阈值均未变，读数逐项相同）；**没有读数作废**。唯一降级的是“manifest 冻结后不可变”这一形式性保证，已在本笔与 manifest 的 `post_reading_corrections` 中并列留痕。

---

## D-20261109-02 方向取舍：队首＝表示层

**1. 编号／时间／类型**：D-20261109-02｜2026-11-09｜方向取舍（四选一：表示层／训练合流／更短地址／外部语料）。

**2. 动了什么（逐字）**：`plans` 四处队首段（`MASTER-PLAN.md`、`MAINLINE.md`、`active/roadmap/03_CURRENT_EXECUTION.md`、`reference/ROADMAP-NEXT_20261104.md`）＋能力卡 U35／§20／§47 的“下一条动作”。

- 原（逐字）：`下一条唯一动作＝把未知集门改成 \`anchored ∧ not contained\`（仅 harness，不改地址层／语料／排序）后重跑同一读数`
- 新（逐字）：`下一唯一动作＝表示层：同一 550 条、同一冻结口径、0 训练、0 新参数，测“同一内容的不同说法”在现有上下文表示上的距离能否与“无关内容”分开`

**3. 触发原因**：记账修正闸判 `continuity_restored`，字节级锚定线已无剩余可测问题（旧 1/550、新 0/550、独立复核 1,092 行 0 违反）。四方向由 §20.6 与 U35 列出，需择一作为唯一队首。读数出处：[taiji_contig_20261109.json](../../reports/taiji_contig_20261109.json)。

**4. 代价**：

- 放弃的短期收益：不能立刻把地址层“接进训练图”，也没有一个可展示的新能力读数。
- 风险：若表示层测不出“同内容异说法”的距离信号，则地址层只能停在工程化，能力上限不再抬升。
- 分母／分类：**不变**（本决策不跑任何读数）。
- 旧结论可比性：全部可比。

**5. 替代方案（及否掉理由）**：

- 训练合流——**否决**：在“同内容异说法”尚未被证明可分的表征上接梯度，只会把噪声写进参数，且需要开训练窗（更重、更难回退）。
- 更短地址——**否决**：与已冻结的“最短可入表 16 字节”结论正面冲突，且会降低证据下限（每少 1 字节，伪造成本下降一档）。
- 外部语料——**否决**：只扩规模，不改能力边界，属能力验证后的后置项。

**6. 撤回路径**：把上述四处队首段与三处文档的“下一动作”改回“训练合流”即可；无代码／常量改动。若表示层已开跑，删除其预注册 manifest、报告与 scene 输出文件，并把队首改回。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）。

**8. 对已有结论的影响**：**无读数作废**；地址层 U35／§20／§47 结论与全部读数仍可比。本笔只改“下一步做什么”，不改“已经测到什么”。

---

## D-20261109-03 表示层预注册：无关对的口径修正（读数前）

**1. 编号／时间／类型**：D-20261109-03｜2026-11-09｜判据改写类——**首个读数之前**的口径修正（分母枚举粒度），没有任何读数被覆盖或作废。

**2. 动了什么（逐字）**：文件 `plans/manifests/taiji_representation_20261109.json`，第一次 `--freeze` 写出的版本（**从未入库**，已删除）与第二次 `--freeze` 的版本对照。

- `definition.unrelated_pair`
  - 原值（逐字）：`two goals with a different object AND a different place: 120. A cross-goal pair that shares exactly one entity is a near pair (70); near pairs are printed and excluded from the verdict.`
  - 新值（逐字）：`two sentences whose goals differ in object AND place: 120 unrelated goal pairs x 16 sentence pairs = 1,920. Enumerated completely, not subsampled: picking one rendering per goal would be a template choice made after seeing the fixture. A cross-goal pair that shares exactly one entity is a near pair (70 goal pairs, 1,120 sentence pairs); near pairs are printed and excluded from the distributions and the verdict.`
  - 同时新增 `definition.imbalance`：`same 120 vs unrelated 1,920. The AUC and the per-distribution quantiles are defined for unequal counts, and no verdict threshold is applied to the counts themselves.`
- `criteria.C2_counts`
  - 原值（逐字）：`80 sentences, 120 same pairs, 120 unrelated pairs, 70 near pairs; every pair is flagged same, unrelated or near before any distance is computed.`
  - 新值（逐字）：`80 sentences, 120 same pairs, 1,920 unrelated pairs, 1,120 near pairs; every pair is flagged same, unrelated or near before any distance is computed.`
- `counts.unrelated_pairs`／`counts.near_pairs_excluded_from_verdict`：两次冻结均为 **1,920／1,120**，**未变**。
- runner 文档串同步（`120 unrelated` → `1,920`，并写明“完整枚举、不抽样”）。

**3. 触发原因**：第一次 `--freeze` 回显 `unrelated_pairs=1920`、`near_pairs=1120`，与同一次冻结写入的文字“120／70”自相矛盾。矛盾在**计算任何距离之前**发现（当时只有 freeze 回显，无任何读数），故删除未入库的 manifest、修正文字后重冻。读数出处：`plans/manifests/taiji_representation_20261109.json` 的 `counts`。

**4. 代价**：

- 失去的保证：manifest“一次写成不再删除”的形式保证被打破一次（首版未入库，无外部引用）。
- 变化的分母／分类：无关对由设计文字的 120 变为完整枚举 1,920；near 由 70 变为 1,120。same 仍 120。
- 旧结论可比性：本笔发生在首个读数之前，**无可比的旧读数**；地址层线的全部读数不受影响。
- 预算：两次 freeze 各 <0.2 CPU，正式读数 1.5 CPU，未超限。

**5. 替代方案（及否掉理由）**：

- 按文字把无关对压到 120（每个 goal 只取一句渲染）：**否决**——选哪一句必须看夹具后决定（模板偏置），且会让 same／unrelated 的模板分布不对称，是典型的“先看数据再定口径”。
- 保留 1,920 但让文字保持含糊：**否决**——预注册内部自相矛盾，事后无法复核。

**6. 撤回路径**：删除 `plans/manifests/taiji_representation_20261109.json` 与 `reports/taiji_representation_20261109*.json*`，把 runner 的 `definition.unrelated_pair`／`criteria.C2_counts`／文档串改回 `120／120／70`，重新 `--freeze` 后重跑；无代码路径、无地址层、无语料改动。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）。

**8. 对已有结论的影响**：**无读数作废**（本笔早于首个读数）。本轮报告与 manifest 在 1,920／1,120 上一致；地址层 U35／§20／§47 与全部旧读数仍可比。

---

## D-20261109-04 夹具字面修正后重冻重跑（读数逐项相同）

**1. 编号／时间／类型**：D-20261109-04｜2026-11-09｜判据改写类——**读数之后**对夹具文件做 lint 字面修正（不加 `strict=True` 无法通过仓库的 B905 闸），因此按“不覆盖原判据、并列保留新旧读数”的方式重冻重跑。

**2. 动了什么（逐字）**：文件 `taiji/native_goal_world_wide.py`，函数 `states()`。

- 原值（逐字）：`tuple(zip(OBJECTS, placement)) for placement in itertools.permutations(PLACES, len(OBJECTS))`
- 新值（逐字）：
  `tuple(zip(OBJECTS, placement, strict=True))`
  `for placement in itertools.permutations(PLACES, len(OBJECTS))`
- 该文件此前**从未入库**（本轮首次提交），`B905` 是 pre-commit 的唯一阻塞项。`strict=True` 只断言两个等长可迭代的长度一致，**不改变任何数据**（`OBJECTS` 恒为 4 项，`placement` 恒为 4 项）。
- 连带：`plans/manifests/taiji_representation_20261109.json` 重写（`fixture_sha256` 中该文件 `3ebe92fb36f4…` → `01ef1914a576…`，另两个夹具文件哈希未变），报告重跑，`fixture_sha256` 同步。

**3. 触发原因**：首次提交被 pre-commit 的 ruff `B905` 拦下（“zip() without an explicit strict= parameter”），且纪律禁止 `--no-verify`。夹具文件字节一变，manifest 的 `fixture_sha256` 就不再描述将入库的文件，故必须重冻重跑。读数出处：原报告（staged，未入库）与新报告 `reports/taiji_representation_20261109.json`。

**4. 代价**：

- 失去的保证：夹具字节在冻结后被改过一次（虽为 lint 字面、无行为变化）。
- 变化的分母／分类：**无**。80 句／same 120／unrelated 1,920／near 1,120 全部不变。
- 旧结论可比性：**全部可比**。逐项比对 `readings`／`verdict`／`decision_arms`／`counts`／`sentences`／`thresholds`／`definition`／`criteria` 八个子树，**完全相同**；runner 双列 sha 仍为 `b04ef3de…`（runner 未改）。
- 预算：重跑 **1.922 CPU／0.605 wall**（上限 200），未超限；该次属“作废重跑”，**不计入有效预算**（有效预算为首次 1.5 CPU），单列于此。

**5. 替代方案（及否掉理由）**：

- `# noqa: B905` 挂在该行——**否决**：同样改夹具字节、同样要重冻重跑，却把 lint 抑制写进数据文件，且遮蔽了真实可加的断言。
- `git commit --no-verify`——**否决**：纪律明令禁止跳过 hook。
- 不提交夹具、只提交 runner 与报告——**否决**：manifest 引用的夹具文件不在仓库里，证据链断裂。

**6. 撤回路径**：把 `taiji/native_goal_world_wide.py` 的 `strict=True` 去掉、恢复 `zip(OBJECTS, placement)`，删除 manifest 与报告后重新 `--freeze` 重跑；或 `git revert` 本笔提交（会同时撤回表示层闸的其余文档与读数）。无地址层、无语料、无判据改动。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）。

**8. 对已有结论的影响**：**无读数作废**——新旧读数逐项相同，判定仍为 `representation_not_supported`。唯一变化是夹具文件 sha；地址层线（U35／§20／§47）与全部旧读数不受影响。

---

## D-20261109-05 R3 由“仅缓存”放宽为“允许下载”

**1. 编号／时间／类型**：D-20261109-05｜2026-11-09｜外部依赖类——≠预算超限（CPU 16.219 远低于上限 200），≠改判据（R1／R2／B 判定与阈值一字未动），≠开训练窗（0 训练／0 优化器步／0 新参数）。

**2. 动了什么（逐字）**：`scripts/training/probe_taiji_representation_r3_20261109.py` 的下载策略常量与文档串。

- 原口径（逐字，来自 [SPEC §21.3](SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)）：`只在本机缓存已存在时使用，否则 unavailable，不下载、不改判据`
- 实际执行（逐字，runner 常量）：`IDLE_LIMIT = 900.0`（模型加载/下载期间允许的空闲上限，秒）；加载路径不再短路为 `unavailable`。
- 实际后果：Hugging Face 缓存目录 `models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2` 于 **2026-11-09T10:35:02** 落盘；权重解析为 `revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42`、`config_digest 584d721354a4443103960a80b1a3a61226765bc470bd4077d96e37beb85b433f`，均写入 [R3 报告](../../reports/taiji_representation_20261109_r3.json) 的 `model.payload`。

**3. 触发原因**：R3 是唯一能回答“这个夹具是否存在可分上界”的臂；而 §21.9 已写死“不能回答……R3 本机无缓存、本轮未下载，留待下一臂”。若继续坚持“仅缓存”，该问题将永久悬空，且后续任何“是否值得给桥／是否值得开训练窗”的判断都失去参照。读数出处：[R3 报告](../../reports/taiji_representation_20261109_r3.json) 的 `reading`／`model.status=available`；[进度日志](../../reports/taiji_representation_20261109_r3_progress.jsonl) 的 `loading_model → model_loaded`。

**4. 代价**：

- 外部依赖：首次引入约 470 MB 权重下载（网络依赖 + 磁盘占用），此前整条线零外部权重。
- 预算：**16.219 CPU／257.23 wall**（上限 200 CPU）；wall 的绝大部分是模型加载与下载等待，不计入 CPU。
- 可复现性：已记录 `revision` 与 `config_digest`，同一 revision 可逐位复现；但**离线环境无法重跑** R3（这是本条唯一的长期代价）。
- 分母／分类：**不变**（80 句／120／1,920／1,120）。
- 旧结论可比性：**全部可比**，无读数作废；R3 只作上界，未进总判定。

**5. 替代方案（及否掉理由）**：

- 坚持“仅缓存 ⇒ `unavailable`”——**否决**：等于主动放弃本轮唯一的架构判别读数，并让 §21.9 的“不能回答”成为永久缺口。
- 改用“标注数据上训练出的判别上界”——**否决**：那会引入可训练自由度与训练窗（需另行批准），且不再是“冻结通用语义空间”的对照，读数含义会被污染。
- 换一个更小的本地模型——**否决**：MiniLM-L12 已是本机可得的标准化多语种句向量；换模型等于改“上界”的定义，且无证据更省。

**6. 撤回路径**：删除 HF 缓存目录 `models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`、删除 `reports/taiji_representation_20261109_r3*.json` 与 `scripts/training/probe_taiji_representation_r3_20261109.py`，`git revert` 本笔提交；把 SPEC §21.3／§21.9／§21.11 与四份 plans 队首改回“R3 `unavailable`、上界未测”。不影响 R1／R2／B 的任何读数。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）。

**8. 对已有结论的影响**：**无读数作废**。总判定仍为 `representation_not_supported`（R1／R2／B 三臂全 `overlaps`，阈值与分母未动）。新增的是“上界可测且判 `overlaps`”，并由此把判负范围精确化为“现有表示 × 跨词法家族”（§49 S-49／S-50）。

---

## D-20261109-06 R4a 读数后给 runner 增补逐对明细

**1. 编号／时间／类型**：D-20261109-06｜2026-11-09｜harness 增补（输出明细），不是判据变更。

**2. 动了什么（逐字）**：`scripts/training/probe_taiji_representation_r4a_20261109.py` 的 `arm_reading()` 增加：

```python
reading["pairs"] = {
    "same": [...],
    "unrelated": [...],
}
```

**判据、阈值、夹具、配对、判决规则全部未改**；manifest 因 runner sha 变化重冻一次，`counts` 与 `checks` 逐项相同（80／20／120／1,920／1,120；describe 改写 20、paraphrase 改动 0）。

**3. 触发原因与读数出处**：R4a 首跑判 `bridge_insufficient`，但 1-NN 只有 0.05／0.0125，无法从汇总读数判断“是桥没生效”还是“表示被模板主导”。逐对明细是回答该问题的唯一材料。出处：[R4a 报告](../../reports/taiji_representation_r4a_20261109.json) `arms.*.nearest_neighbour.rows`；[R4a 拆分](../../reports/taiji_representation_r4a_20261109_split.json)。

**4. 代价**：报告体积增大（含 3,040 条配对距离）；重跑 4.156 CPU／0.47 wall，**未超上限 200**。

**5. 替代方案（及否掉理由）**：
- 另写脚本重算矩阵（**否决**：会把同一计算路径复制两份，且无法证明与判定同源）。
- 不做拆分、只报汇总（**否决**：汇总读数对 `bridge_insufficient` 的机制解释为零，等于把“为什么”留给猜测）。

**6. 撤回路径**：删掉 `reading["pairs"]` 段、重冻重跑（读数不变），或 `git revert` 本笔。

**7. 授权来源**：用户 2026-11-09 的放权条款（“要改已冻结判据……自行决定并执行，但必须留档”）；本笔严格说未改判据，仍按同规格留档。

**8. 对已有结论的影响**：**无读数作废**。首跑与重跑的八项读数逐项相同（R1 gap −0.023106／AUC 0.930655／1-NN 0.05；R2 gap −0.037904／AUC 0.902808／1-NN 0.0125）。新增的是机制读数：1-NN 同模板 R1 95%／R2 98.75%。

---

## D-20261109-07 R5 首跑超预算：CPU 线程与 wall 计时修正后重冻重跑

**1. 编号／时间／类型**：D-20261109-07｜2026-11-09｜**预算超限类**（首跑 844.984 CPU、次跑 701.953 CPU，上限 200）＋读数后 runner 修正（只改资源与计时，不改判据／夹具／读数口径）。

**2. 动了什么（逐字）**：`scripts/training/probe_taiji_slot_probe_20261109.py` 三处。

- 新增（在 `import numpy` 之前，逐字）：
  ```python
  for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
      os.environ.setdefault(_variable, "1")

  import numpy as np
  import torch
  ```
  以及导入段之后的 `torch.set_num_threads(1)`。
- `phase_run()` 计时原值（逐字）：`started = time.process_time()`；`wall_seconds = round(time.time() - started, 2)`——**把 epoch 秒减成了 CPU 秒**，故首跑报 `wall_seconds = 1794197664.92`（无意义）。
- 新值（逐字）：`started_cpu = time.process_time()`、`started_wall = time.time()`、`"wall_seconds": round(time.time() - started_wall, 2)`，并在 `cost` 增 `"threads": torch.get_num_threads()`。
- **判据、阈值、`LAMBDAS`、`FOLDS`、标签、480 条样本、5 折分组、`READABLE_JOINT=0.80`／`NOT_READABLE_MARGIN=0.10` 全部未改**；manifest 因 runner sha 变化重冻一次，`counts` 逐项相同（480 样本／120 states／4 物件／5 地点／折大小 88,112,80,124,76）。

**3. 触发原因与读数出处**：首跑结束时 `cost.cpu_seconds = 844.984`，**超出 200 上限 4.2 倍**。逐段归因（诊断命令，非正式读数）：`features_for('R1')` **0.08 CPU／0.08 wall**、`features_for('R2')` **0.39 CPU／0.39 wall**，而 `arm_reading` 单臂 **381.3 CPU／32.24 wall**（R1）与 **356.7 CPU／29.45 wall**（R2）——即开销几乎全部在 80 次 128×128 的 `np.linalg.solve`：BLAS 为小矩阵也拉起全部 worker 线程自旋。第二次只加 `torch.set_num_threads(1)`，仍 **701.953 CPU／59.59 wall**（CPU/wall≈11.8，线程未被限住）；把线程限制移到 numpy 导入之前后，**0.516 CPU／0.54 wall**，降幅约 1,600 倍。出处：[R5 报告](../../reports/taiji_slot_probe_20261109.json) `cost`；首跑原样留档 [run1](../../reports/taiji_slot_probe_20261109_run1.json)。

**4. 代价**：
- 超限本身不可撤回：两次超预算运行共 **1,546.9 CPU 秒**，如实计入本轮预算，**不按“作废重跑”豁免**。
- 失去的保证：manifest 的 `runner_sha256` 在首跑后改写过两次（每次重冻）；首跑报告的双列 sha 已不描述入库 runner，故按“读数逐项比对 + 旧报告留档”方式保留证据链。
- 变化的分母／分类：**无**（判定与三组 `per_lambda` 读数逐项相同，见第 8 点）。
- 旧结论可比性：**全部可比**。

**5. 替代方案（及否掉理由）**：
- 保留首跑、只把 845 写进报告——**否决**：`wall_seconds` 字段仍是无意义值，且后续每次重跑都会再烧 700+ CPU 秒。
- 只改 wall 公式、不限线程——**否决**：治不了超预算这个真问题。
- 换 `scipy.linalg.solve` 或手写 Cholesky 绕开 BLAS——**否决**：同一 BLAS 后端仍在，且引入新依赖／新数值路径，读数可比性下降。
- 降低 `FOLDS` 或 `LAMBDAS` 把开销压下去——**否决**：那是改判据与分母，代价远大于限线程。

**6. 撤回路径**：删掉三段线程限制与 `threads` 字段、把计时改回单变量，重冻重跑（会重新超预算）；或 `git revert` 本笔。判据与读数不受影响。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档；“预算要超”属三类之一）。

**8. 对已有结论的影响**：**无读数作废**。三次运行的 `arms`、`verdict`、`counts` 逐项相同：R1／R2 在 λ=0.001／0.1／10 上 **object 1.0、place 1.0、joint 1.0**，λ=1000 时 place 降到 **0.627083**（R1）／**0.552083**（R2），总判定 **`content_present_readout_wrong`**。新增的只是资源读数（最终 **0.516 CPU／0.54 wall／threads=1**）与可用的 wall 计时。

---

## D-20261109-08 R6 的 joint 列结构性无效：冻结新口径重跑（R6b）

**1. 编号／时间／类型**：D-20261109-08｜2026-11-09｜**判据改写类**——读数之后发现预注册的 `joint` 分子本身不可测，冻结新口径并重跑；阈值一字未改。

**2. 动了什么（逐字）**：

- R6 manifest（`plans/manifests/taiji_cross_family_readout_20261109.json`）原判据（逐字）：`readable iff joint >= 0.75 and object >= 0.85 and place >= 0.80; not_readable iff joint <= chance + 0.10; otherwise inconclusive.`
- R6 的 `definition.labels`（逐字）：`object (4 classes), place (5 classes), joint (20 classes), one-hot targets.`
- 实际读数（逐字，[R6 报告](../../reports/taiji_cross_family_readout_20261109.json)）：R1／R2／R3 四档 λ 的 `joint_accuracy` **全部为 0.0**，而同一批行的 `object_accuracy` 为 **1.0**、`place_accuracy` 为 **1.0**（λ=0.001）。R6 总判定 `cross_family_readout_fails`。
- 修正（R6b，新冻结 `plans/manifests/taiji_cross_family_readout_joint_20261109.json`）：`joint_accuracy = fraction of held-out rows where the arg-max object label and the arg-max place label are both correct on that row.`
- **阈值、折划分（sha256 发牌的 5 折、按 goal 分组）、λ 网格、80 句夹具、三臂定义全部不变。**

**3. 触发原因**：R6 的 joint 是 20 类 one-hot 回归，而交叉验证**按 goal 分组**——测试折的 4 个 goal 在训练折中完全缺席，因此对应那 4 个联合类的 20 个权重列没有任何训练行（恒为 0），argmax 永远不可能选中它们。**joint=0.0 是构造决定的，不是测出来的**；以它驱动的总判定 `cross_family_readout_fails` 不可解释，不能作为对表示或读法的读数。发现时 R6 报告已落盘，故不覆盖、不删除，改为并列冻结 R6b 并在两份文档中互相指向。

**4. 代价**：

- 失去的保证：R6 的预注册判据在读数后被证明不可用，这是本轮第二次“读数后改口径”（第一次为 D-06 增补输出）。区别是：D-06 只增字段、读数不变；本笔是**判据的分子本身重定义**。
- 变化的分母／分类：`joint` 的分子由“20 类 argmax 命中”改为“同一行 object ∧ place 同时命中”；分母仍 80。object／place 两列的读数**一字未变**（R6 与 R6b 都是 1.0／1.0／0.975）。
- 旧结论可比性：R3／R4a／R5 全部可比（夹具、折、λ、阈值、表示均未动）。
- 预算：R6 **17.578 CPU／37.03 wall**（首跑，留档 `reports/taiji_cross_family_readout_20261109_run1.json`）、R6b **15.609 CPU／19.16 wall**（首跑，留档 `..._joint_20261109_run1.json`），各自低于 200。提交前两个 runner 因本地 B/SIM 闸做了**只改语法的等价修正**（`zip(strict=True)`、Yoda 条件、未用循环变量），重冻重跑后成本为 **15.609 CPU／19.18 wall**（R6）与 **15.469 CPU／18.64 wall**（R6b），`arms`／`verdict`／`counts` 与首跑逐项相同。

**5. 替代方案（及否掉理由）**：

- 保留 R6 判定、写一句“joint 无意义”——**否决**：总判定会带着一个构造性零继续在计划里流转，正是本轮要清理的“旧机制残留”。
- 把折改成按 object 或随机分组，让联合类在训练中出现——**否决**：那会泄漏“同一 goal 的四种说法分处训练与测试”，直接摧毁本探针要测的跨说法迁移。
- 用 20 类但把未见类的分数按 object+place 组合重建——**否决**：等价于本笔的逐行合取，却把逻辑藏进代码、不可复核。
- 直接改 R6 runner 重跑覆盖报告——**否决**：R6 的冻结报告必须保留，否则“读数前冻结”的纪律无从检验。

**6. 撤回路径**：删除 `plans/manifests/taiji_cross_family_readout_joint_20261109.json`、`reports/taiji_cross_family_readout_joint_20261109.json` 与 R6b runner，把 R6 的 `cross_family_readout_fails` 恢复为唯一判定即可；无夹具／语料／阈值改动。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档；“改写已冻结判据”属三类之一）。

**8. 对已有结论的影响**：R6 的 `cross_family_readout_fails` **作废**（构造性零，不是读数）。R6b 判定 **`cross_family_readout_transfers`**：R1／R2／R3 三臂在 λ=0.001 上 **object 1.0、place 1.0、joint 1.0**，λ=0.1 时 R2 **0.975**、R3 仍 **1.0**，λ=10 降到 0.125／0.0125／0.475，λ=1000 全部塌到 ≤0.0375。**限定**：object／place 在夹具中各只有一个名称串与一个描述串，同物件的其它 goal 在训练折里出现过，所以 1.0 是“固定槽位串的词汇查表 + 跨组合外推”，**不**是“理解改写语义”；R1（纯字节直方图）与 R3（句向量）同为 1.0，正说明此处不需要句向量。§21.4／§21.11／§21.12 的判负**不撤销**（那是整句余弦读法），R5 的“同族可读”**不扩大**为“跨说法语义可读”。

---

## D-20261109-09 R8 名字域串用的实现修正（判据／分母／夹具未变）

**1. 编号／时间／类型**：D-20261109-09，2026-11-09，读数后 runner 修正（不属于预算超限、封存集、判据改写、训练窗口四类；照 D-06／D-07 先例登记）。

**2. 动了什么（逐字引用）**：`probe_taiji_slot_bridge_20261109.py` 的 `detail()` 签名由

> `def detail(rows, predicted, slot) -> list[dict]` … `names = object_names if slot == "object" else place_names`

改为显式传入名字表：

> `def detail(rows, predicted, slot, names) -> list[dict]`，调用处分别为 `detail(payload["phrase_object"], phrase_object_pred, "target", object_names)`／`detail(payload["phrase_place"], phrase_place_pred, "target", place_names)`。

**3. 触发原因与读数出处**：首跑报告 `verdict=control_failed`，但三臂的 `phrase.object_accuracy` **全为 0.0**，且 `phrase.object_detail` 的 `predicted` 里出现“左盒／窗台／书架”等**地点名**——object 槽位不可能出现这些字符串。定位为：短语级调用传 `slot="target"`，而 `detail()` 用同一个 `slot` 决定名字表，于是 object 预测被拿 `place_names` 解释、并与地点名比较，24 个判对全被算成错。出处：`reports/taiji_slot_bridge_20261109_run1.json`（原样留档，未覆盖）。

**4. 代价**：重跑 **13.016 CPU／18.59 wall**（跑前按上限估 60，低于 200 未立预算账），`threads=1`。

**5. 替代方案**：①保留 0.0 当读数——**否决**，那是实现错误不是读数；②只改文档说明、不改代码——**否决**，会让“短语级 object 无桥”的错误结论进能力账；③重写整个度量——**否决**，过度改动会换掉已冻结的口径。

**6. 撤回路径**：把 `detail()` 恢复为单 `slot` 参数即可回到 run1 的读数；无夹具／语料／阈值改动，`fixture_sha256` 与 `criteria` 在两次 manifest 中逐字相同。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）。

**8. 对已有结论的影响**：修正后短语级 object 从 0.0 变为 **R1 0.5／R2 0.5／R3 1.0**，place **0.6／0.6／0.6**；整句级 **R1 0.2／0.35、R2 0.5／0.3、R3 0.85／0.6**；paraphrase 控制 **R1 0.85／0.867、R2 0.65／0.85、R3 1.0／1.0**。总判定仍 **`control_failed`**（冻结规则要求所有可用臂控制 ≥0.90；R1／R2 未过 ⇒ 最近名度量对低维臂不成立），`project_bridge=absent` 不变。**R3 自身控制 1.0／1.0 通过**，其读数可解释：object 短语 **1.0** → 整句 **0.85**（掉 0.15，未达 0.25 稀释阈），place 短语 **0.6** = 整句 **0.6**（掉 0）⇒ **地点名不存在“句内稀释”**，在短语级就已经只有 3/5；object 的桥在短语级完整、整句有轻微损失。该解释只作 R3 上界观察，**不升级能力账**（见 D-05）。

---

## D-20261109-10 R9 的 `project_line` 汇总字段由控制臂构造性满足

**1. 编号／时间／类型**：D-20261109-10｜2026-11-09（决策在读数之后、写文档之前）｜**判据缺陷**：不改 runner、不改读数、不改冻结判据，只把“该字段不采信”写进文档与能力卡。

**2. 决策内容**：R9 的冻结口径把项目侧汇总字段定义为 `project_phrase_localised iff 任一臂两槽均定位成功`。而 R1 同时承担**度量正控**，R9 的句子又把短语原样嵌在里面（`让{物件短语}到{地点短语}。`），R1 的 128 维字节直方图对子串天然高分，于是 R1 必然满分 ⇒ **只要正控通过，该字段就恒真**。处置：runner、manifest、报告一字不改（判定与全部读数都不依赖该字段），在 SPEC §21.17、账本 §55、能力卡 U43 与三处计划里写明：**该字段不采信，项目侧只读 R2**。

**3. 触发原因与读数出处**：报告里 `project_line=project_phrase_localised`，而 `arms.R1` 两槽均 **1.00**（构造性）、`arms.R2` 为 object **1.00**／place **0.70**、`slot_verdict.R2.place=inconclusive`。出处：`reports/taiji_phrase_localisation_20261109.json`。

**4. 代价**：本决策本身 **0**（不重跑、不改判据）。但同批提交前 black 行宽整改动了 runner 两处换行，按“runner sha 必须与报告一致”的纪律重冻重跑两次（内联串 **16.766 CPU／23.24 wall**、终版 **17.875 CPU／24.32 wall**），三组读数逐项相同；终版 runner sha `403021cd…`、报告 `cost` 与七处计划均已对齐该次。相对上限 200，两次合计约 35 CPU，未立预算账。

**5. 替代方案**：①重冻重跑、把 `project_line` 改成只看 R2——**否决**：读数必然逐项相同（本轮已因格式化重跑两次，三组读数逐项一致），为改一个汇总标签再动判据是白花 CPU；②把 R1 从三臂里去掉——**否决**：R1 是本轮唯一可行的度量正控，去掉就没有“度量没坏”的证据；③维持原字段照发——**否决**：它会被读成“项目表示能定位短语”，与 R2 的 0.70 直接矛盾。

**6. 撤回路径**：把该字段的读取口径改回“任一臂”，或在其旁附 `control_arm_excluded: false`；报告文件本身不需要变。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）；并按用户规则“归因优先看架构、汇报不许含糊”。

**8. 对已有结论的影响**：不影响 R9 的判定 `phrase_survives_name_link_absent`（只看 R3 地点定位 1.00），也不影响 R8 的任何读数。项目侧的可信表述收紧为：**R2 能把描述短语定位回自身（object 1.00、place 0.70）**，place 落在 0.60–0.80 之间、按冻结阈值判 inconclusive——**不得**写成“项目表示能定位地点短语”。后续轮次（R10 起）项目侧汇总只算在 R2 上。

---

## D-20261109-11 R10 的 0.00 是构造性零

**1. 编号／时间／类型**：D-20261109-11｜2026-11-09（读数之后、写文档之前）｜**判定解释类**：不改 runner、不改 manifest、不改读数、不改冻结判据；只收紧该判定**允许被引用的含义**，并规定取得“泛化”读数所必需的下一个闸。

**2. 决策内容**：R10 三臂的 `after` 列在整个 λ 网格上全为 **0.00**，判定按预注册落为 `phrase_name_bridge_memorised`。但这条 0 是**构造性**的：词表是 1:1（每个名字只有一条描述短语），留一删掉该句自身词条时，等于把这个概念**唯一**的训练目标删掉，任何回归方法都不可能猜中——不是方法失败，是留一后这个问题没有解。

因此本笔把判定含义显式收窄为两条可引用、两条禁引：

- **可引用**：①该表在样本内可表示（`before` 三臂物件／地点全 1.00）；②在 1:1 词表下，短语→名字这条映射**只能靠写入**（地址层存表），不能靠从表示里拟合出来。
- **禁引用**：①“通用句向量不会把说法映射到名字”（R3 的 0.00 同样由留一造成，不含泛化信息）；②“表示层无桥”已被证明（R9 已把缺口定位在“短语→名字”这一支，R10 只证这一支在 1:1 词表下不可由拟合得到）。

“泛化 vs 背诵”的经验读数尚无，必须由 **R10b 多说法留一闸**取得：给每个名字补第 2 种描述说法（夹具扩展，其余冻结口径不动），再重复 `after` 列——那时删掉一种说法，另一种仍在词表里；读出仍掉到 chance 才叫“背过”，保住才叫“真泛化”。

**3. 触发原因与读数出处**：`reports/taiji_phrase_bridge_20261109.json` 的 `verdict=phrase_name_bridge_memorised`；`counts.bank_before={object:4,place:5}`、`counts.bank_after={object:3,place:4}`、`counts.deleted_per_held_out=1`；`arms.R1/R2/R3` 的 `before.accuracy` 均 1.00、`after.accuracy` 在 λ∈{1e-3,1e-1,1,1e1,1e3} 上均 0.00；`control_ok=true`（R1 样本内 ≥0.95，闭式通路正常，故 0 不能归给实现）。`cost={cpu:18.609, wall:26.34, cap:200}`。

**4. 代价**：本决策本身 **0 CPU**（不重跑）。代价在**证据强度**：R10 因此不能作为“否掉表示层桥”的依据，只能作为“写入式桥的必要性”依据；要让结论往前走，必须付 R10b 的夹具扩展成本（预计仍是数十 CPU 量级，跑前估，超 200 先立预算账）。

**5. 替代方案**：①把 0.00 直接写成“桥不存在”——**否决**：那是把构造性零当经验零，属于典型的口径污染；②改判据让 R3 的 0.00 参与判定——**否决**：R3 的 0.00 与 R1／R2 同因，改判据不增加任何信息，只会制造“判据被读数推动”的记录；③立刻加多说法重跑但复用同一 runner 与同一 manifest——**否决**：夹具变了，`fixture_sha256` 与 `counts` 都要动，必须作为新闸（R10b）重新预注册，不能在同一判据下落两套夹具。

**6. 撤回路径**：本笔只改文档口径，撤回＝把 SPEC §21.18、账本 §56、能力卡 U44 与三处队首里“构造性零／只能靠写入”的限定删掉、恢复为直接引用 `phrase_name_bridge_memorised` 的字面含义；runner／manifest／报告无需变动。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）；并按用户规则“归因优先看架构、汇报不许含糊”“方案取上限更高者”。

**8. 对已有结论的影响**：R9 的 `phrase_survives_name_link_absent` **不变**（该判定只看短语定位，与名字无关）；R8 的 `control_failed` 与 R7 的 `inconclusive` **不变**；R10 的 `before` 列（表可表示）是新增可信事实。**被降级**的只有“R10 的 0.00 可作泛化证据”这一读法——它从未被允许，本笔把它写死。

---

## D-20261109-12 R10b 落中区：两个方向都不采信，并把 λ 敏感性归因写清

**1. 编号／时间／类型**：D-20261109-12｜2026-11-09（读数之后、写文档之前）｜**判定解释类 ＋ 机制归因**：不改 runner、不改 manifest、不改读数、不改冻结判据；不重跑。

**2. 决策内容**：R10b 按读数前冻结的规则判 **`phrase_bridge_inconclusive`**。三件事必须一起写清，否则该读数会被误引：

- **为什么不是 `generalises`**：定判点 λ=1 上 R3 地点 `sentence_after` = **0.40**，门槛 0.80。
- **为什么不是 `memorised`**：该规则要求 `sentence_after` 在**整个 λ 网格**上都 ≤0.60，而网格最大值为 **0.75**（λ=1e-1），条件不成立。
- **为什么 0.40 不该被当成“半个能力”**：λ=1 对 **10 行**词表的岭回归是**过正则**——`(XᵀX+λI)⁻¹` 把预测压向零，行范数过小后**余弦最近名**退化成噪声。λ=1e-3／1e-1 上的 0.70／0.75 与 λ=1e1／1e3 上的 0.15／0.20 同源：读数由正则强度主导，而不是由“桥有没有泛化”主导。因此本轮**只能**得出“跨说法信号存在但不稳定”，不能得出任何方向的能力结论。

**可引用**：①R10 的 `before` 1.00 不是 1:1 词表的假象——两说法词表（8／10 行）在 λ=1 上仍 1.00；②跨说法信号只出现在**地点槽**，物件槽整个网格都在 chance（`sentence_after` 0.00–0.25）；③写入式桥（SPEC §21.18，9 条、0 训练）仍是唯一可靠路径。
**禁引用**：①“桥会背诵”（R10 的零是构造性的，R10b 未复现全网格零）；②“桥能泛化”（未达 0.80，且随 λ 翻转）；③把 0.75 当作“接近可用”（那是过正则最弱的一档，不是工作点）。

**3. 触发原因与读数出处**：`reports/taiji_phrase_bridge_alt_20261109.json` 的 `verdict=phrase_bridge_inconclusive`、`control_ok=true`、`project_line=project_bridge_memorised`；`arms.R3.object.primary` = before 1.00／after 0.00／sentence_before 0.90／sentence_after 0.05，`arms.R3.place.primary` = 1.00／0.20／0.95／**0.40**；`arms.R3.place.grid.*.sentence_after` = 0.70／0.75／0.40／0.15／0.20；`counts` = 9 names／18 phrases／20 sentences／bank 8→7、10→9／`deleted_per_held_out=1`；`cost={cpu:18.562, wall:24.94, cap:200, estimate_upper:40, threads:1}`；runner sha `d790f328…`。

**4. 代价**：本决策本身 **0 CPU**（不重跑）。代价在**结论强度**：R10 与 R10b 合起来只能把“桥必须写入”从“1:1 词表下成立”抬到“在两种说法下也未被推翻”，并没有给出“泛化率为 X”的可引用数字；把它写成数字需要固定一个非过正则的工作点（λ≪1）并重新预注册，那属于新闸，不在本轮。

**5. 替代方案**：①事后把定判 λ 改成 1e-1 去拿 0.75——**否决**：读数前已冻结 λ=1，读数后改工作点就是“按结果调判据”；②把 `memorised` 的“整个网格”改成“定判点”——**否决**：那会把 R10 的结论反向覆盖，且两轮不可比；③只报 0.40、不报 λ 网格——**否决**：λ 敏感性正是本轮最有信息量的部分，隐去等于把机制藏起来。

**6. 撤回路径**：删除 §21.19／§57／U45 与三处队首中的 R10b 段，恢复“下一动作＝R10b”的表述；runner／manifest／报告无需变动（本轮未改判据，故不存在需要回滚的判据）。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）；并按用户规则“归因优先看架构”“汇报不许含糊”“只给唯一一个下一步”。

**8. 对已有结论的影响**：R9 的 `phrase_survives_name_link_absent`、R8 的 `control_failed`、R7 的 `inconclusive`、R10 的 `phrase_name_bridge_memorised` **全部不变**。**新增可信事实**：两说法词表在样本内仍可表示（λ=1 三臂 1.00／1.00）。**被降级**的读法：①“R10b 证明了背诵”（不成立）；②“R10b 证明了泛化”（不成立）。**报告字段缺陷登记**：`contrast.r10b_R3.<slot>.sentence_after_worst_lambda` 实际取的是网格**最小**值（`min`），而判定用的是网格**最大**值；该字段只作展示、不参与判定，此处登记以免后续被误读，报告本身不改。

---

## D-20261109-13 R11 读数前修正两处“证据口径”：回滚证据改用内容摘要、未知短语改用 token 不相交筛选

**1. 编号／时间／类型**：D-20261109-13｜2026-11-09（首跑之后、正式冻结之前）｜**度量口径修正 ＋ 构造性缺陷登记**：只改 runner 与桥模块的“证据怎么算”，**不改判据、阈值、夹具、语料、电池**。

**2. 决策内容**：R11 首跑判 `written_bridge_mechanism_failed`，根因不是机制，是两处口径写错：

- **回滚证据用了含计数器的摘要**：`write()` 一进批就 `batches += 1`，冲突还会 `rejected_batches += 1`／`rollbacks += 1`；而 `digest()` 包含这些计数器 ⇒ 冲突批必然“前后不等”，`digest_unchanged=false`。改为新增 `content_digest()`（只含 `capacity/peak_capacity/records`），用它比较回滚前后；计数器本来就应当变。
- **“未知短语”是手工挑的语义无关词**：2 字节窗口会跨 UTF-8 字符边界，这些词继承了邻居字符的尾字节，实际与已写键共享 token——首跑 `unknown_tally={"ambiguous_multiple_anchors":1,"no_match":4}`。改为从 12 个候选里按“token 与全表完全不相交”自动挑前 5 条，并把被拒候选连同共享 token 写进 manifest 的 `unknown_selection.rejected`；冻结时选出、运行时断言一致。

**可引用**：①判据（≥0.95 且零错指）与机制/弃答/关闭态四组条件**一字未改**，重冻后正式读数 `written_bridge_operational`；②正式 runner sha `8be1bef0…`、桥 sha `ddb5a48b…`；③R11 关于“键对偏移敏感”的结论由这次缺陷直接暴露，属架构级限制。
**禁引用**：①把首跑 `written_bridge_mechanism_failed` 当作机制失败（它测的是计数器）；②把“手工挑的语义无关短语”当作“任意新短语都会被拒答”。

**3. 触发原因与读数出处**：首跑 `verdict=written_bridge_mechanism_failed`、`mechanism.conflict.digest_unchanged=false`、`abstention.unknown_tally={"ambiguous_multiple_anchors":1,"no_match":4}`；修正后 [manifest](../manifests/taiji_written_bridge_20261109.json) 的 runner／bridge 双列 sha 与 [报告](../../reports/taiji_written_bridge_20261109.json) 一致，`verdict=written_bridge_operational`。

**4. 代价**：修正本身 **0 新训练／0 新参数**；重跑成本 **1.672 CPU／1.68 wall**（估 60、上限 200）。

**5. 替代方案**：①保留旧 digest 并放宽机制判据——**否决**（等于按结果改判据，且旧口径测的是计数器不是地址内容）；②继续手工挑未知短语——**否决**（会重复“共享 token”的构造性污染，把构造缺陷当成机制读数）；③只报 5 条成功、隐去 4 条被剔候选——**否决**（隐去机制，违反“报不知道”的要求）。

**6. 撤回路径**：删除 §21.20／§58／U46 与三处队首的 R11 段，恢复“下一动作＝R11”；正式报告与 manifest 随之作废（首跑已不可引用，无需回滚判据）。

**7. 授权来源**：用户 2026-11-09 授权（“判据冻结后直接跑”“跑前估 CPU”“三条写结局各有读数”“同时报该说不知道的情形”）。

**8. 对已有结论的影响**：R9／R10／R10b **全部不变**。**新增可信事实**：9 条写入式记录在定位后可读回 **40/40**，且三写结局与两不知出口行为正确。**新增范围限制**：①键为 2 字节滑窗 token 集合，拒答保证只对 token 不相交成立；②`mechanism.conflict.digest_unchanged` 的正式口径是 `content_digest`；③`sentence_direct_wrong_pointers=4` 为诊断字段，不参与判定。

---

## D-20261109-14 R12 键单位对照：字节伪 token 确认是缺陷，但换码位未达零线，offset 只治位置漂移

**1. 编号／时间／类型**：D-20261109-14｜2026-11-09｜**架构判断 ＋ 键定义扩展**：在同一 9 条表、同一 12 个未写候选、同一 0 误答线上并列 2 字节／码位／码位＋字符位三臂；判据、夹具、候选清单**未按结果增删**。

**2. 决策内容**：

- **键函数成为参数**：`BridgeTable(key_fn=...)`；`byte_pair_key` 是 R11 `phrase_key` 的逐字节等价别名（runner 对 9 条短语逐条断言），`codepoint_key` 按 UTF-8 码位整取、非法字节返回空键，`codepoint_offset_key` 加字符位标签（诊断臂）。R11 的读数在提交 `0c8cad59e` 上可复现；R12 报告里的 bridge sha `248a0493…` 是扩展后的文件。
- **判定 `bridge_key_unit_still_leaks`**：2 字节版 3/12、码位版 4/12 给出非空答案，**都没到 0**。R11 的 4 个剔除候选在码位版**全部归零**（2 字节版只有 1 条原本会作答），字节碎片 **12→0**、整句 **4→0**、改写错指 **4→0**——所以“键单位选错”这条**部分成立**；但码位版另有 4 条单字共享碰撞（“长”“木”“带”“木”），总数反多 1 条，因此**不得**宣称“换码位就修好了”。
- **offset-aware 的处理范围**：未知 4→1、碎片 12→0、整句 4→0、改写错指 4→0，残留 1 条“同字同位”（深红木箱的“木”在字符位 2 与“四方木台”同位）。⇒ 它治**位置漂移**，不治**同字同位**，也不治任何其他共享；原预注册的 offset-aware 不能作为零线的收尾方案。
- **下一动作**：R13“查询必须被表解释完”闸——一条**结构性、无阈值**的读规则（查询含任何表外 token 即 `no_match`），在同一 12 候选＋9 改写上测未知误答是否归零，并要求 40/40 能力不降、关闭态九项仍全等。

**3. 触发原因与读数出处**：[报告](../../reports/taiji_bridge_key_unit_20261109.json)、[manifest](../manifests/taiji_bridge_key_unit_20261109.json)；三臂能力均 40/40、错指 0；成本 **1.781 CPU／2.549 wall**（估 60、上限 200）。

**4. 代价**：0 新训练／0 新参数；键函数仅 3 个纯函数，`BridgeTable` 仅新增一个默认参数；R11 的旧读法与判据不受影响。

**5. 替代方案**：①只跑 2 字节版并宣布“未写短语会误答”→ **否决**（漏掉“碎片被当地址”这一更硬的证据）；②只跑码位版并宣布“键单位修好了”→ **否决**（4 条单字碰撞仍在，且比 2 字节版多）；③直接上 offset-aware 收尾 → **否决**（残留同字同位，零线仍不达）；④按结果删掉会碰撞的候选 → **否决**（等于改输入凑结论）。

**6. 撤回路径**：删除 SPEC §21.21／账本 §59／U47／D-14 与三处队首的 R12 段，`BridgeTable` 去掉 `key_fn` 参数，恢复“下一动作＝R12”；正式报告与 manifest 随之作废。

**7. 授权来源**：用户 2026-11-09 授权（先跑原预注册对照、再跑码位键；两版都不为 0 才上 offset-aware 并写明它治哪类失败；关闭态零扰动、跑前估 CPU、判据冻结后直接跑）。

**8. 对已有结论的影响**：R11 的 `written_bridge_operational` **不变**（它只测“已写记录能读回＋两不知出口”，未声称任意新短语必拒答）。**新增可信事实**：字节碎片在 2 字节键下 12/18 被当成地址，是拆字伪 token 的直接证据；码位键清掉这一类。**新增限制**：码位键仍有单字共享碰撞；offset-aware 只治位置漂移；在 R13 之前，桥**不得**扩规模或接进默认图。

---

## D-20261109-15 R13 身份收口＋别名写入：零线成立，代价是同义说法全拒答

**1. 编号／时间／类型**：D-20261109-15｜2026-11-09｜**架构判断 ＋ 键身份改写**：把键从“字符集合”改为“整串字符（保序保重数）”，读从“共享词即锚”改为“整条键连续出现在查询里才作答”；同日把 9 条同义说法作为附加键写入（一值多键）。判据、夹具、12 候选**未按结果增删**，两段**分开判**。

**2. 决策内容**：

- **键身份**：`sequence_key` 返回字符元组（保序、保重数）；`BridgeTable(read_mode="sequence")` 用 `contains_sequence` 判连续包含。`read_mode` 默认 `"token"`，R11／R12 行为与摘要不变（`_record_payload` 只对非 frozenset 键改用保序列表）。空键／非法 UTF-8 一律 `no_match`／`empty_key`。
- **A 段判定 `sequence_identity_clean`**：12 未写候选 **0**、18 条半字碎片 **0/18**、20 条整句误指 **0**（20/20 `ambiguous_multiple_anchors`）、能力 **40/40** 错指 0、关闭态九项全等。
- **A 段的代价（写进判定同段）**：9 条同义改写只靠原键时**正确 0／错指 0／`no_match` 9**——零误答来自“不答”，不是“理解了同义”。
- **B 段判定 `alias_write_clean`**：别名批 `accepted`、`appended 9／merged 0／replaced 0`、别名 **9/9 读回同一实体**、写后三项零线**仍为 0**、原 9 条仍 40/40；键 **18**／实体 **9**／比值 **2.0**。
- **下一动作**：R14“全量说法写入”闸——把夹具里每个实体的全部说法（20 `describe`＋60 paraphrase）都写成键，测 80/80 读回、键/实体比值与三项零线是否仍为 0。

**3. 触发原因与读数出处**：R12 判 `bridge_key_unit_still_leaks`，4 条碰撞全部来自“共享一个单字”，说明失败不在键单位而在**身份定义**。读数：[报告](../../reports/taiji_bridge_sequence_identity_20261109.json)、[manifest](../manifests/taiji_bridge_sequence_identity_20261109.json)；成本 **1.766 CPU／2.535 wall**（估 30、上限 200）。

**4. 代价**：0 新训练／0 新参数；模块新增 1 个键函数、1 个包含判定、1 个读出分支（`read_mode` 默认关），`find`／`write` 的类型注解放宽但行为不变。**能力代价**：放弃模糊匹配，未写入的说法一律不答。

**5. 替代方案**：①继续用字符集合＋“查询必须被表解释完”规则 → **否决**（“长”“木”本来就在表内，该规则拦不住这类碰撞）；②上 offset-aware 收尾 → **否决**（R12 已测，残留同字同位且整句过度回答上升）；③改用编辑距离/相似度阈值 → **否决**（引入阈值即引入调参，且会把“像”重新变成身份）；④只写别名不做身份收口 → **否决**（前提错误：不收口则别名也在共享字上碰撞）。

**6. 撤回路径**：删除 SPEC §21.22／账本 §60／U47／D-15 与三处队首的 R13 段，`BridgeTable` 去掉 `read_mode` 参数与 `read_sequence`，恢复“下一动作＝R13”；报告与 manifest 随之作废。

**7. 授权来源**：用户 2026-11-09 授权（A 段身份收口、B 段别名写入；两段分开报；A 过 B 不过要如实报并说明失败类；同时报改写变 `no_match` 的条数与“只靠原键能答多少”；跑前估 CPU、关闭态零扰动、判据读数前冻结）。

**8. 对已有结论的影响**：R11 `written_bridge_operational` 与 R12 `bridge_key_unit_still_leaks` **均不变**（前者测读回与两不知出口，后者测键单位对照）。**新增可信事实**：整串连续包含可把未写短语、字节碎片、整句误指同时压到 0，且不损失 40/40；一值多键可写入并读回。**新增限制**：①零误答是“不答”换来的，9/9 同义说法在 A 段全 `no_match`；②别名必须逐条写入，**不得**由相似度自动生成；③规模仍 9 实体／18 键，不证扩大后碰撞率；④桥**仍不得**接进默认图（关闭态零扰动是硬约束）。

---

## D-20261109-16 R14 全量说法写入＋一说法两实体对抗：比值随说法数上升、冲突必须显式拒绝

**1. 编号／时间／类型**：D-20261109-16｜2026-11-09｜**架构判断 ＋ 写冲突策略新增**：给写入端加 `on_conflict`（`replace` 默认／`reject` 新增），并用三段读数回答“成本随规模怎么变、密表零线是否仍成立、同一说法指向两个实体怎么办”。判据、12 候选、18 碎片、20 整句、9 改写**未按结果增删**，三段**分开判**。

**2. 决策内容**：

- **冲突策略**：`BridgeTable(on_conflict="reject")` 在同键异值时返回 `rejected_batch_address_already_bound`，整批不应用、`content_digest` 不变、先写者保留；同键同值仍走 `merged`，批内同键异值仍走 `rejected_batch_address_collision`。默认仍为 `replace`（R11 行为不变、摘要逐字节不变）。
- **C 判定 `cost_curve_measured`**：键/实体 **9/9＝1.0**（min／中位／max 均 1）→ **18/9＝2.0**（均 2）→ **78/9＝8.667 且 `admitted=false`**。比值随说法数增长、不随实体数收敛；“80 键”因 **60 句含实体名**而不可达。
- **D 判定 `large_table_zero_clean`**：18 键表上 12 未写候选 **0**、18 条半字碎片 **0/18**、20 整句误指 **0**、能力 **40/40**。
- **E 判定 `conflict_rejected_first_writer_kept`**：3 例全部 `rejected_batch_address_already_bound`、摘要不变、先写者仍读回、同批合法项未写入；`replace` 对照为 `accepted`／`replaced 1`／读出转后写者。
- **下一动作**：R15“句子级两段读”闸——先取查询里连续包含的最长键作答，再遮蔽该片段重读，测 20 句能否读出 40/40 个实体、12 未写候选是否仍为 0。

**3. 触发原因与读数出处**：R13 只报“18/9＝2.0”一个比值，无法回答“说法变多时成本怎么走”；且别名一多就必须面对“同一说法被两个实体声明”的写冲突。读数：[报告](../../reports/taiji_bridge_wording_scale_20261109.json)、[manifest](../manifests/taiji_bridge_wording_scale_20261109.json)；成本 **1.891 CPU／2.418 wall**（估 30、上限 200）。

**4. 代价**：0 新训练／0 新参数。模块新增 1 个构造参数、1 个拒绝分支、1 个计数键（`rejected_reasons`）；`read()`／`read_sequence()` 未改。**能力代价**：`reject` 下后写者被整批拒绝（含同批合法项），写入必须一次对齐实体，不能靠“后写覆盖”。

**5. 替代方案**：①默认改 `reject` → **否决**（会改变 R11 已冻结的替换读数与摘要，旧证据不可比）；②冲突时静默保留先写者、其余项照写 → **否决**（正是用户点名禁止的“静默取其中一条”）；③按相似度自动合并冲突说法 → **否决**（引入阈值即引入调参，且语义合并无法判定）；④把 60 条含名字的句子去掉名字再写入 → **否决**（改动夹具语料，读数不可比）。

**6. 撤回路径**：删除 SPEC §21.23／账本 §61／U49／D-16 与四处队首的 R14 段；`BridgeTable` 去掉 `on_conflict` 参数与 reject 分支，恢复“下一动作＝R14”；报告与 manifest 随之作废。

**7. 授权来源**：用户 2026-11-09 授权（R14 全量说法写入闸；成本曲线要按实体出分布并报随规模的变化、比值上升要写明；大键表重跑三项零线；必测“同一说法两个实体”，绝不允许静默取一条；关闭态九项全等；跑前估 CPU、超 200 先写台账；跑完自行继续推进）。

**8. 对已有结论的影响**：R11／R12／R13 的判定与读数**全部不变**（本轮只加策略与更大键表）。**新增可信事实**：在更密键表上三项零线仍为 0、能力不降；写冲突有显式、可复核的拒绝出口。**新增限制**：①成本比值随说法数上升，**不得**再报为常数；②78 键刻度被拒未跑，不得引用；③先写者保留依赖显式 `reject`；④规模仍 9 实体／18 键；⑤桥**仍不得**接进默认图。

---

## D-20261109-17 R15 句子级两段读：`replace` 路径删除、零误答仍靠“不答”换来

**1. 编号／时间／类型**：D-20261109-17｜2026-11-09｜**架构判断 ＋ 机制收敛**：把 R14 的“E 段教训”（默认 `replace` 会静默转主）从**文档约定**改成**结构约束**——从模块里删掉 `replace`／`history` 路径，使“后写者拒”成为唯一分支；同时新增句子级两段读。判据在读数前冻结，四判定分开。

**2. 决策内容**：

- **删除 `replace` 路径（结构）**：`BridgeTable.__init__` 去掉 `on_conflict` 参数与 `replaced` 计数；`values_at()`／`history_at()` 删除；`write()` 的 replace 分支与 `snapshot/restore` 空操作删除；拒绝分支**无条件**执行（同址异值 → `rejected_batch_address_already_bound`＋`rollbacks`＋原因计数），接受路径末尾 `raise AssertionError("unreachable")` 防漏。写结局只剩 appended／merged／rejected。

- **两段读（新增 `read_sentence`）**：第一遍按 `_earliest_contained` 取**起点最早**的已写键；命中后把该键**首次出现**的整段按**同字符宽度**替换为 `MASK_CHARACTER="\x00"`（位置不移动），第二遍再读。第二遍 0 锚 → `no_match`（`reason=masked_no_match`）；同起点两记录 → `ambiguous_same_start`（弃答，不猜）；同址重复 → `answered_single_duplicate_key`（按 1 个实体计）；只有 1 个答案 → `answered_single`。

- **判定**：A `replace_path_absent`、B `two_pass_clean`、C `zeros_clean`、D `exits_declared`。

**3. 触发原因与读数出处**：R14 的 `on_conflict="replace"` 默认会在同一说法被两实体声明时**静默转主**，而“接线时要选 reject”只是一句文档——它不构成保证。用户要求把它变成结构并给出“不可达”的证据。两段读则是单遍直读 20/20 多锚的直接后果（同一句话里有物件与地点两条已写说法，一起命中即弃答）。读数：[报告](../../reports/taiji_bridge_two_pass_20261109.json)、[manifest](../manifests/taiji_bridge_two_pass_20261109.json)；成本 **2.016 CPU／2.684 wall**（估 30、上限 200，无需预算台账）。

**4. 代价**：0 训练／0 新参数。**新增能力**：20 句 × 2 槽 40/40（答出 2 个 20、只答 1 个 0、答错 0、弃答 0）。**能力代价**：① 两段读是**两遍**读，成本翻倍；② “起点最早”是**阅读顺序**规则、不含语义，同义说法仍不答；③ **旧 runner 在当前 HEAD 不可跑，按提交号可复跑**：R11–R15 的 runner 绑定各自版本的模块（R14 及以前构造时传 `on_conflict`），当前模块下会抛 `TypeError`／`AttributeError`；但它们的字节都在 git 历史里，检出对应提交即可复跑——R11→`0c8cad59e`、R12→`31ef1850c`、R13→`ff79e7094`、R14→`edcf623ca`、R15→`c1fdeaa7d`（每个 runner 的 blob sha1／sha256 与模块 sha256 见 R16 报告的 `rerun_anchors` 表）；④ 位置相交的两个已写说法本轮未测（R16 已补测）。

**5. 替代方案**：① 保留 `on_conflict` 参数、默认改 `reject` → **否决**（参数存在即“可被写成 replace”，用户要的是不可达而非默认值改变；且会保留一条无调用者的分支）；② 保留参数但在构造时对 `"replace"` 抛 `ValueError` → **否决**（仍是“参数存在”，且 AST 里留着 `replace` 字面量，扫描证据不再干净）；③ 两段读改为“取最长键” → **否决**（本轮 20 句里物件与地点说法同为 5 字，最长键并列，实测 16 句 `ambiguous`）；④ 遮蔽改用删除而非替换 → **否决**（位置会移动，第二遍的起点会漂移，错误不可归因）；⑤ 保留 `values_at`／`history_at` 供旧 runner 复跑 → **否决**（它们的存在正是 `replace` 语义的残留表面积，且会诱导“同键多值”的错误心智）。

**6. 撤回路径**：`git revert` 本笔代码提交即可恢复 `on_conflict`／`replace`／`values_at`／`history_at` 与旧 `write()`；同时删除 SPEC §21.24、账本 §62、U50，把 U49 限制栏与四处队首改回 R14 段；R15 报告与 manifest 随之作废。文档提交可单独 revert（两笔独立）。

**7. 授权来源**：用户 2026-11-09 授权（R15 句子级两段读；把 E 段教训变成结构并给出 `replace` 不可达的证据，做不到要报风险面；把“键/实体＝每实体说法数、不随实体数收敛、写多少认多少、不具备同义压缩”写进能力卡限制栏并与“不得说桥理解了同义说法”并列；两段读要报遮蔽规则、无唯一锚点的出口、两实体共键的输出与四类分布；关闭态九项全等；跑前估 CPU、超 200 先写台账；跑完自行继续推进）。

**8. 对已有结论的影响**：**R11–R14 的判定与读数全部不变**（旧读数本身未改；变的是模块的写入口，使 `replace` 不再可选）。**新增可信事实**：① `replace` 在桥里**不可达**（四重证据），写冲突不再依赖调用方纪律；② 一句话两个槽位可以各读一次都答出（40/40），且单遍直读 0/20 说明两遍不是冗余；③ 三出口把“只答 1 个／同址重复／同起点并列”分别命名，不再混成“partial”。**新增限制**：① 零误答仍是“不答”换来的，**不得**说桥理解了同义说法；② 键/实体＝说法数、不随实体数收敛 ⇒ **写多少认多少、无同义压缩**；③ “起点最早”不含语义；④ 旧 runner 在当前 HEAD 不可跑、按提交号可复跑，历史证据不得据新模块重算（R16 报告 `rerun_anchors` 已逐轮登记提交号与 blob sha）；⑤ 桥**仍不得**接进默认图。

---

## D-20261109-18 R16 说法重叠与三实体句：判据读数前冻结，并给旧轮次逐个登记复跑入口

**1. 编号／时间／类型**：D-20261109-18｜2026-11-09｜**判据冻结（读数前新增两个几何情形）＋记账要求（复跑锚）**。不改地址层、不改排序、不改零线；只把 `read_sentence` 从“固定两遍”扩为“可配遍数”，并把“旧 runner 可复跑”的入口逐轮写死。八例期望值在首跑前写入 manifest 并冻结。

**2. 决策内容**：

- **模块（唯一改动）**：`read_sentence(query, max_passes=2)` 改成循环，直到“首次非命中／同址重复／遍数上限”三者之一；输出新增 `masks`（全部遮蔽记录）、`stopped_at_pass`、`final_exit`；出口新增 `answered_triple`／`answered_multi`；同址重复时同时改写 `reason="same_key_answered_twice"`。默认 `max_passes=2` 使 R15 行为逐项不变。
- **判据（读数前冻结）**：重叠 4 例（`nested_long_first`／`share_one_character`／`abutting_no_overlap`／`partial_overlap_no_containment`）各自预声明 `answers/values/decision/stopped_at_pass/final_exit/reason/swallowed`；三实体 4 例（`three_disjoint`／`middle_wording_swallowed`／`three_records_same_start`／`same_address_three_times`）同样预声明。**“吞”的判据是几何的**：某已写键的首次出现区间与该键以外的任何遮蔽区间相交，即记为 `swallowed`——不靠“答出几个”打分。
- **记账（用户要求）**：R11–R15 每轮登记“最后可跑提交＋runner blob sha1＋报告 sha256＋该提交下桥模块 sha256＋`matches_report`”，措辞统一为“**在当前 HEAD 不可跑，按提交 X 可复跑**”。

**3. 触发原因与读数出处**：R15 只测了 20 句“两条说法互不重叠”的情形，“起点最早＋同宽遮蔽”是否会把另一条说法所需的字符一起抹掉，没有读数；三实体句能否剥到 3 个也未测。另用户指出“不可复跑”说过头：旧 runner 只是构造函数签名变了（R14 及以前传 `on_conflict`），字节仍在 git 历史里。读数：[报告](../../reports/taiji_bridge_geometry_20261109.json)、[manifest](../manifests/taiji_bridge_geometry_20261109.json)；成本 **1.578 CPU／2.252 wall**（估 40、上限 200，无需预算台账）。

**4. 代价**：0 训练／0 新参数。**新增读数**：重叠 **4/4** 与冻结判据逐项相同、误指 0；`abutting_no_overlap` 仅相邻时答 **2**（`answered_pair`）说明“不相交就不受扰”；三实体不相交 **3/3**、停第 **4** 遍且由 `no_match` 退出（**非遍数上限截断**）；跨接说法 **2/3**、停第 **3** 遍、吞 `乙丙丁`；同起点 **0 答**（`ambiguous_same_start`）；同址三现计 **1 实体**。**能力代价**：① 位置相交即在该层停，被吞时出口是“不知道”，不报错也不猜值；② 同起点仍是弃答；③ 只测到 3 实体；④ 遍数是上限保护，不是能力提升。

**5. 替代方案**：① 遮蔽改为只抹“该键独占的字符” → **否决**（会引入语义判断，破坏“键＝整串字符保序保重数”的单一身份规则，且无法在等宽遮蔽下保持位置不漂移）；② 重叠时改答“两个都答” → **否决**（相交时无法判定后一条的边界，等同猜值）；③ 三实体句改为按分句切分再读 → **否决**（引入标点/句法依赖，本轮零参数、0 训练的口径会被打破）；④ 旧 runner 只在文档里写“不可复跑” → **否决**（用户指出的正是这一点：没有入口即孤本）；⑤ 不扩 `max_passes`、另写一个三遍读函数 → **否决**（会产生两套阅读语义，退出码不可比）。

**6. 撤回路径**：`git revert` 本笔代码提交即可回到 R15 的固定两遍读（`max_passes` 参数、`masks`／`stopped_at_pass`／`final_exit`、`answered_triple`／`answered_multi` 一并消失）；R16 报告与 manifest 随之作废；文档侧删除 SPEC §21.25、账本 §63、U51、D-18，并把四处队首改回 R15 段。代码与文档两笔独立，可分别 revert。

**7. 授权来源**：用户 2026-11-09 授权（R16 专测说法重叠与三实体句，判据读数前冻结；把“不可复跑”改准为“在当前 HEAD 不可跑、按提交 X 可复跑”并逐轮登记提交号与 blob sha；重叠要报遮蔽是否会误删另一条说法所需字符，三实体要报剥到几个、在第几层停、出口是什么；关闭态九项全等；跑前估 CPU、超 200 先写台账；跑完自行继续推进）。

**8. 对已有结论的影响**：**R11–R15 的判定与读数全部不变**（默认 `max_passes=2` 下 R15 的 20 句仍 40/40、分布 20/0/0/0）。**新增可信事实**：① 遮蔽的边界是**几何**的——说法互不相交（含仅相邻）时下一条照常读出，位置相交时该层即停；② 三实体句在说法不相交时可剥到 3/3，且停层由 `no_match` 给出而不是被遍数截断；③ 八例的“停层＋出口”全部预声明并逐项命中。**新增限制**：① 不得表述为“能处理任意重叠”；② 被吞是“不知道”而非“答错”；③ 同起点仍弃答；④ 只测到 3 实体；⑤ 同义压缩未变（承接 U49①②），重叠处理不产生任何跨说法理解；⑥ 桥**仍不得**接进默认图。**记账更正**：S-84／D-17／SPEC §21.24 的“不可复跑”统一改为“在当前 HEAD 不可跑，按提交 X 可复跑”，逐轮入口见下表。

| 轮次 | 最后可跑提交 | runner blob sha1 | 报告 runner sha256 | 该提交下桥模块 sha256 | 报告与 blob 一致 |
|---|---|---|---|---|---|
| R11 | `0c8cad59e` | `4b35beed…` | `8be1bef0…` | `ddb5a48b…` | ✓ |
| R12 | `31ef1850c` | `255cb07c…` | `3d484121…` | `248a0493…` | ✓ |
| R13 | `ff79e7094` | `0f430971…` | `d600c33a…` | `d48821ba…` | ✓ |
| R14 | `edcf623ca` | `490e0050…` | `da86411e…`（磁盘 CRLF） | `104077f8…` | ✗（blob `b807deff…` 为 LF，代码相同，已声明） |
| R15 | `c1fdeaa7d` | `36c77d23…` | `b6721d6f…` | `967017e1…` | ✓ |

---

## D-20261109-19 R17 同值包含压缩：判据读数前冻结，键级压缩不进入默认路径

**1. 编号／时间／类型**：D-20261109-19｜2026-11-09｜**判据冻结（读数前新增四个压缩算例）＋架构判断（键级压缩不进入默认路径）**。不改地址层、不改排序、不改零线、**不改桥模块**；压缩算法只在 runner 内运行，测的是候选操作。四个算例的期望值在首跑前写入 manifest `definition_digest`。

**2. 决策内容**：

- **A 值级去重**：三条键绑同一个值，分别报告记录数、不同值数、键减少量，防止把“一个值”误称为“键压缩”。
- **B 同值包含**：长键可删当且仅当①短键与长键**同值**、②短键是长键的**连续子串**、③**没有其他记录的整条键**完全落在长键内部且在被保留区间之外（表级守卫）。短键取“最短者优先、同长取小下标”，删除顺序按长键降序。
- **C 边界反例**：另一条键只有一半落在长键内（跨边界），守卫看不到它，压缩放行后读出一个压缩前不可见的值。
- **D 跨值包含**：长键包含另一值的短键时**一律不删**，并登记跨值对；强行删除会答错值。
- **默认表**：9 条语料键＋9 条别名压缩后必须**删除 0 条**、`content_digest` 不变、R15 40/40、零线 0/0。

**3. 触发原因与读数出处**：R14 的 C 段把成本曲线量到 18/9＝2.0，并写死“键/实体＝说法数”；R16 测完阅读几何后，唯一没回答的问题是“这个比值能不能靠结构操作降下来”。R17 的回答是分层的：值可以只存一份，键不能自动减少。读数：[报告](../../reports/taiji_bridge_compression_20261109.json)、[manifest](../manifests/taiji_bridge_compression_20261109.json)；成本 **1.781 CPU／2.648 wall**（估 30、上限 200，无需预算台账）。

**4. 代价**：0 训练／0 新参数／0 优化器步。**读数**：值级去重 3 键→1 值、键减少 **0**；安全嵌套 2→1、压缩前同起点答 0、压缩后答 1（停第 2 遍）、短键仍可答；残差守卫 3→3、`blocked=[甲乙丙丁]`、无守卫模拟会答 2 个值；边界泄漏 3→2、压缩后答 2 个值；跨值 2→2、登记 1 对、强行删长键会答 `木桌`（错值）；默认表 18 条、删除 0、摘要不变、40/40、零线 0/0、九项全等。**能力代价**：① 值级去重不减少键数；② 键级压缩只在“安全嵌套”几何成立，守卫不是查询级保证；③ 跨值压缩必须禁止；④ 默认路径不压缩。

**5. 替代方案**：① 对**所有**同值键自动压缩 → **否决**（边界反例会多答一个值，见 R-D29-286）；② 允许**跨值**包含压缩 → **否决**（R-D29-287：会把长说法的值答成短键的值）；③ 把“值只存一份”当作键压缩 → **否决**（R-D29-283：键减少量为 0）；④ 引入相似度阈值决定删除 → **否决**（重新引入调出来的常数，违背本轮“无阈值”口径）；⑤ 直接把压缩写进 `BridgeTable` 默认写路径 → **否决**（没有查询级安全保证，且会改变 R15/R16 的冻结基线）；⑥ 不测边界、只报安全嵌套 → **否决**（等于把失败几何藏起来）。

**6. 撤回路径**：删除 R17 runner／manifest／报告即可，桥模块未被修改、无需回退；文档侧删除 SPEC §21.26、账本 §64、U52、D-19，并把四处队首改回 R16 段。若未来批准把压缩写进模块，必须另开一轮并重新冻结 R15/R16 的回归。

**7. 授权来源**：用户 2026-11-09 `/goal continue`（继续推进唯一后继 R17；判据读数前冻结；0 训练／0 优化器步／0 Adam／0 confirm；关闭态九项全等；跑前估 CPU、超 200 先写台账）。本轮不改桥模块，属于读数与判据冻结，不涉及方向二选一。

**8. 对已有结论的影响**：**R15／R16 的读数全部不变**（默认表 40/40、零线 0/0、九项全等；`address_bridge.py` 字节未动，R16 的 `bridge_sha256` 在当前 HEAD 仍对得上）。**新增可信事实**：① 值级去重是记账事实，不是键压缩；② 同值包含删长键是无阈值操作，且能救回“压缩前因同起点而弃答”的长说法；③ 表级守卫能挡下“残差整键”，但看不到“跨边界键”，因此删除不是查询级安全；④ 跨值包含必须排除。**新增限制**：① 不得说桥能压缩同义说法；② 键/实体比仍由说法数决定；③ 键级压缩必须由外部限定查询集或标注；④ 边界例多出的值是可见性变化，不是语义；⑤ 桥仍不得接进默认图。**下一步**：R18“显式纠正协议”闸——先写者保留使第一次写错就永久错；只测“调用方给出显式纠正凭据、旧值可审计撤回、普通冲突写仍整批拒绝”的一种方式，判据读数前冻结。

---

## D-20261109-20 R18 显式纠正协议：CAS 纠正与审计日志，普通冲突写仍拒

**1. 编号／时间／类型**：D-20261109-20｜2026-11-09｜**判据冻结（读数前新增纠正算例）＋机制新增（`correct` 比较并交换与审计日志）**。普通 `write` 的冲突规则未动；新增的是独立方法，不是 `write` 的参数、不是第二条冲突规则。六个纠正算例与默认表往返在首跑前写入 manifest `definition_digest`。

**2. 决策内容**：

- **新增 `BridgeTable.correct(items)`**：每项 `(text, expected_value, new_value)`。校验顺序：键非空 → 地址存在 → **当前值逐字节等于 `expected_value`**（凭据）→ 新值不同 → 批内同键不重复；全部通过后才改值。任一失败整批拒绝，`content_digest` 不变。
- **审计日志**：每次成功纠正向 `self.corrections` 追加 `{key, from, to}`，只增不改；`digest()` 含日志与纠正计数，`content_digest()` 仍只含地址与容量（往返可复原）。
- **不触碰 `count`**：纠正不是观测；`count` 仍只由 `write` 的 merge/replace 路径增加。
- **普通 `write` 不变**：同键不同值仍 `rejected_batch_address_already_bound`，盲写者不能改地址。
- **复跑锚**：R16→`93e292840`、R17→`f6b627fd8`；每行带 runner blob sha1、sha256、报告 sha256 与当时模块 sha256。

**3. 触发原因与读数出处**：R15 删除了 `replace`，R14/E 证明“后写者赢”会静默转主；代价是第一次写错就永久错。R18 要回答的是“能否在不恢复后写者赢的前提下纠正”。读数：[报告](../../reports/taiji_bridge_correction_20261109.json)、[manifest](../manifests/taiji_bridge_correction_20261109.json)；成本 **1.703 CPU／2.541 wall**（估 30、上限 200，无需预算台账）。

**4. 代价**：0 训练／0 新参数／0 优化器步；模块新增一个方法与四个计数字段。**读数**：普通冲突写仍拒（内容摘要不变、读回原值、日志 0）；CAS 纠正一步生效（读回新值、日志 1）；过期凭据拒绝（读回已纠正值、日志仍 1）；未写地址拒绝；批内撞键整批拒绝；纠正链 `蓝→红→绿` 两步有序、读回绿色；18 键表往返 `蓝→红→蓝` 后 `content_digest` 复原、R15 **40/40**、零线 **0/0**、关闭态**九项全等**。**能力代价**：① 凭据是当前值本身，知道当前值就能纠正，桥不判断权限；② 协议不校验新值对错；③ 日志只增不改，旧值不再可读为当前值；④ 纠正不改变键数、不产生语义。

**5. 替代方案**：① 恢复 `replace`／后写者赢 → **否决**（R14/E 已测会静默转主，且 R15 已把该路径删除）；② 给 `write` 加 `on_conflict="correct"` 参数 → **否决**（把纠正混进写冲突规则，重新制造“接线时选错”的风险面）；③ 允许盲覆盖但记日志 → **否决**（盲写者仍能改地址，日志只能事后看，不能阻止）；④ 用新键写纠正值 → **否决**（新键不匹配原说法，读不回来）；⑤ 先撤回再写入两步 → **否决**（中间态可被读到，且两步之间可被插写；CAS 是原子的）；⑥ 不加审计日志 → **否决**（旧值一改就蒸发，无法复核）；⑦ 让桥自己判断新值对错 → **否决**（需要语义与权限，超出本轮零语义口径）。

**6. 撤回路径**：`git revert` 本笔代码提交即可删除 `correct` 方法与审计日志，回到 R17 的桥；R18 报告与 manifest 随之作废；文档侧删除 SPEC §21.27、账本 §65、U53、D-20，并把四处队首改回 R17 段。R16／R17 的复跑锚仍按各自提交有效。

**7. 授权来源**：用户 2026-11-09 `/goal continue`（继续推进唯一后继 R18；判据读数前冻结；0 训练／0 优化器步／0 Adam／0 confirm；关闭态九项全等；跑前估 CPU、超 200 先写台账）。本轮属于机制新增，方向由 R15 之后唯一悬而未决的“写错不可纠”问题决定。

**8. 对已有结论的影响**：**R15／R16／R17 的读数全部不变**（默认表 40/40、零线 0/0、九项全等；两段读、重叠几何、压缩边界均未动）。**但 R16／R17 在当前 HEAD 不可跑**：它们 manifest 固定了旧模块 sha，R18 改了模块；按 `93e292840`／`f6b627fd8` 可复跑，已逐行登记。**新增可信事实**：① 纠正可以是一步 CAS，不需要恢复后写者赢；② 过期凭据/未写地址/批内撞键都整批拒绝，没有半批落盘；③ 纠正链有序可审计，往返后内容摘要复原；④ 普通盲写仍被拒。**新增限制**：① 不判断权限；② 不校验新值对错；③ 旧值只在日志里；④ 纠正不改变键数、不产生语义；⑤ 桥仍不得接进默认图。**下一步**：R19“读—核验—纠正闭环”闸——读出一个值，外部冻结核对器给出正确值，若不同则用读到的值作凭据纠正；测一步纠正、过期读被拒、核对器给错时协议无法识别（如实报边界）、普通冲突写仍拒、关闭态九项全等。判据读数前冻结。

---

## D-20261109-21 R19 读—核验—纠正闭环：单键闭环可用，核对器本身不可校验

**1. 编号／时间／类型**：D-20261109-21｜2026-11-09｜**判据冻结（读数前新增闭环算例）＋架构判断（单键闭环可用，核对器不可校验）**。不改地址层、不改排序、不改零线、**不改桥模块**；闭环算法只在 runner 内运行。八个算例在首跑前写入 manifest `definition_digest`。

**2. 决策内容**：

- **闭环定义**：一次迭代＝读键、把读到的值与冻结核对器比较、不一致时调用 `correct(键, 读到的值, 核对器值)`；**读到的值就是凭据**，不做任何硬编码。
- **A 一步纠正**：表 `蓝色杯子`、核对器 `红色印章` → 读出蓝、`corrected`、读回红、日志 1。
- **A2 读值作凭据**：表 `红色印章`、核对器 `绿色瓶子` → 日志 `红→绿`，证明凭据来自读值。
- **B 已一致**：`agreed`、日志 0、内容摘要不变。
- **C 过期读**：先读蓝、他人改红、再用旧读纠正 → `rejected_batch_correction_mismatch`、读回红、日志仍 1。
- **D 核对器给错**：真值蓝、核对器绿 → 协议接受、读回绿；`verifier_error_accepted=true`、`final_matches_ground_truth=false`。
- **E 普通冲突写**：仍 `rejected_batch_address_already_bound`、读回原值、日志 0。
- **F 收敛**：两轮，第一轮 `corrected`、第二轮 `agreed`、日志恰好 1。
- **G 默认表往返**：18 键表别名 `蓝→红→蓝`，`content_digest` 复原、R15 40/40、零线 0/0、九项全等。

**3. 触发原因与读数出处**：R18 只测了“纠正原语”本身；真正要回答的是“读到错值后，能不能用读到的值把错值改对，并且不破坏别的状态”。R19 把这个回路写成一次迭代并冻结每个出口。读数：[报告](../../reports/taiji_bridge_correction_loop_20261109.json)、[manifest](../manifests/taiji_bridge_correction_loop_20261109.json)；成本 **1.766 CPU／2.592 wall**（估 30、上限 200，无需预算台账）。

**4. 代价**：0 训练／0 新参数／0 优化器步；未改模块。**读数**：一步纠正 1/1；读值作凭据（`红→绿`）1/1；已一致不写 1/1；过期读被拒 1/1；普通冲突写仍拒 1/1；两轮收敛 1/1；默认表往返摘要复原、40/40、零线 0/0、九项全等。**代价**：① 协议不校验新值，核对器给错时接受；② 只测单键；③ 不判断权限；④ 纠正不改变键数、不产生语义。

**5. 替代方案**：① 读到不一致就自动纠正、不要凭据 → **否决**（等于后写者赢，R15/R18 已排除）；② 读到一致也写一遍 → **否决**（无意义写入，且会污染日志与摘要）；③ 用硬编码的“已知旧值”作凭据 → **否决**（A2 证明必须用读值；硬编码会在值已变时静默失败或误改）；④ 过期纠正自动重试直到成功 → **否决**（会覆盖更新的值，破坏先写者/最新值语义）；⑤ 让桥自己判断新值对不对 → **否决**（需要语义与权限，超出零语义口径）；⑥ 只测一步、不测收敛 → **否决**（会掩盖“每轮都重写”的退化）；⑦ 只测单键就宣称句级能力 → **否决**（句级双实体留给 R20）。

**6. 撤回路径**：删除 R19 runner／manifest／报告即可；桥模块未被修改，无需回退；文档侧删除 SPEC §21.28、账本 §66、U54、D-21，并把四处队首改回 R18 段。R16／R17 的复跑锚仍按各自提交有效；R18 在当前 HEAD 仍可复跑（R19 未改模块）。

**7. 授权来源**：用户 2026-11-09 `/goal continue`（继续推进唯一后继 R19；判据读数前冻结；0 训练／0 优化器步／0 Adam／0 confirm；关闭态九项全等；跑前估 CPU、超 200 先写台账）。

**8. 对已有结论的影响**：**R15／R16／R17／R18 的读数全部不变**（默认表 40/40、零线 0/0、九项全等；R18 的纠正原语未动，且本轮未改模块，R18 在当前 HEAD 仍可复跑）。**新增可信事实**：① 纠正原语可以闭环使用，凭据确实来自读值；② 已一致时不写，两轮内收敛；③ 过期读被整批拒绝，不会覆盖更新的值；④ 核对器给错时协议无法识别——这是已声明的边界。**新增限制**：① 不得说循环能判断纠正对错；② 只在单键上测；③ 不判断权限；④ 纠正不改变键数、不产生语义；⑤ 桥仍不得接进默认图。**下一步**：R20“句级双实体纠正闭环”闸——一句两实体，核对器按槽位给值，只纠正不一致的槽位；测单槽纠正不影响另一槽、两槽都错两步纠正、过期子集纠正被拒且已纠正槽不受影响、整句两实体读回正确、普通冲突写仍拒、关闭态九项全等。判据读数前冻结。

---

## D-20261109-22 R20 句级双实体纠正：单槽只动一槽、双槽同批、过期子集整批拒绝

**1. 编号／时间／类型**：D-20261109-22｜2026-11-09｜**判据冻结（读数前新增双实体算例）＋架构判断（批量原子性可作能力）**。不改地址层、不改排序、不改零线、**不改桥模块**；双实体算法只在 runner 内运行。六个算例与默认表双键往返在首跑前写入 manifest `definition_digest`。

**2. 决策内容**：

- **双实体读**：一句两键，两段读返回 object／place 两个值；槽位从值里读，键文本从记录里读。
- **单槽**：object `蓝色杯子`→`红色印章`，place 已对 ⇒ 只提交 object 一项，place 不动。
- **双槽**：object→`红色印章`、place→`书架` ⇒ 同一批提交两项，`corrected=2`。
- **过期子集**：先读 `(蓝, 木桌)`，他人把 place 改成 `书架`，再用旧对提交两项 ⇒ 整批 `rejected_batch_correction_mismatch`，object 仍 `蓝`、place 仍 `书架`。
- **已一致**：不提交、日志 0、摘要不变。
- **普通冲突写**：仍 `rejected_batch_address_already_bound`。
- **收敛**：两轮，第一轮 `corrected`、第二轮 `agreed`，日志 1。
- **默认表**：首条 describe 句两键同批改出再同批改回，摘要复原、R15 40/40、零线 0/0、九项全等。

**3. 触发原因与读数出处**：R19 闭环只在单键上；真实句子返回两个槽位，且核对器通常只指出其中一个或两个槽位错。R20 要回答“纠正能否按槽位组合，并且子集出错时是否整批原子”。读数：[报告](../../reports/taiji_bridge_pair_correction_20261109.json)、[manifest](../manifests/taiji_bridge_pair_correction_20261109.json)；成本 **1.844 CPU／2.616 wall**（估 30、上限 200，无需预算台账）。

**4. 代价**：0 训练／0 新参数／0 优化器步；未改模块。**读数**：单槽只动 object、place 未动、日志 1；双槽同批 `corrected=2`、日志 2；过期子集整批拒绝、有效项也不落盘、日志仍 1；已一致日志 0；普通冲突写仍拒；两轮收敛日志 1；默认表双键往返摘要复原、40/40、零线 0/0、九项全等。**代价**：① 单槽纠正只对两槽都能读出的句子成立；② 过期子集拒绝时有效项也不落盘（原子性）；③ 协议不校验新值；④ 不判权限；⑤ 纠正不改变键数、不产生语义。

**5. 替代方案**：① 每个槽位分两批纠正 → **否决**（失去原子性：过期子集会让有效项先落盘，R20/C 明确禁止）；② 过期子集只回滚出错项 → **否决**（`correct` 没有逐项回滚，只有整批；逐项回滚会引入新的部分应用语义）；③ 两槽一致也写一遍 → **否决**（污染日志与摘要）；④ 用硬编码的槽位值作凭据 → **否决**（读值才是凭据，R19/B 已证）；⑤ 只测单槽就宣称句级能力 → **否决**（双槽与过期子集是句级的关键）；⑥ 把双实体纠正写成新的 `write` 规则 → **否决**（写规则仍只有先写者保留）。

**6. 撤回路径**：删除 R20 runner／manifest／报告即可；桥模块未被修改、无需回退；文档侧删除 SPEC §21.29、账本 §67、U55、D-22，并把四处队首改回 R19 段。R18／R19 在当前 HEAD 仍可复跑（R20 未改模块）。

**7. 授权来源**：用户 2026-11-09 `/goal continue`（继续推进唯一后继 R20；判据读数前冻结；0 训练／0 优化器步／0 Adam／0 confirm；关闭态九项全等；跑前估 CPU、超 200 先写台账）。

**8. 对已有结论的影响**：**R15–R19 的读数全部不变**（默认表 40/40、零线 0/0、九项全等；`correct` 原语未动，R18／R19 在当前 HEAD 仍可复跑）。**新增可信事实**：① 纠正可以按槽位组合，单槽只动一槽；② 两槽可以在同一批纠正；③ 过期子集导致整批拒绝，有效项也不落盘——原子性可被当作能力引用；④ 两轮内收敛。**新增限制**：① 单槽纠正只对两槽都能读出的句子成立；② 原子性意味着“有效项也不落盘”，不得写成缺陷；③ 协议仍不校验新值；④ 不判权限、不记来源；⑤ 纠正不改变键数、不产生语义；⑥ 桥仍不得接进默认图。**下一步**：R21“纠正来源与授权”闸——给 `correct` 增加可选的 `authority`／`reason` 并写入审计日志；表可配置允许的来源集合（空集＝不设策略、保持现行为）；测无来源/未知来源在设策略时被拒、允许来源可纠正且日志带来源与理由、普通冲突写仍拒、默认表不设策略时往返与 40/40 不变、关闭态九项全等。判据读数前冻结。

---

## D-20261109-23 R21 纠正来源与授权：来源可记账、可选允许集合、CAS 仍是唯一凭据

**1. 编号／时间／类型**：D-20261109-23｜2026-11-09｜**判据冻结（读数前新增来源算例）＋机制新增（`authority`／`reason` 与可选允许集合）**。默认空策略保持 R18–R20 行为；非空策略只影响 `correct`，不影响 `write`。

**2. 决策内容**：

- `correct(items, *, authority="", reason="")`：来源与理由是**记账**，写入每条纠正日志。
- `BridgeTable(..., allowed_authorities=())`：空集＝不设策略；非空时缺来源 ⇒ `rejected_batch_missing_authority`，未列出来源 ⇒ `rejected_batch_unknown_authority`，均在计划任何一行之前整批拒绝。
- 来源**不替代** CAS：允许来源但旧值过期仍 `rejected_batch_correction_mismatch`。
- 普通 `write` 的冲突规则不变。
- 七个来源算例 + 默认表往返；三行复跑锚（R18/R19/R20）。

**3. 触发原因与读数出处**：R18–R20 的日志只有 `from`／`to`，没有“谁、为什么”，且任何调用方都能纠正；这使“可核验的纠正”无法追溯到来源。R21 加最小来源层，同时保持空策略的旧行为。读数：[报告](../../reports/taiji_bridge_authority_20261109.json)、[manifest](../manifests/taiji_bridge_authority_20261109.json)；成本 **1.766 CPU／2.654 wall**（估 30、上限 200，无需预算台账）。

**4. 代价**：0 训练／0 新参数／0 优化器步；模块新增两个可选参数、一个构造参数与日志字段。**读数**：空策略仍 `corrected`、来源为空；缺来源/未知来源整批拒绝、内容摘要不变、日志 0；允许来源 `corrected`、日志逐字记录 `verifier:v1`／`evidence:r21`；过期凭据仍拒；普通冲突写仍拒；两步链两条日志均带来源与理由；默认表往返摘要复原、40/40、零线 0/0、九项全等。**代价**：① 来源是自称，不是身份认证；② 允许集是策略，不是安全边界；③ 理由不校验真伪；④ 来源不校验新值；⑤ 纠正不改变键数、不产生语义。

**5. 替代方案**：① 把来源做成必填、空策略也必须给 → **否决**（会破坏 R18–R20 的调用与已冻结读数）；② 用来源替代 CAS → **否决**（来源不是“当前值”，会让盲写者绕过凭据）；③ 来源不匹配时只警告不拒绝 → **否决**（等于没有策略）；④ 把允许集写进 `write` → **否决**（写路径仍只有先写者保留）；⑤ 用签名/密钥做真认证 → **否决**（超出本轮零密钥、零外部状态的口径，且需要密钥管理）；⑥ 不记录理由 → **否决**（只记“谁”不记“为什么”，审计不完整）。

**6. 撤回路径**：`git revert` 本笔代码提交即可删除来源参数、允许集与日志字段；R21 报告与 manifest 随之作废；文档侧删除 SPEC §21.30、账本 §68、U56、D-23，并把四处队首改回 R20 段。R18–R20 的复跑锚仍按各自提交有效。

**7. 授权来源**：用户 2026-11-09 `/goal continue`（继续推进唯一后继 R21；判据读数前冻结；0 训练／0 优化器步／0 Adam／0 confirm；关闭态九项全等；跑前估 CPU、超 200 先写台账）。

**8. 对已有结论的影响**：**R18–R20 的读数全部不变**（空策略下 `correct` 行为逐项相同；R15 40/40、零线 0/0、九项全等）。**但 R18–R20 在当前 HEAD 不可跑**：它们 manifest 固定了旧模块 sha，R21 改了模块；按 `64a221f37`／`dcda39ca9`／`56c87bbd0` 可复跑，已逐行登记。**披露**：首跑报告尚无复跑锚（2.000 CPU／2.856 wall），随后只增补 `RERUN_ANCHORS` 输出并重冻重跑，七个来源算例与全部读数逐项相同（1.766／2.654）。**新增可信事实**：① 来源与理由可逐条记录；② 非空允许集能整批拒绝缺来源/未知来源；③ 来源不替代 CAS；④ 旧调用与空策略完全兼容。**新增限制**：① 来源是自称，不是认证；② 允许集不是安全边界；③ 理由不校验；④ 来源不校验新值；⑤ 桥仍不得接进默认图。**下一步**：R22“桥状态导出/导入”闸——把整张表（键、值、计数、纠正日志含来源/理由）导出为字节，导入新表后完整摘要一致、R15 40/40、零线 0/0；导入端允许来源策略由新表构造决定；损坏字节整批拒绝；关闭态九项全等。判据读数前冻结。

