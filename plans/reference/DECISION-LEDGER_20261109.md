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

