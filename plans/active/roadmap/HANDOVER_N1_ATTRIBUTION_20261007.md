# 交接：N1 基底归因轮（PLAN-N1-02）——2026-10-07 深夜，额度中断前快照

> **✅ 已收官（2026-10-07 下一会话）**：本文 §2 待办全部结清——判读报告已入库＝`plans/reference/PLAN-N1-02_ATTRIBUTION_ADJUDICATION_20261007.md`，卫星同步（03／reference README／PLAN_INDEX §1b／09 §3.2）与门（doc-sync 43/0/0＋hygiene 18/0/0）同批完成，台账＝**08 ㊵-479**（本文按㊵-478⑤ 保留为过程快照，不再作为活动断点；现行状态一律以 03 顶部条与 09 §3.2 为准）。**下一件事**：PLAN-N1-01（S1 实施预注册）起草＋回 owner。

> **给下一个会话**：本轮（N1 归因）做到"预注册已冻＋仪器已就绪＋判读面待复跑"处中断；**没有任何已判读数**（唯一一次判读尝试被 fail-closed 守卫正确拒绝）。本文是断点快照，复跑收官后可蒸馏进 08 台账并删除本文。

## 0. 一分钟状态

- **上游已收官（已提交）**：㊵-476（c52b1c0a）＝S5 预注册；owner 弹窗四裁（S5 批准开跑／C6 按推荐组合批／N3 甲乙并行／挂账不处理）；㊵-477（66554649）＝S5 实施＋三支面＋判读（**J1 不成立**：never-LF 拖写行 209＞118，P1 降级为部分成立，转 §4 基底归因）。判读报告＝`plans/reference/PLAN-N1-00_S5_ADJUDICATION_20261007.md`。
- **本轮（PLAN-N1-02，提交 099e3699＋fb619a58）**：归因预注册已冻＋v43 仪器就绪＋判读面已跑完＋**冻结判读已出**。
- **判读结果（verdict 件在库，行为复现基线 247/236/41＝G1′ 过、1:1 全量 71,937 行对上）**：主群（never-LF 拖写代，48,896 可判步）F1 逐步跟随率 **0.549**（随机基线 0.060）、F2 质量命中率 **0.940**、F2−F1=**0.390**、F3 末字节发射率 **0.0021**、F4 gate 中位 **53.2**、F5 后继权重中位 **0.237**。**按 §3 冻结规则判级＝H-A 非主因（not_primary，第二支 F1≥F1c+0.20）**——序贯信号在场并被跟随，非"被淹没"。**归因输入（如实登记，不改判级）**：39% 的步被质量 winners 劫走＋逐步损耗复合 ⇒ 末字节到达率仅 0.2% ⇒ 轨迹统计性死亡 ⇒ S5 终点检测器无火可点；stopper 群 8,705 步全部 event_absent（自停代跑在空库上）。S1 设计杠杆读数＝逐步位置保真（55%/步 复合到 0.2% 端到端）。

## 1. 本轮落盘物（未提交，随本文同一提交入库）

| 文件 | 状态 |
|---|---|
| `plans/reference/PLAN-N1-02_attribution_prereg_20261007.md` | 预注册（判据先冻）：§1 完整计算图（11 环节带锚点）；§2 参数级预读（**gate_bias=+2.5 饱和、copy_induce_bias=+2.5 饱和、gate 范数≈23** ⇒ 门大开、诱导在场，假设收敛到池化聚合层）；§3 假设 H-A/H-B/H-C/H-D＋冻结读数 F1–F5＋三带判别规则；§4 面与守卫 |
| `scripts/training/probe_taiji_a30_stop_failure.py` | **v43**：新旗标 `--trajectory-following`（只读记录器：帧遍历判环内、与环内 observe 严格 1:1；行字段 prev/gate/top_pos/top_pos_byte/succ_byte/succ_w/succ_mult/last_byte/ev_top/ev_top_mass/n＋回填 target_kind/emitted_next/follow/mass_hit；逐轮挂 `surface_checks[].trajectory_follow_v43`）；与 S5 旗标互斥；无旗标 ⇒ 与 v42 逐位不变 |
| `scripts/training/count_taiji_n1_trajectory_following.py` | 判读脚本：分群（never-LF eaters 主群／all eaters／stoppers）→ F1–F5 → 套 §3 三带规则；fail-closed（面指纹、行为复现基线 247/236/41、1:1 总数、轮数步数交叉核对，任一不符 rc=2 拒判）——拒绝路径已用 1 件烟测验证 |
| `reports/taiji_a30_stop_failure_self_v43_trajectory_follow_96_20261007.json`（未跟踪） | **缺陷件**（别名 bug 时代的 45MB 旧件）——复跑会原地覆盖，不用管它 |

## 2. 接手者立即要做的事（按序）

1. ~~复跑判读面~~ ~~判读~~ ⇒ **已完成**（面件 rc=0；verdict＝`reports/taiji_n1_trajectory_following_verdict_20261007.json`：status ok、HA_verdict_on_primary_group＝**not_primary**；判读脚本在出数后补修一处——stopper 群整体 event_absent 属数据性质，该群出 null 读数不拒判，仅主群为空才拒）。
2. **判读报告**入库（仿 `PLAN-N1-00_S5_ADJUDICATION_20261007.md` 形状，命名 `PLAN-N1-02_ATTRIBUTION_ADJUDICATION_<日期>.md`）：守卫判读（G1′ 已过）＋F 读数表（§0 有数）＋H-A 判级＝not_primary＋归因输入（39% 步被质量劫走、逐步损耗复合 ⇒ F3=0.2%、stopper 群空库现象）＋对 PLAN-N1-01（S1）的设计约束输出（逐步位置保真为杠杆；冻结规则的字面路由＝解码/训练层归因另议——两条都如实写，判级不改）。
4. **卫星同步**：03 执行日志一条；`plans/reference/README.md` N 系列入列；`plans/active/PLAN_INDEX.md` §1b 加 PLAN-N1-02 行；08 台账 **㊵-478**（先 grep 确认空闲；本轮"预注册＋仪器＋两次仪器返工＋复跑判读"可并入一条，或复跑后拆两条，行首标号纪律见 08 头部横幅）。
5. **门**：`cd taiji-harness && corepack pnpm run doc-sync`（43/0/0）＋ `corepack pnpm run hygiene`（18/0/0）；提交。
6. **H-A 判级后的路由**（预注册 §3 已冻）：confirmed ⇒ PLAN-N1-01（S1）设计约束＝位置级/序贯读取（不做字节质量池化）；not_primary ⇒ 转解码/训练层归因；middle_band ⇒ 第二判别实验另立预注册。

## 3. 本轮踩过的坑（接手者免踩）

1. **列表别名 bug（本轮最大坑）**：`surface_checks[-1]["trajectory_follow_v43"] = traj_turn` 原来存共享引用，下一轮 `traj_turn.clear()` 连带清空已挂轮的行 ⇒ 三条 check 全指向最后一轮（缺陷件总数 73728＝96×3×256 精确暴露）。**已修**（挂 `list(traj_turn)` 副本）。判别特征：diff 全集中在 stopper 代、且 eaters 代 diff=0。
2. **`addressing()` 不回传 weights**——只有 `scores`；须 `scores.softmax(dim=0)` 重算（与 `_position_weights` 同式同源）。
3. **探针不 import torch**——用张量方法（`.softmax`），别引 NameError。
4. **store 空的生成步**：`addressing()` 返回 None ⇒ 行记 `event_absent: true`，回填时 follow/mass_hit 置 **null**（不冒充 0）。
5. **MSYS 路径转换只对 argv 生效**：`python -c "open('/tmp/x')"` 里的 /tmp 不转换（会落到 `E:\tmp`）；要 `python - 脚本参数 <<EOF` 或把路径放 argv。
6. **pnpm 不在 PATH**：门用 `corepack pnpm run doc-sync / hygiene`（taiji-harness 目录下跑）。
7. 提交时 Mimosa 钩子报 `scanner_enobufs`（按兼容放行）——如实登记，不宣称项目安全状态。
8. 冒烟可用 `--limit 1`；开/关旗标各跑一支、`compare_taiji_a30_report_identity.py --subtree per_item` 应 identical=true（记录器行为中性）；schema_diff＝新增披露字段属预期。

## 4. 本轮之后的队列（N 主线）

- **N2（C6 已批，按推荐组合）**：WorkbenchCapabilityAdapter（kind 白名单 a 案：`taiji/evolution_experience.py:245-251` 行内字面集加 `workbench_artifact`，EVOLUTION_CONTRACT_VERSION 维持 1）＋②b no_prose 渲染＋④生产者挂 `sleep_pass` project 段＋⑤消费者同批（折进 `data/consolidated/corpus-*.jsonl` 并进 `spec.datasets`）＋⑥P4 只读面板行；Z1 前置否证门结论在库（16 条物料、CJK 0.503）。
- **N3（甲乙并行已批）**：乙＝训练器接 bridge+trigger 的预注册（**压强阈先在线面冻结**：accuracy 0.59 ⇒ 残差阈 ≥0.4，生长节奏以面为准）；甲＝scaling 探针预注册（参数×10 单点；算力预算未单独批过，开跑前跟 owner 确认）。
- 主线外挂账（1.9GB／发布面／icon.ico）owner 已裁本轮不处理。

## 5. 不变项

M6 收官、M7-CI 绿（每次 push 复验）、M8 挂起；N1–N6 排序不变；台账行序以行首标号为准（㊵-419⑥）；门 doc-sync 43 叶＋hygiene 18 叶提交前实跑。
