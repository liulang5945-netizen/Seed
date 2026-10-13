# N1 阶段 1（S1 再激活重放）要不要批——一页纸（十一裁 ⑷ 的交付物；决定权在 owner）

现取时刻＝2026-10-13T16:36Z，HEAD `435d7877e`。**这份件不给建议顺序之外的任何主张**：每条都带 `文件:行`，我核对过原文；件里没有的数（例如机时）我不编。

## 一、"阶段 1"在现行文档里指什么（唯一权威口径）

- `plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md:218`：N1 线"**预注册与判读已多轮闭环，当前待 owner＝S1 阶段1（再激活重放＋训练）机时批准**"。
- `plans/reference/PLAN-N1-01_s1_readout_prereg_20261007.md:15`＝H-S1b 的定义：召回驱动 fabric 沿存储轨迹重放，改动面 `taiji/fabric.py` 与生成侧注入点 `taiji/model.py:2389-2427`。
- 同件 `:43`（§3.2 标题）："阶段1（H-S1b，训练与三器官验收）——**本件只登记形状，数值随实施附件冻结**"，`:45` 补一句"**不在本件预先定价**"。⇒ **文档里没有机时数字，谁报价都是编的**；要批就得先出一份实施附件把那六项形状的数值冻住（`:47-52`）。
- 并存编号系不冲突：09 §2 的 N1 推进步骤 1–4（`:109-113`）不带"阶段"号；`PLAN-N1-03_forced_predecessor_prereg_20261008.md:44-45` 用的是 A／B 档。03 与 09 现行活的是 PLAN-N1-01 的 阶段0／阶段1（阶段0 已跑完并判读）。

## 二、它现在挡住什么

- `plans/reference/PLAN-N4-13_trajectory_state_prereg_20261013.md:15-16`："一条不属于本件却挡着本件的话……N1 阶段 1（改产品读取＋训练机时）十裁里**没有**批准。所以本件冻的是'什么形状才算轨迹态落地'"。
- 同件 `:22`（判据 `J-N4i-1`）："『替代静态混合』这句在 N1／S1 通路落地前**不许说**"。
- 另有一处依赖：`plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md:3001` 把"等 N1 阶段 1"列为 N5 批次 C 喂入的选项丙。

## 三、N1 侧已经落地到哪一步（决定"批了之后从哪接"）

| 判据 | 冻结 | 仪器 | 读数件 | 判定 |
|---|---|---|---|---|
| `J-N1c-1`／`J-N1c-2`／`J-N1c-4` | `PLAN-N1-03_chain_and_roles_prereg_20261011.md:48,50,52` | `scripts/training/audit_taiji_n1_03_role_table_face.py` | `reports/taiji_n1_03_role_table_face_20261011.json`（`chain_complete`、`table_registered` 20/20、`recovery_claims_verifiable`） | **三条出值** |
| `J-N1c-3` | 同件 `:58` | `audit_taiji_answer_source_provenance.py`＋`read_taiji_envelope_recipe_face.py:202` | 20261011 件 `carrier_count 0`；20261012 件 `carrier_count 1`；`reports/taiji_envelope_recipe_face_20261012.json` `verdict=no_recorded_recipe_in_sample` | **无判定**（`05_TECH_DEBT_REGISTER.md:1200` DEBT-G112 明写"既不成立也不被否证"） |
| `J-N1d`（强制前驱，A／B 档） | `PLAN-N1-03_forced_predecessor_prereg_20261008.md:35-37` | 无（`scripts/` 零命中） | 无 | **只有文字** |

⇒ §6 的第一出口要求四条 `J-N1c` 全过，现在**不成立**（差 `J-N1c-3` 一枚判定）。

## 四、三个可批的形状（差别在"花不花训练机时"与"改不改产品码"）

- **甲＝只批 H-S1a（零训练的前测）**：`PLAN-N1-01:14` 定义它把寻址支持集从"整条事件所有位置"改成"刚发射字节位置的后继位置集"，改动面＝`taiji/copy_circuit.py:277-290` 的 `_successor_bonus` 升成硬位置掩码（`:292-308` softmax 前置 −∞）＋`_pool_into`（`:310-337`）只在候选后继内求字节质量。`:16` 写清了为什么先跑它：**它是 H-S1b 的最小可证伪前测——若硬序贯读取都打不动 never-LF 拖写行数，病就不在"读取丢序"，b 的训练投资应推迟**。代价＝产品码改动（`taiji/`）＋一趟评测；**不花训练机时**。
- **乙＝直接批 阶段1＝H-S1b**：`taiji/fabric.py`＋`taiji/model.py:2389-2427` 改动＋训练机时＋三器官验收。前置不是"我估个数"，而是先出 PLAN-N1-01 的实施附件把 §3.2 那六项形状数值冻住（`:47-52` 已列形状，含三器官对第 0 点的对照读数量级）。**在附件出来之前，本选项的代价无法回答**。
- **丙＝都不批，先补 N1 侧缺的判定面**：把 `J-N1c-3` 的载体面补齐、给 `J-N1d` 造仪器。**但这条不能报成"零算力"**：`05_TECH_DEBT_REGISTER.md:1150`（DEBT-G62）已核实"PLAN-N1-03 §4 把 A 档报成零算力可自办"这句**不成立**——`taiji/model.py:3190` 的 `generate(...)` 13 个参数里没有任何"喂入指定字节序列"的口子，而探针的发射面由 `runtime.chat()` 产生（`probe:1477`），探针不拥有发射环 ⇒ 补 `J-N1d` 实际要动产品入口，不是纯仪器活。

## 五、排期硬约束（不管选哪个都撞得上）

1. 窗口约束 `G-N5v-1`：两枚重复臂之间不许提交 `taiji/`／`seed/`／`scripts/training/readout_retrain_spec.py`／跑器本体。治疗臂现取末行 `symbols_consumed=3,000,000／8,900,000`（`written_at_utc=2026-10-13T16:04:25Z`，约 290 符号每秒 ⇒ 余约 5.6 小时），对照臂尚未起跑。⇒ **甲／乙的真实开工最早在对照臂退出之后**。
2. 同一窗口后面已经排了四件已批的产品码（`PLAN-N5-24` 奖励回读、`PLAN-N5-25` 受体通道重映射、`PLAN-N2-08` 显式来源、`PLAN-N4-13` 前半句）。甲/乙若现在批，等于给那条队列再加一位；我不擅自给它排序，位置由你定。
3. 一处过期声称（顺手提，不等这批决定）：`09_NEXT_MAINLINE_PLAN.md:62` 把发表门引成 `pending_not_claimed`（2 satisfied＋1 not_established，对 20261011 那件），而 `reports/taiji_n1_03_publication_claim_gate_20261012.json` 现读已是 `publishable: true`（3 satisfied）。**这句过期的是我维护的状态列**，我会在下一格就地打日期戳更正（原句不删）。

## 六、这份件不声称的东西

- 不声称阶段 1 已被任何文档判定为"该开工"或"不该开工"——`03:3`、`09:218/225`、`PLAN-N4-13:16`、`N_MAINLINE_COMPLETION_AUDIT_20261012.md:25` 四处一致停在"未批、未跑"。
- 不声称甲比乙省多少——没有数字，`PLAN-N1-01:45` 明写不预先定价。
- 不声称 `J-N1c-3` 缺的是"证据"还是"判据"：DEBT-G112 的现读是**既不成立也不被否证**，这两件事不能互换。
