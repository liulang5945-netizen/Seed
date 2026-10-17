# EXP-N1-96 · v6 未见题集的一次性确认查询判读：范围准入首次成立、四门合取不成立、卡口点名两格 · 2026-10-17

owner 于 2026-10-17 弹窗裁 **甲＝签 `name_neighbor_distance=1` 并跑那一次确认查询**（登记＝08 ㊵-1042），同一弹窗裁 DEBT-G152 **以仪器为准、更正文句**。本件是那三步跑完之后的判读件：题面已按签署形状落盘、牺牲半范围检验**首次达线**、确认子集已**一次性消费**（`retry_allowed=false`），四门合取**不成立** ⇒ 按 PLAN-N1-93 §5 落**出口 ⑵**。

## 0. 一句话结论

**尺能在没见过的面上开出动态范围（7/9 枚端点 ≥ 门槛 6），能力也在一门上过了线（`advantage_terminal` 17/60 ≥ 12），但四门合取不成立 ⇒ N1 的 C3「未见确认可判」不成立**；卡口被这次消费买到了具体两格：**seed13 的检索塌方**（1/20）与**位置臂的弃权缺失**（`unknown` 类 0/15）。

## 1. 三步的实际读数（全部为当轮现取，命令与 rc 一并登记）

| 步 | 命令 | rc | 关键读数 | CPU／墙钟 |
| --- | --- | --- | --- | --- |
| 1 造面 | `build_taiji_n1_fresh_confirmation_v6.py --execute-signed` | 0 | `accepted=true`、`model_loads=0`、面 SHA `7dddf00f3f2e2438…66d49`；J93_8 `pairs=110, min_hamming=max_hamming=1`；J93_5 两半各 20、`overlap=[]`；J93_3 `oldest=5, newest=4, 形状上界=6, frequency_strategy_defined_items=0` | 内存造面，不占测量预算 |
| 2 范围检验（牺牲半） | `verify_taiji_n1_v6_range_check.py --readiness` → `--execute-approved` | 0 → 0 | readiness：九枚端点、`confirm_queried=false`；正式：`status=passed`、`items_queried=180`、`endpoints_with_triple_errors=7`／`min_usable_endpoints=6` ⇒ `range_check_passed=true` | CPU 98.2188 s ≤ 上限 180；墙钟 30.2 s |
| 3 确认查询（确认半） | `eval_taiji_n1_transition_factorial_confirmation_v6.py --readiness` → `--execute-approved` | 0 → 0 | readiness：`guards_total=149`、`range_check_passed=true`；正式：`status=published`、`confirmation_queries=180`、`judged_mismatch_count=0`、`real_change_count=0`、`sources_unchanged=true`、`parameter_updates=0`、`endpoint_neutrality` 全真 | CPU 151.7344 s；墙钟 42.7 s |

入库件：`plans/manifests/n1_fresh_confirmation_v6_20261017.json`、`reports/taiji_n1_v6_face_construction_checks_20261017.json`、`reports/taiji_n1_v6_range_check.json`、`reports/taiji_n1_v6_range_check.json` 的执行件 `reports/taiji_n1_transition_factorial_confirmation_v6.json` 与 attempt 件。

## 2. 动态范围：牺牲半逐枚读数（这条线只 gate 端点枚数，见 §6）

`addressable_errors`＝该端点在 15 道**可解码**题（牺牲半 20 题里 `unknown` 那 5 题不进分母）上的 native 错误数，门槛 ≥6：

| seed | position | terminal | cut |
| --- | --- | --- | --- |
| 11 | **5（不达标）** | 13 | 7 |
| 12 | **4（不达标）** | 13 | 6 |
| 13 | 12 | 13 | 13 |

`usable_by_arm`＝`{position:1, terminal:3, cut:3}`，合计 7 枚 ≥ 6 ⇒ **过线**。逐类平均（披露列，不参与判级）：`lookup 4.1111`（8 枚过 3）、`revision 2.8889`（5）、`transition 2.5556`（6）、`unknown 3.8889`（8）。与预飞预测的对照：预飞按 d=1 四个独立抽样给 7／7／7／8，真面落在 **7**；预飞里"position 臂常在底"这一形在真面复现（seed11/12 的 position 正是两枚不达标端点）。

## 3. 四门判级（判级权在执行器的 `evaluate_gates`，本件不手算；线值为 PLAN-N1-94 冻的原绝对值）

| 门 | 线 | 读数 | 判定 |
| --- | --- | --- | --- |
| `each_position_seed` | 每 seed ≥10／20 | 11:12、12:12、**13:1** | **假** |
| `each_position_kind` | 每 (seed,类) ≥2／5 | 11: lookup5／revision5／transition2／**unknown0**；12: lookup2／revision5／transition5／**unknown0**；13: **lookup0／revision0／transition1／unknown0** | **假** |
| `advantage_terminal` | ≥12／60 | **17**（position 25 − terminal 8） | **真** |
| `advantage_cut` | ≥12／60 | **0**（position 25 − cut 25） | **假** |

`all_four_gates=false`、`behavioral_pass=false`、`can_promote=false`、`retry_allowed=false`。三臂 free-exact 合计：position 25（12/12/1）、terminal 8（2/3/3）、cut 25（10/12/3）。**注意 `advantage_cut` 的 0 与 v5 那次的 0 不同形**：v5 是两臂逐字输出相同，v6 是**总数相等而逐格不等**（13 号上 cut 3 对 position 1、11 号上 position 12 对 cut 10），所以这次不能写"两条臂是同一件事"，只能写"求和后相抵"。

## 4. 卡口点名（这次消费买到的东西，全部来自同一枚件的 `rows`，零额外查询）

- **J-96-1（seed13 塌方不是解码形状）**：seed13 的 position 臂 20 题全部 `stop_reason=boundary`、生成长度 5～7、均值 6.55（seed11 同臂均值 6.7）⇒ 长度与停止原因正常，错的是内容。**跨臂同错**：seed11 有 4/20 题三臂全错、seed12 有 4/20、**seed13 有 14/20**；三个 seed 里"三臂同对"的题数都是 **0**。⇒ seed13 的缺口在那枚检查点本身，不在臂。
- **J-96-2（位置臂从不弃权）**：`unknown` 类 5 题 × 3 seed，position 臂输出 `unknown` 的次数 **0/15**（逐题给出 6 字符材质词，例 `vvqpii`、`jrrhxi`）；terminal 臂 4/15、cut 臂 2/15。⇒ 位置臂在"该弃权"的题上**总是检索出一个词**：`each_position_kind` 这条线对 seed11/12 之所以红，唯一破格就是 `unknown` 那一格（其余三类 11 号 5/5/2、12 号 2/5/5 都 ≥2）。这条读数同时否证"弃权类是题面缺陷"——同一张面上另两条臂能答对该类。
- **J-96-3（`transition` 与 `lookup` 的不对称）**：12 号 position 臂 `lookup` 恰为 2/5（压线）、`transition` 5/5，而 11 号相反（`lookup` 5/5、`transition` 2/5）⇒ 同一个 seed 换一次绑定就换一类掉下来，与 J93_8 施加的"只差 1 个字符"的可混淆度方向一致；这不改变判级，只作为下一格的取题依据。

## 5. 现在可说什么、永久不许写什么

可说：①v6 面通过动态范围准入，**这是 N 主线第一次在未见面上证明尺有范围**；②`advantage_terminal` 在未见面上过线（17/60，线 12），即位置臂对末端臂的领先在这一格成立；③四门合取不成立，落 PLAN-N1-93 §5 出口 ⑵，N1 的 C3 判定**未达成**；④卡口点名到 seed13 与弃权类两格（J-96-1／J-96-2）。

永久不许写（本件的收回句）：
- **"位置读出臂在未见集上领先"** ⇒ 四门合取为假，且 `advantage_cut` 为 0；单门过线不等于合取过线。
- **"C3 达成／N1 可晋升"** ⇒ `can_promote` 由执行器钉为 `false`，PLAN-N-01 的 C3 行不消费本件。
- **"换个 seed 或重跑一次确认半"** ⇒ `retry_allowed=false`，v5 与 v6 两张面都已消费；任何再跑只能标"已知面上的机制探针"，不得冠"未见"。
- **"弃权类是尺的问题"** ⇒ 同面上 terminal 4/15、cut 2/15，题面可答。
- **"`unseen_claim_allowed=true` 是仪器判出来的"** ⇒ 该列在执行器里是常量字面量 `True`（连 `status=interrupted` 那支也会出版它），今天的真值来自旁证列（`status=published`、180 行、`endpoint_neutrality` 全真、`complete_checks` 全真），不是这一列自己算的 ⇒ 缺陷登记为 **DEBT-G155**，修法见 §7。

## 6. 口径来源（防止后来者把两条线读混）

DEBT-G152 已由 owner 裁"以仪器为准"：唯一 gate 的线＝"≥6/9 枚端点在三类合计错误 ≥6/15"；"三臂每类平均 ≥3/5"是披露列（件里点名 `reported_not_enforced`）。四门的线值由 PLAN-N1-94 冻结的原绝对值给出，本件未改任何线值、未改分母、未改聚合层级。

## 7. 本件顺手抓到的一条仪器缺陷（DEBT-G155）

`eval_taiji_n1_transition_factorial_confirmation_v6.py:327` 的 `"unseen_claim_allowed": True` 是**恒真字面量**，而 `status="interrupted"` 那支（`:305-307`）同样写报告 ⇒ 一次被中断、行数不满 180 的跑会出版一枚"自称允许未见声称"的件。修法（零重跑、不改判据）：把该列改为由 `status == "published"` 与 `complete_checks` 全真**推导**，并在合同测里补一条负对照"中断跑必须让它为假"。今天的读数不受影响：`complete_checks` 四项全真、`confirmation_queries=180=len(rows)`、`endpoint_neutrality` 全真。**〔㊵-1044 已落地〕**修法照上句执行：该列改由纯函数 `unseen_claim(status, rows, queries, restore_checks)` 推导，测册 `test_n1_run94_v6_confirmation_contract.py` 7→9 passed，含六支负对照与一支"把新算法打在已出版真件上"的验收；过程中我自己的第一版把总行数写成 20×3＝60（真面 20×3 seed×3 臂＝180），被新测当场判红——这条也算本件登记的一部分（同 08 ㊵-1044）。

## 8. 下一格（队首，含各自的前置）

v6 消费后，N1 侧在**不新建面**的前提下已经没有可发射的一次性查询。三个候选，按可声称上限排列（不是按成本）：

- **甲（上限最高，需 owner 批产品码或训练）**：攻 J-96-2 那格——位置臂从不弃权是**行为缺口**，不是测量缺口；它是四门里唯一"有明确成因、且同面另有两臂能做对"的门。形状＝先做零算力的弃权行为普查（用仓内既有 88/89 批次产物逐题数"该弃权时给了什么"），再决定是否动训练配方。
- **乙（需一次性签署）**：v7 面（第三次一次性消费）。今天已知的价值只有一条：d=1 的尺能开线，所以 v7 买到的仍是"上限读数"而不是新通道；除非同时改判据（那是另一格裁）。
- **丙（自办、零算力）**：C4 侧的 PLAN-N-02 已经给出三个点名字段缺口，B 档需改权重批文（owner 已裁"先停下"）；A 档剩余＝DEBT-G155 这类仪器修口，不产生能力读数。

本件不自选：甲要产品码或训练批文、乙要 §1 式签署、丙已被停。

相关：EXP-N1-95（预飞与 d=1 那条维）、PLAN-N1-93（构造与范围准入）、PLAN-N1-94（一次性确认查询与四门）、08 ㊵-1042／㊵-1043。
