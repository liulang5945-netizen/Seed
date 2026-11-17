# 22.5.U.1r2 设计卡：logit 尺度修正＋训练标签注意力监督（单位核心 B 修订 2，最后一次）

> 卡号：`22.5.U.1r2`｜日期：2026-11-17（设计起于 2026-11-16）｜状态：**设计＋定价冻结完成，smoke 63/63、契约测试 22/22、库外输入件实测可重建，停在训练授权节点**。
> 上级：[22.5.U 总卡](DESIGN-TAIJI_U2_UNIT_CORE_B_QUALIFICATION_20261116.md)｜方向依据 D-20261116-24（owner 裁决＝乙）｜父卡 [U.1r1 结果](RESULT-TAIJI_U2_U1R1_POSITION_20261116.md)（D-20261116-23）。
> 契约：[taiji_u2_u1r2_scale_supervision_20261116.json](../manifests/taiji_u2_u1r2_scale_supervision_20261116.json)（criteria_sha256 `c8f2d98b8a539013aa465a09978a7325e95f3a94ad3f4369ad6a7fc90905eb3f`，status **frozen**，training_authority **null**，revision **2/2**、剩 0）。
> owner 裁决原文：「2026-11-16 乙：logit 尺度修正＋训练标签注意力监督（选甲/乙＝先做设计与定价，零训练；跑之前再按实价请一次授权）」。本卡全程**零项目训练**：smoke 0 优化器步，定价与剂量共 210 步全部花在**用后即弃**的模型上（`models_discarded: true`），封存分区读取 0。

## 1 单问题与修订依据

r1 判 `read_prerequisite_failed`，定位（D-20261116-23）：**注意力从未聚焦**——训练后熵达 ln16 上限的 0.999266–0.99999、有效列数最低 15.967507/16、logit 极差仅 0.025293–0.203172 nats；‖W_q‖ 相对变化绝对值最大 0.109499（一 seed 缩小）；W_q 的任务信用只有 `position_head` 的 1/228.4103–1/11.0712。强制 ‖W_q‖ ×171.9652–207.6915 后极差 2.70737–5.531337 nats、有效列数 9.762043–13.29281 ⇒ **列 key 可区分、仪器有量程**，缺口在尺度与信用，不在编码（96/96 忠实）。均匀注意力下 `attended` 就是列算术平均、`position_head` 无 bias ⇒ 与 U.1 读**同一个列均值统计量**（代数事实）。

r1 明写的「值侧可解码性未测」已补测（D-20261116-24，零模型训练、仪器 33/33、8.1094 CPU）：出生 fit max **0.625**（达 0.80 门 0/3）对 r1 训练后 **0.90625／0.71875／0.90625**（双门 2/3），几何正对照 **1.0**、输入侧参照 **0.84375**、置换标签 max **0.0625**、不补丁注意力 max **0.375／0.28125** ⇒ **乙的前提未推翻**，但新增一条设计约束：**出生值侧低于门 ⇒ r2 的 384 更新必须同时长出聚焦与值侧可解码性**。

**修订 2 的结构变量只有一个标量**（见 §2），训练侧另加两件事：W_q 自己的学习率组、以及只碰训练标签的注意力监督。**编码合同 `unit-input-column-color-v1`（digest `00319fff32cf1ac14a89c6999c8ea4fe63f0ea1b26924eb6e73baa45877c0a55`）冻结不变**；数据根与 `data_digest` `40813b4c…` 与 r1 逐字相同；门沿用 r1（fit≥0.80／train≥0.90／distinct≥3／双控制 agreement<0.5／四干预／中性出生／独立恢复）。

## 2 机制：一个标量，整条链委托给父卡

新增 [taiji/u2_unit_temperature_readout.py](../../taiji/u2_unit_temperature_readout.py) 的 `TemperaturePositionGraph`，继承 r1 的 `PositionRouteBGraph`，**前向整条链 `super().forward()` 委托**，唯一差别是 `attn_scale` 在前向期间是可微张量而不是常数 float：

- `attn_scale = (1/√hidden_dim) · exp(log_tau)`，`hidden_dim=8`⇒`1/√H = 0.35355339059327373`；`log_tau` 是 `nn.Parameter`，**numel 1、初值 0.0**。
- `log_tau = 0` 时 `exp(0)=1`⇒**与父卡逐位相同**（不是近似）。route 由 `B_position` 改为 `B_position_temp`，所以 r1 的胶囊会被 `apply_capsule` 响亮拒绝（`capsule_route_mismatch`），不会静默错载。
- 前向用 `try/finally` 把 `attn_scale` 复位成 float：**`with_neutral_birth` 走 `copy.deepcopy(self)`，非叶张量属性不可 deepcopy**（这是本轮实测撞到的 `RuntimeError`，已加回归测试 `test_module_survives_deepcopy_after_forward`）。

**为什么尺度住在 W_q 外面**（被测量否证的替代形式见 §4）：logits 对 W_q **恰好线性**，所以「能不能聚焦」是一个零优化器步就能解的线性最小二乘问题；而 Adam 的每坐标位移由 lr 定、不由梯度幅度定，把尺度塞进 W_q 范数会同时（a）让随机出生方向饱和 softmax、（b）把 query 方向冻住（384 步 ×lr 1e-3 只能移动范数的 0.05–0.11%）。一个标量把「幅度」与「方向」拆开：幅度交给 `log_tau`，方向留给 W_q 继续可训。

**逐位锚点（smoke 实测，3/3 seed）**：`readout`／`attention`／`attended`／`next_state_z`／`next_state_a` 五项 `torch.equal` 全真，`max_abs = 0.0`、`atol = 0.0`；`state_dict` 只多出 `{log_tau}` 一枚；参数总数比父卡多 1。

## 3 训练、监督与出口

**三个参数组**（`make_optimizer`，契约测试断言分组覆盖全部参数、无重复、逐名 lr 正确）：

| 组 | 参数 | lr | 依据 |
|---|---|---|---|
| 值侧 | 除 W_q／log_tau 外全部（units.r/p/c/η/b/w、input_projection、input_encoders、state_mix、position_head） | 0.001（**不变**） | 值侧表示只由任务损失训练 |
| query | `W_q` | 0.002（multiplier **2.0**） | 实测旋转剂量不足，见 §5 |
| 温度 | `log_tau` | 逐 seed **0.041420056／0.043513149／0.039739542** | `2·ln(scale_target)/384`，`dose_margin=2.0` |

`scale_target_by_seed` = **2843.101383／4249.363903／2059.033836**，由最小二乘解的 ‖W_q‖ 除以出生 ‖W_q‖ 得到（4026.1787976369974/1.4161221338787644 等，机器恒等式已复验）；`log_tau_needed = ln(scale_target)` = **7.952650771584998／8.354524580834456／7.629992140112775**。

**监督**：对 attention 向量的**软目标交叉熵**，目标＝查询实体的逐列颜色质量归一化（`oracle_soft`），取自**冻结编码自己的 rgb 通道**。披露三件：① 它**比任务标签细**（4 类 → 16 列），是一条训练信号，**不是给单位预先贴标签**（`unit_prelabeling: none`）；② 只施加在**有标签的 train 场景**（48 条；另 16 条无标签 train 不监督），**fit／check 一律不做监督**（机检：监督 id ⊆ train、与 fit 不相交）；③ 权重 1.0、`log_clamp 1e-12`。

**信用限制（`credit_restriction: ["W_q","log_tau"]`）**：每步先对任务 CE `backward(retain_graph=True)` 并克隆梯度，`zero_grad` 后再对聚焦损失 backward，然后重组——**非 query 参数只拿任务梯度，W_q／log_tau 拿任务＋1.0×聚焦**。机检 `combined_matches_task_plus_restricted_focus` 三 seed 全真、`max_abs = 0.0`；聚焦损失本来会给值侧送去 **0.7172876447478589／0.30174648742612137／1.2674022540378136** 的绝对梯度总量（`focus_credit_off_query_total`），限制后这些参数与纯任务梯度逐位相同；`focus_parameters_not_in_task = []`、`parameters_in_combined = 17`。**为什么必须限制**：在选定尺度上，聚焦损失对值侧参数的信用是任务信用的 **70.9922–1375.4569×**（`units.p` 104.8322/70.9922/94.5029、`input_projection` 206.5007/220.9678/284.3565、`input_encoders.rgb.weight` 315.1169/472.8582/1375.4569；契约 reason 字段写作约 71–1375×，是同一组读数的舍入）——不限制就等于让聚焦损失重写值侧表示，而值侧正是 §1 要求同时长出来的那一半。

**聚焦门（新增，尺子来自 `probe_u1r2_scale_focus_20261116.py`）**：面＝**fit**（32 条，其中有标签 24 条），`answer_mass_mean ≥ 0.5` 且 `effective_columns_mean ≤ 8.0`。**尺子有量程**（smoke 实测，同一 fit 面）：出生 answer_mass **0.135536／0.135735／0.135209**、有效列数 **15.999929／15.999939／15.99878**⇒`focus_met` 三 seed 全 **false**；强制 `oracle_soft` 后 answer_mass **1.0**、有效列数 **4.961452**⇒全 **true**。标定报告另在 train 面验了六支动态范围（birth_c1／r1_trained_c1／birth_chosen 判 false，oracle_soft／oracle_onehot 判 true，wrong_column 判 false）全真。

**五个预注册出口（互斥、按序判定，写在契约 `criteria.exits`）**：

1. `budget_stop`——超实测帽即停，按预算停止结案；
2. `engineering_failure`——完成臂数少于 seed 数；
3. `focus_not_met`——**聚焦仍未达标**（尺度／监督没把注意力教会）；
4. `focus_met_value_failed`——**聚焦达标而值不达标**（正是 C.2 先例与 §1 值侧约束要区分的那一支）；
5. `read_prerequisite_passed`——聚焦与值侧同向达门。

出口纪律：判读只认冻结判据，读平不换次要指标；正结果**只签读前提**（U.2 纠正／U.3 正式 A/B 另案）。

## 4 设计探针：可行性与被否证的替代形式

**可行性探针** [probe_u1r2_focus_feasibility_20261116.py](../../scripts/training/probe_u1r2_focus_feasibility_20261116.py)／[报告](../../reports/taiji_u2_u1r2_focus_feasibility_20261116.json)（schema `taiji-u2-u1r2-focus-feasibility-v3`，**0 优化器步**、3.0469 CPU／3.047 wall、只读 train＋fit、封存 0）：logits 对 W_q 线性 ⇒ 建 `A[t,(d,h)] = attn_scale·keys[t,h]·question[d]`，每场景另加一个自由 logit 偏移（softmax 平移不变），`torch.linalg.lstsq(driver="gelsd")`。系统 768 方程／112 未知（64 W_q＋48 偏移）、rank 66；列 key 与模块 logits 的锚点 `max_abs = 3.9968028886505635e-15`。

- **拟合 train 目标后在 fit 上评分**：answer_mass **0.864842／0.776586／0.678967**、有效列数 **3.404741／2.717741／3.258413**、`focus_met` **3/3 真**（train 面 0.882831／0.841122／0.865177）⇒ **存在能聚焦且泛化到未见场景的 W_q**。
- **错排目标对照**：fit answer_mass 掉到 **0.155587／0.082774／0.050431** ⇒ 聚焦不是「随便一个解都行」。
- **范数扫描**：最小二乘解的范数／标定范数＝**6.602804020572852／8.283828721453872／24.501515236388563**（聚焦所需的 query 范数）；扫描出的最小满足 train 聚焦因子 **0.5×3**，即所选范数／标定范数 **3.301402010286426／4.141914360726936／12.250757618194282**（恰为前者一半），`calibrated_norm_already_sufficient` **false×3** ⇒ 标定尺度本身不够聚焦。
- **仪器 10/12，两支红是设计要的证据**：`init_focus_gradient_direction_survives` 与 `init_task_gradient_direction_survives` 均 **false**。在「把 W_q 初值直接放大到所需范数」这个候选起点上，max attention 达 **0.550492／0.711803／0.985539**（饱和），任务梯度与标定尺度下的梯度余弦只有 **0.2994522016045337／0.3543189984104281／−0.31416493566894504**（聚焦梯度余弦 0.9623893764140008／0.5742075923242334／0.6666168327468923）⇒ **「放大 W_q 初值」这一形式被测量否证**：它买来幅度的同时毁掉梯度方向。分类 `focus_reachable_and_generalizes`。

**标定探针** [probe_u1r2_scale_focus_20261116.py](../../scripts/training/probe_u1r2_scale_focus_20261116.py)／[报告](../../reports/taiji_u2_u1r2_scale_focus_20261116.json)（schema `…-calibration-v4`，verdict `scale_calibrated_supervision_stronger`，**仪器 16/16**、**0 优化器步**、1248 次 backward、6.1094 CPU／6.094 wall、**只测 train 面、fit 场景测 0**）：

- 出生 logit 极差 **0.006967189584054588／0.00584828378200752／0.03569837902765149**，**seed 间离散比 6.104078** 超过声明带宽 3.0 ⇒ **一个全局乘数不行**：全局 185.514 的预测极差 1.292511／1.084939／6.622549，`all_in_band false`，已拒绝。逐 seed 乘数 **430.59／512.971／84.037** 把极差标定到 **3.000002／3.0／2.999985**，`all_in_band true`。
- **只去掉 `1/√H` 阻尼不够**：残余仍需 **152.236554／181.362636／29.711566×**。
- **剂量可行性（理论上界）**：W_q numel 64、384 步、lr 0.001 ⇒ 每坐标位移 ≤0.384、L2 ≤**3.072**，可达乘数上界 **3.169304／3.068541／3.207459**，对所需 430.59／512.971／84.037 的**短缺下界 135.863／167.171／26.2×** ⇒ 幅度不可能由 W_q 自己长出来，必须外置标量。
- **被撤回的代理判据（留痕）**：schema v2 曾用「一步 1% 相对梯度步后 answer_mass 增量的符号」判可动性，得出 `focus_not_movable`。该尺**无判别力**：一步只移动 ~1e-3，而 384 步要覆盖 0.125→1.0；且符号不恒正（softmax 下 `d(answer_mass)/d(logit_j) = a_j·(span_j − answer_mass)`，压平峰内尖峰会把质量漏到峰外）。解析一阶预测与实测在每一行同号同量级 ⇒ 负读数是目标函数几何、不是仪器坏。**已撤回为判据，读数保留为描述**，改用同面同分母的 W_q 梯度信用比较。

## 5 验证与定价（已完成，零项目训练）

**smoke** [报告](../../reports/taiji_u2_u1r2_scale_supervision_smoke_20261116.json)：verdict `smoke_passed`、**63/63**、`uncovered = []`、编码 oracle **96/96**、面 train 64／fit 32／监督 48／有标签 fit 24、**0 优化器步**、0.2656 CPU／0.266 wall、封存 0。逐 seed 覆盖：逐位锚点、B 路径身份（不是 H＋头）、温度是一枚初值 0 的标量、四干预改变读出＋`cut_h`／`h_perturb` 逐位不变、中性出生精确＋新单位 r 信用路接通、全参数有梯度（含 `log_tau`）、信用限制机检、聚焦尺出生 false／oracle true。（出生时 `rgb_zero_agreement` 1.0／0.0／1.0、`fixed_question_agreement` 1.0 是**出生读数**，不是门读数——均匀注意力下置零 RGB 本来就不改读出。）

**契约测试** [tests/taiji_native/test_u1r2_scale_supervision.py](../../tests/taiji_native/test_u1r2_scale_supervision.py)：**22 passed**（rc 0，不经管道取）；U.1 家族三件合跑 **39 passed in 3.30s**（U.1／r1 无回归）。含正负两向：篡改 pin／聚焦门漂移／信用限制漂移各自响亮拒绝；`--run` 授权守卫两支都走（draft⇒`contract_not_frozen`，frozen＋null⇒`missing_training_authority`），并对**真实契约**按当前授权态断言 run 门。

**定价** [报告](../../reports/taiji_u2_u1r2_scale_supervision_price_20261116.json)（schema **`taiji-u2-u1r2-price-v2`**，verdict `price_measured_within_design`）：

- **批量计时**（3 次重复取最大）：一个完整 train epoch（64 更新，其中 48 条带监督）→ `per_update_cpu` **0.00366211**；一个 fit 面（32 次前向）→ `per_eval_cpu` **0.00048828**；8 条干预样本（含历史前向）→ `per_intervention_item_cpu` **0.00390625**。计时面＝出生图（c=0,w=0）。
- 计数：384 更新 ×3 seed、普通评分 **288** 次/seed、干预 **32** 条/seed ⇒ `measured_cpu = 3×(384×0.00366211 + 288×0.00048828 + 32×0.00390625) =` **5.0156**（恒等式机器复验）。
- **帽＝实测 ×2.0 余量**：`cap_cpu` **10.0312**、`cap_wall` **20.0625**（wall＝CPU 帽 ×2，均按未舍入值计算后舍入），设计帽 600.0 CPU／1200.0 wall 之内。参照：r1 同样 3×384 更新实耗 6.1094 CPU。
- 纪律：**210 优化器步 = 3×64（计时 epoch）＋18（剂量）**，模型全部用后即弃；项目数据读 1、check/final/confirm 读 0；本次定价自身耗 1.7969 CPU／1.985 wall。

**旋转需求与剂量（定价探针内，18 步、用后即弃）**：温度供幅度之后，W_q 只需**转方向**。所需相对位移＝`2·sin(θ/2)`，θ＝出生 W_q 与最小二乘解的夹角：

| seed | 夹角（度） | 余弦 | 所需相对位移（**上界**） | r2 分组实测外推 | 余量 | 统一 lr 外推 |
|---|---|---|---|---|---|---|
| 2026111601 | 79.85789007798752 | 0.17609024595749684 | 1.2836742219445736 | **2.193942355247808** | 1.709111× | 1.100725073769154 |
| 2026111602 | 77.0789857448869 | 0.22360761214399938 | 1.2461078507545007 | **1.9672341269836733** | 1.578703× | 0.9824065758243421 |
| 2026111603 | 88.5786152910696 | 0.02480529860862705 | 1.3965634259791948 | **2.486697791621312** | 1.780583× | 1.2393032231618415 |

「上界」是因为最小二乘解只是**零空间维数 46** 的解集里的一点（`bound_note: distance to the minimum-norm solution; the focusing set is larger`），真实需求只会更小。**统一 lr 三 seed 全部不够**（1.100725／0.982407／1.239303 对 1.283674／1.246108／1.396563），2× 组后全部够。温度侧：`log_tau` 外推 **13.064765675142887／15.062589881370172／13.090104833762705** 对所需 7.952650771584998／8.354524580834456／7.629992140112775 ⇒ `r2_groups_reaches_scale` **true×3**；统一 lr 只有 **0.31694772763216184／0.345989007574615／0.33030955589352584** ⇒ **false×3**。外推口径：**3 步位移线性外推到 384 步**，依据是 Adam 每坐标位移由 lr 封顶（`note` 原文），**不是 384 步实测轨迹**。

**库外输入件可重建（R-REBUILD-1，实测而非声明）** [verify_u1r2_rebuild_20261116.py](../../scripts/training/verify_u1r2_rebuild_20261116.py)／[报告](../../reports/taiji_u2_u1r2_scale_supervision_rebuild_20261116.json)：bundle 与数据都在 `.gitignore:160:data/` 之下（`git check-ignore -v` 现测、`git ls-files --error-unmatch` rc1 ⇒ 未入库）。分类 **`regenerable_exact`**：3/3 出生胶囊经 `build_graph→build_capsule→save_capsule` 重建后**三方 sha256 逐字节相同**（各 **27683** 字节）；**6/6 负对照全拒**（`position_head.weight` 与 `log_tau` 各 +1e-3，三 seed 各两枚）；3/3 数据分区 `value_digest` 与 id 序列相同、整集等于 `data_digest`；跨 seed 重建摘要互不相同；临时目录跑完即删（`scratch_removed true`）。**1.1875 CPU／1.344 wall、0 优化器步、封存 0**。数据半边直接复用 r1 核验器的 `bytes_sha`／`dataset_rows`／`classify`／`PARTITIONS`（同数据根同分区），**r1 核验器一字节未改**（`git diff --quiet HEAD` rc0）；只有 bundle 半边是 r2 自己的（多一枚 `log_tau` 负对照）。重建后再冻结一次，三枚 bundle 摘要与报告里记录的逐条相同。

## 6 自造缺陷与更正留痕（本轮我自己造出来的）

1. **「W_q 保持出生范数所以旋转可达」是假设，被自家测量否证**。我先用理论 L2 上界 3.072 声称可达；实测统一 lr 的 384 步外推只有 1.100725／0.982407／1.239303，对需求 1.283674／1.246108／1.396563 ⇒ **三 seed 全不可达**。另发现可行性探针的 `rotation_dose` 被喂的是**拟合后**范数（4026.1787976369974）而不是出生范数（1.4161221338787644），所以它答的根本不是「出生方向能否转到位」。修法：新增 `rotation_requirement` 直接量夹角，并给 W_q 开 2× lr 组，**重测**后三 seed 可达（余量 1.709111／1.578703／1.780583×）。
2. **1.5 CPU 的帽是计时抽签，已撤回**。`process_time` 分辨率约 0.0156 s，单次调用计时全部量化到 0.0 ⇒ 算出 `per_update 0.0`、帽 1.5。改为批量计时（epoch／面／8 条干预各除以批量）＋声明 2.0 帽余量，schema 升 **price-v2**，撤回理由留在报告的 `low_tick_limit` 字段。
3. **`rebuild_command` 原写 `--freeze`，那是一条跑不了的命令**（该模式对已冻结契约抛 `contract_already_frozen`，且会重写契约）。改为专用核验器；`input_artifacts: regenerable_exact` 从**声明**升级为**实测**（§5）。
4. **`with_neutral_birth` 的 deepcopy 崩溃**：前向把非叶张量留在 `self.attn_scale` 上 ⇒ `RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol`。`try/finally` 复位＋回归测试。
5. **契约曾以 GBK 落盘**（`write_text` 漏 `encoding=`，中文「乙」变 0xd2）⇒ 按字节转回 UTF-8，`freeze` 改 `write_bytes(...encode("utf-8"))`，与 `u1.write_report` 同口径；现测 11691 字节、CRLF 0。
6. **两处恒真/空转件已删**：为 justification 一个未用 import 而写的 `assert encode_scene is not None`，以及一个无条件 `raise` 的 stub 函数。
7. **一条我自己写错的不变量**：`test_frozen_contract_carries_authority` 断言「status frozen ⇒ 已有训练授权」，与 r1 设计冻结先例（`git show c32afc541` 实测 status frozen、authority null）矛盾。改为「frozen ⇒ 帽已实测且为正」＋对真实契约按授权态两分支断言 run 门。

## 7 能签与不能签

**能签**：设计＋判据＋出口冻结（criteria_sha256 `c8f2d98b…`、12 枚 source pin、3 枚出生胶囊摘要）；机制是**薄扩展**（`log_tau=0` 与父卡逐位相同、只多一枚标量、route 不同故旧胶囊响亮拒绝）；信用限制**真的在做事**（值侧 0.30–1.27 的聚焦梯度被挡掉，机检 max_abs 0.0）；聚焦尺**在 fit 面有量程**；**存在**能聚焦且泛化的 W_q（最小二乘存在性）；幅度与旋转在**声明的外推口径下**可达；价格实测（帽 10.0312／20.0625）；库外输入件 `regenerable_exact`；零项目训练纪律（smoke 0 步、定价 210 步全弃、封存 0）。

**不能签**：父门／能力／check②／22.6 一律不签，0 产品默认改动。**「r2 必过门」不成立**——最小二乘拟合成功**不证** Adam 能走到那个解（46 维零空间里的一点）；3 步线性外推**不证** 384 步轨迹；值侧探针界定的是**冻结权重下读出头的上限**，不是可训底物的上限（seed02 完美聚焦也只 0.71875 ⇒ 逐 seed 全过非探针能保证）。四干预全真与 `cut_h` 逐位只证**机制接通**；编码 oracle 96/96 只证**输入信息忠实**。

**未测／限制（必须随读数引用）**：① r2 自己的 384 步能否让**聚焦与值侧同时**达门——这正是待授权的那一次跑；② **本轮同时动三个变量**（温度标量、W_q 2× lr、注意力监督），帽内**无消融臂**⇒ 若判负，**不能归因到单一变量**；这是修订上限用尽（2/2、剩 0）下的取舍，owner 裁决的乙本身就含尺度＋监督两件事；③ 温度与监督的交互未测；④ 本卡剂量响应未重测（引用 U.1 同面 8× 读数）；⑤ 全量回归本会话未重跑，只声称 U.1 家族 **39 passed in 3.30s**，「384 回归全过」沿用更早冻结时点声明、**不作为本卡证据**。

## 8 当前状态与下一步

**停在 owner 训练授权是非题**：`read_3seed`，帽 **10.0312 CPU／20.0625 wall**（实测 5.0156 ×2.0 余量），3 seed ×384 更新、float64、三参数组、48 条 train 场景带聚焦监督、零封存读取。契约 `training_authority` 现为 **null**，`guards(contract,"run")` 实测抛 `missing_training_authority` ⇒ **授权前 `--run` 起不来**。

授权后按序执行 `--run`→`--recover`→`--verify`，按 §3 五个冻结出口结案；授权字符串写进契约、**用完即止不外推到下一卡**。判负则修订上限已用满（2/2、剩 0），按程序进入**单位核心 B 读路径收束**裁决，须写清残留（聚焦是否达标、值侧是否达标、三变量不可归因）。**裁决前 0 训练、0 能力、0 A 父；check② 不读、22.6 不解锁；字节世界 effort 保持全局收束。**
