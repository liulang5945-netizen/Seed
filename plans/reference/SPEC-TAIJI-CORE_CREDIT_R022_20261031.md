# R022：09.2.2 负责参数信用与实际数值判别合同

> 文档状态：[合同] — 只冻结09.2.2在[R021正式参考图](SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)上的信用/导数/负对照/实际数值更新范围。不授予任务训练、定价、出生或能力结论。调度只见[MAINLINE](../MAINLINE.md)。

日期2026-10-31。对象为`taiji/living_core.py`在R021回执所绑定的实现与101个参数张量；本合同把R021回执中`learning_ownership_and_actual_updates`（09.2.2）从未实现态推进为已验态。09.2.3覆盖出生/噪声标定、编码迁移，09.2.4完整主体恢复仍分别验收；本合同全部通过也不准入训练。

## 1. 范围与边界

- 只用合成夹具（n=2/d=8/m=12/a=2/L=2/k=v=4/J=2，与R021工程夹具同构，44,905参数）；不读任务语料分区、不做任务评价、不接持久服务、不改产品默认；`data_splits_read=[]`保持。
- 本阶段的"优化器更新"是为验证`Delta_theta`、矩状态与输出改变而**声明的合成更新表达式**（见§5），不是训练；更新后的参数即刻废弃，不落盘、不结算成模型。
- 主dtype为float64；float32按§6逐组比较。验证失败的唯一诚实出口是负结果或工程失败，不以阈值调整换通过。

## 2. 声明的负责目标（synthetic、预声明）

| 目标 | 定义 | 声明拥有的参数族 |
|---|---|---|
| T_EXPR | 合成prompt/target/goal/records经`score_targets`的教师强制预测NLL（多步，allowed=True，含coverage/prior可微递归） | encoder、nu_projection、message_q/k/v/goal、fast_queries、fast_outputs、episodic、W/U/V/R/P/tau_logit、xi、readout、expression、write_heads（经跨步F链） |
| T_WRITE2 | 两步链：第1步以allowed=True写F，第2步读F并取该步预测NLL | write_heads（隔离链；同时复核动力学参数） |
| T_WORLD_PROXY | 三个世界头对**声明常数**代理目标（固定种子合成后继表示与reward）的MSE，仅对world族收集梯度 | world（代理目标；真实世界损失按R021回执仍属09.3/09.4，代理通过不签世界学习闭环） |

每个张量必须在其声明拥有者的目标上给出**可分辨且经有限差分确认**的梯度，或落入§4某个声明零类并配对通过正对照；二者之外一律记`unexpected_broken`并判本阶段失败。`requires_grad=True`本身不构成信用。

## 3. 导数与尺度匹配判别

1. 一阶：每张量取≤12个坐标（确定性行距采样＋最大|g|坐标），autograd中心有限差分在h∈{1e-4,1e-5,1e-6}（相对坐标尺度max(1,|θ_i|)）三点扫描，容差逐坐标尺度匹配`|g_a-g_f| ≤ 1e-8 + 1e-3·(|g_a|+|g_f|)`；门＝三eps中**至少两个**坐标级通过且最优eps通过（FD收敛到autograd而非单点巧合），逐eps失配全量记录。
2. 方向聚合：每张量沿固定种子单位方向的方向导数，autograd vs 中心差分（h=1e-5），同容差。
3. 二阶：每张量沿自身归一化梯度方向（方向在基点冻结）的方向二阶导（create_graph双重反传）vs 中心二阶差分（h=3e-4），容差`1e-6 + 5e-2·(|d2_a|+|d2_f|)`；若路径中出现范数裁剪分支激活，该张量二阶记`unresolved_clamped`，不算失败也不算确认。
4. 信用活动下限：声明`max|g| ≥ 1e-10`（float64工程下限，非任务门）；低于者记`below_noise_unresolved`，须在float32比较与二阶项中同样标记，不得写成已确认。

## 4. 零类与负对照（区分正常零、许可阻断、意外断图）

| 声明零类 | 探针 | 必须配对的正对照 |
|---|---|---|
| birth_zero（出生U/V一侧） | 初值U=0⇒V在三个目标上梯度恰为0 | 实际更新使U≠0后V梯度非零（§5） |
| ordering_blocked（表达后写） | 单步T_EXPR（一步即损失）write_heads梯度恰为0 | 多步T_EXPR/T_WRITE2中write_heads非零 |
| permission_blocked（硬许可） | T_WRITE2同夹具allowed=False，write_heads梯度恰为0 | allowed=True非零 |
| state_conditioned_zero | coverage≡0时`expression.coverage`为0；prior≡0时`expression.continuation`为0；空记录时情景位置路（a/h/g/meta/second/score/value/value_meta）为0 | coverage>0/prior>0/含记录夹具中对应张量非零 |
| forced_mix_cut（无合法复制位） | 空view时g_mix强制1⇒copy位置路（a/meta/position/coverage/score/mix）恰为0 | 含记录夹具非零 |
| inert_shift_invariant（softmax平移不变，本阶段发现并补录） | `episodic.score.bias`（I3的b_3）：位置softmax对全体分数共同平移不变，该参数对任何输入任何取值都不影响输出 | 值置换对照：置为固定种子非零值后损失不变；梯度\|g\|≤1e-14；FD最优失配≤atol。这是R021 I3声明的结构性死参数，建议后续接口修订删除或标注；修订前保留原位、永不获信用、Adam矩恒0 |

- **断路负对照**：以上硬分支的"恰为0"用逐元素`max|g|==0`（精确零）验证，不是近似小。
- **置零负对照**：每族取max|g|张量（含W/U/tau_logit全查）。对照的目的＝证明参数与损失之间存在实际因果通路，效应大小由FD确认的梯度负责，不在此处设大效应门。判据：`ΔL_zero > max(1e-10, 1e-8·|L|)`，或一阶预测`|g·θ| ≤ max(1e-10, 1e-8·|L|)`（梯度本身预言低于噪声的效应时，连通性由已FD确认的梯度承担）；ΔL_zero、ΔL_sign与一阶预测逐行落账。V置零无改变须与birth_zero一致（记expected_no_effect）。初值即全零的参数（U）的置零/翻符号对照空转，替换为**激活对照**：置为固定种子非零值并翻符号，损失均须改变。
- **翻符号负对照**：同批张量翻符号后损失改变非零且与置零改变可区分（|ΔL_sign−ΔL_zero|>同阈值），证明响应非退化。
- 任何"梯度为零＋无声明零类"的组合＝意外断图→`failed_unexpected_graph`。

## 5. 实际数值更新（声明合成更新表达式）

- 声明Adam(lr=1e-2, betas=(0.9,0.999), eps=1e-8, 无weight decay)，更新目标L=T_EXPR+T_WRITE2（T_WORLD_PROXY不进入更新，代理不得成为训练信号），共**2步**；每步逐张量记录max|Δθ|、‖Δθ‖₂、Δθ/ULP(θ)比（ULP=eps_mach·max|θ|）、Adam矩m/v的max/mean。
- 梯度恰为0的张量（初值V、世界族）Δθ须精确为0；inert参数（`episodic.score.bias`，梯度为~1e-18级浮点抵消残留）在声明更新中的累计|Δθ|须≤`lr·1e-9`且更新后梯度仍≤inert下限（更新可忽略，不构成激活）。
- 第1步后重算：V梯度非零（birth解锁）、同夹具输出分布max|ΔP|>0并落账；此验证只证明数值信用链通，不证明任务收益。
- 全程`optimizer_steps=2`落账；updated参数hash记录后废弃。

## 6. float64/32逐组比较

同夹具分别以float64/float32完整重算三个目标的梯度；对float64下`max|g|≥1e-6`的张量，逐张量相对偏差`max|g32-g64|/max(max|g64|,1e-8)`门限0.1（族median另记）；float64下<1e-10的张量记below_noise_unresolved不判负。全部梯度须有限。本门是数值判别，不是float32训练可行性证明。

## 7. 预算、出口与交付

- 预算：本验证全run≤900 CPU秒/20墙钟分钟；超限即`budget_stop`，不缩eps重跑。
- 出口：`passed_r022_credit_only`／`failed_unexpected_graph`／`failed_fd_disagreement`／`failed_control`／`failed_fp32_deviation`／`budget_stop`／`engineering_failure`。通过仅表示：101张量的信用归属、导数数值、负对照与声明更新表达式在该合成夹具上判别成立；不表示任何任务能力、生长、保持或恢复成立。
- 交付：本合同、`tests/taiji_native/test_living_core_credit_r022.py`（检查与夹具）、`scripts/training/verify_taiji_living_core_credit_r022.py`（write-once报告器）、`reports/taiji_core_r022_credit_validation_20261031.json`（逐张量分类、FD/eps/二阶读数、负对照、Δθ/矩/ULP、fp32对比、CPU/wall、源码SHA与baseline绑定）。
- 报告须明列：backward_calls>0、optimizer_steps=2、data_splits_read=[]、full_subject_restore=False、default_model_replaced=False、无持久主体/服务。本合同不修改living_core.py实现；若实现需变更，先出R021接口修订再重跑。
