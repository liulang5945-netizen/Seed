# M4.V2 B3 pilot 预注册（K 相位三臂：frozen / continuation / fast+slow+replay）

> 冻结日期：2026-09-10。前置：修订执行序列 §B3（诊断 pilot 定义）、[B3 pilot probe](../../reports/taiji_m4v2_b3_pilot_probe_20260910.json)（F1 parent S/G 相位的三臂机械验证，11 项门全过）、[C-entry 证据线收束](M4V2_B3_K_C_PARITY_V3_PREREGISTRATION_20260910.md)（合成方式不可分；fixed-large 概率平均保留为 strong arm）、[信号空间扩展](M5_K_SIGNAL_SPACE_EXPANSION_PREREGISTRATION_20260910.md)（5 类 v4 workers）。本文冻结 K 相位的 B3 诊断 pilot；pilot 只检查可学习性与机制闭合，不晋级。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 诊断问题（非晋级假设）

> C-entry 已证：在等预算等容量下，continuation learner 的**输出合成方式**不影响表现（权重平均 ≈ 概率平均）。B3 的未检验变量是**学习机制本身**：把局部 delta 写进 fast（wake）再经真实经历 replay 巩固进 slow（sleep），与直接写权重的 continuation，在相同流与预算下是否产生不同的泛化表现？pilot 产出三臂的 validation 表现与机制闭合证据，为 C 阶段的正式比较冻结判据提供依据。

## 2. 基底与臂（全部从同一 v4 父代派生）

- **基底**：`checkpoints/taiji_k_workers_v4/model_17/` 的 K1/K2 workers（5 类词汇、K1 472 / K2 5176 参数）+ K3 不变；单 model seed、单课程（course 0 的 5 类类块）——沿 §B3「一个模型 seed、一个课程」；
- **流**：150 experiences（5 类平衡、课程独立性设计延续）；训练超参不变（semantic 2.0 / transition 0.2，每 experience 1 epoch）；无 optimizer state；
- **三臂**：

| 臂 | wake（流上 150 experiences） | sleep（replay + 巩固） | 新增步口径 |
|---|---|---|---|
| F frozen | 无 | 无 | 0 |
| C continuation | 局部 delta 直接写权重（C-entry 同款） | 无 | 300（150×2 workers） |
| FS fast+slow+replay | 同一局部 delta 写入 **fast_delta**（slow 不动） | replay 缓冲记录真实经历；sleep = 先对 **b=50** 条确定性抽样的 replay 经历做 slow 更新，再 consolidate（slow += fast; fast = 0） | wake 300（fast 写入）+ replay 100（50×2，**单列成本**）+ 巩固 1 次 |

- **fast/slow 语义（冻结）**：每个可学习头张量 T 拆为 `slow + fast`，有效权重 = slow + fast；wake 的局部 delta 只写 fast；sleep 先 replay 更新 slow、后 consolidate 并清 fast；checkpoint 同时保存 slow/fast，fresh restore 后有效权重逐位一致；
- **实现**：新模块 `taiji/k_fast_slow.py`（沿 k_widened 的变体模式，不改 frozen 的 k_continuation v1 / k_widened v2 合同）；
- **replay 抽样**：确定性（seed 派生的 Fisher-Yates，同 anchored_permutation 模式），b=50，可 fresh restore 重算。

## 3. 量尺与口径

- **主量尺**：validation combined MSE delta vs frozen（validation 已被既有 formal 读取——诊断 pilot 允许，sealed v3 **保持未读**，留给 C 阶段正式比较）；
- **口径**：`new_update_steps`（inherited 2400/6720 单列）；FS 的 replay 100 步单列成本，不计入预算比较；
- **描述性**：逐分量 MSE、训练损失轨迹摘要、fast 范数轨迹（wake 期间应单调变化、sleep 后归零）。

## 4. 机制门（pilot 的主要交付；任一失败即停）

1. F 臂零更新（权重 digest 与父代一致）；
2. FS 出生时 fast=0；wake 期间 slow 逐位不变、fast 单调累积；sleep 后 fast=0 且 slow 包含 replay+consolidate 的贡献；
3. replay 事件 = 流上真实经历的 digest（内容寻址，非合成）；
4. 三臂 checkpoint 落盘 → fresh restore → 有效权重逐位一致；rollback 回父代；
5. parent checkpoint 与 K3 不变；
6. 预算核算：C=300、FS wake=300（replay 100 单列），流 digest 一致。

## 5. 诊断读数（非门，如实呈现）

- 三臂 validation combined delta 排序（F=0 基线；C 与 FS 的差值即 fast/slow+replay 机制的净效应）；
- FS 的 replay 抽样与巩固对 D/R 类（父代弱项）holdout 分量的影响；
- 每臂耗时/内存摘要。

## 6. 停止线与边界

- 机制门失败 → 停，修 fast/slow 实现，不调学习率/抽样；
- **sealed v3 不读取**——它保留给 C 阶段的正式比较；
- 不引入 provider/联网/真实客户端写入/default runtime；`can_promote=false` 固定；
- pilot 结果不改变 C-entry 的收束结论（合成方式不可分），只回答 fast/slow+replay 机制在 K 相位是否闭环、是否值得进入 C 阶段正式比较。

## 7. 产物顺序

1. `taiji/k_fast_slow.py`（fast/slow 语义 + replay 缓冲 + checkpoint/restore）+ 定向测试；
2. `scripts/training/eval_taiji_m4v2_b3_k_pilot_v2.py`（三臂 + 机制门 + 诊断读数；先 py_compile/ruff）；
3. 报告 `reports/taiji_m4v2_b3_k_pilot_v2_20260910.json`；
4. 路线图执行记录 + 独立提交；C 阶段正式比较的判据预注册在 pilot 读数之后另行冻结。

## 8. 执行修订注记（2026-09-10，运行中发现并修正）

**「wake 轨迹与 continuation 逐位一致」的断言仅在精确算术成立**：FS 把 delta 累加进零初始化的 fast（`W_parent + fl(Σd)`），C 把 delta 就地加进运行态（`fl(fl(s+d)+d')`）——两者舍入路径不同，150 步后产生微小状态差，位等价不可达。这不是实现 bug（delta 函数与顺序完全一致），是预注册断言的浮点盲区。修正：该门从位等价改为**容差门**（max abs diff ≤ 1e-5，实测偏差进报告）。机制意图（同 delta、同顺序）不受影响。

## 9. 执行记录（2026-09-10，pilot 已运行，机制门 12/12 全过）

定向测试 5/5（fast/slow 语义、wake≡continuation 轨迹、replay/consolidate 语义、checkpoint 往返、replay 抽样确定性）。三臂 pilot（单 model seed 17、course 0、150 experiences 5 类平衡）结果：

- **机制门 12/12 通过**：F 零更新 ✓；FS fast 出生为零 ✓、wake 期间 slow 逐位不变 ✓、wake 后 fast 非零 ✓；wake 轨迹偏差 **2.38e-7**（容差 1e-5）✓；replay digest ⊆ 流 digest ✓、replay 写 slow ✓；consolidate 清零 fast ✓ 且**保持有效权重不变** ✓；fresh restore ✓；parent/K3 不变 ✓；预算 C=300、FS wake 150+replay 50+consolidate 1 ✓。
- **诊断读数（validation-only，sealed v3 未读）**：continuation delta **−0.000455**、FS delta **−0.000335**——FS 的 replay+consolidation 在整体 validation 上略逊于 C（对已见经历的二次 pass 未带来泛化收益）。
- **D/R 弱类探针（诊断性）**：frozen combined 0.1996（父代在弱类上误差大）；continuation delta −0.1914；**FS delta −0.1928**——**FS 的 replay+consolidation 在父代弱类上优于 C**。这是一个真实信号：replay 对新扩展类（父代训练不充分的类）有针对性收益，尽管整体 validation 略负。
- fast 范数轨迹：wake 期间 K1 0.91→5.72、K2 0.36→1.61（单调累积可见），consolidation 后归零。

### 9.1 pilot 结论

1. **机制闭环**：fast/slow 拆分、wake 写 fast、真实经历 replay 写 slow、consolidation 清零且保持有效权重、checkpoint/restore——全部在 K worker 上闭合。
2. **机制效应定位**：replay 的收益是**类选择性**的（弱类改善、整体略负），不是全局泛化收益。C 阶段若引入 fast/slow+replay，其判据应按类分解（弱类改善 vs 整体非劣），而非单一 combined 指标。
3. **预算事实**：FS 比 C 多消耗 100 次 replay 更新（单列），换来弱类 −0.0014 的额外改善；是否值得由 C 阶段判据决定。
4. 下一步：C 阶段正式比较判据预注册（若启动）需冻结类分解判据；在此之前不训练、不读 sealed v3。
