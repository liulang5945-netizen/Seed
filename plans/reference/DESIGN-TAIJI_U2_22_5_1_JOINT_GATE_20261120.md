# 设计卡：22.5.1 联合门（fit 面）S2 零训练设计＋实测定价（已冻结，训练待 S3 授权）

> 文档状态：[设计卡] — 2026-11-20。对象＝MAINLINE §2 的 `22.5.1` 联合门 `S2`。**本卡只完成零训练设计＋实测定价＋冻结合同＋smoke 接线（0 优化器步）；正式执行（S4）另由合同 `training_authority` 授权。** 定价回执：[`taiji_u2_22_5_1_joint_gate_price_20261120.json`](../../reports/taiji_u2_22_5_1_joint_gate_price_20261120.json)；合同：[`taiji_u2_22_5_1_joint_gate_20261120.json`](../manifests/taiji_u2_22_5_1_joint_gate_20261120.json)。

## 1. 身份、口径与边界（owner 已裁定）

| 项 | 冻结内容 |
|---|---|
| 父 | **单一主候选**＝已接受 v1 A 父 seed `20261112` 的 bundle（`output/a_pixel_parallel_20261119T070447Z/segment.bin`，`pixel-salience`、`d=32`、2,000 更新；记录 check `174/192`、fit `31/32`）。合同绑定：文件 sha256＋读取时 strict load＋记录 fit 复现（S1 已由 owner 豁免，生成链未重跑风险已登记，D-20261119-12）。 |
| 3 seed 统计 | **同一父 bundle × 3 个训练子种子**（`20261110/20261111/20261112`）；父不重训、不换、不池化；禁拼 seed 凑「全绿主体」。 |
| D 判据 | **相对父代**：逐项 `drop = parent_d − child_d`，单侧 95% 上界 `≤0.02`；同时报告绝对 `child_d`。绝对 `D≥20/24` 不再作门（结构上受父代 A 精度上界约束，D-20261119-11）。 |
| 数据 | 反馈训练＝`first_a_support_22_5_v2` train 前 `64` 个 world，已见 2 模板；评测＝fit `32` world 的 24 个非未知 B 项，分已见（2 模板）/未见（第 3 模板倒装）两面。 |
| 分区 | 仅 `train/fit`；**0 check、0 final 读取**（22.5 check 曝光保持 2/2 用尽）；不消耗新曝光账。 |
| 架构门 | **不执行**（`architecture_gate: not_run`）：等算力配平前置未满足（22.5 前置要求），架构净益保留 `negative/inconclusive`，不得以本卡代签架构优越性。 |

## 2. 臂、干预与预注册指标

每个训练子种子在同一父上构造以下臂（父参数逐位冻结，仅 `record_organ` 可训；父 digest 预检必须逐位不变）：

| 臂 | 操作 | 排除的解释 |
|---|---|---|
| `main` | 器官训练 256 步、batch 2 world，已见 2 模板反馈 | 主候选 |
| `frozen_update` | 同数据同调度、`lr=0`（参数零更新） | 「学习」是否真的来自更新（增益对照） |
| `permuted_feedback` | 同场景、监督目标按环移位（build_b_items 闭包补丁） | 增益是否来自正确配对反馈 |
| `overwrite_probe` | 主候选副本 + `delta_head.2.weight += 2.0`（无保护覆盖探针，不训练） | 保持门是否**可失败** |
| 因果切断 | `record_organ`/`fast`/`record_injection` 冻结副本 | 机制必要性 |

**预注册指标**（fit 面，分母 24/32）：
1. `B 成功率 ≥0.80` 且 `纠正率 ≥0.80`（本任务面上二者同项同分母）；
2. **反馈学习净益** `gain = B_main − B_frozen`，逐项配对 bootstrap（10,000 次，seed 固定）95% 下界 `>0`、均值 `≥0.05`、3 seed 同向（均 `>0`）；
3. **A 保持**：`a_drop = A_parent − A_child` 单侧 95% 上界 `≤0.02`；
4. **D 相对保持**：`d_drop` 单侧 95% 上界 `≤0.02`；
5. **因果**：`record_organ` 切断后 B 命中 `≤2/24`；置换反馈臂 B 命中 `≤2/24`；`fast`/`record_injection` 切断读数报告（不作数值门）；
6. **可失败保持**：覆盖探针的 A 下降 `≥0.02`（否则 `retention_gate_not_failable`）；
7. **恢复**：主候选保存→**独立进程**重载→同条件 4 项 A/B 逐字节一致；
8. 父代逐位不变；`0` check/final。

## 3. 出口顺序（读前冻结）

`engineering_failure` → `not_admitted`（父 preflight fit 复现失败）→ `budget_stop` → `recovery_failure` → `retention_gate_not_failable` → `causal_control_failure` → `retention_breach`（A 下降上界 >0.02）→ `d_backoff_breach`（D 相对下降上界 >0.02）→ `learning_gain_absent`（增益或下界或 3 seed 同向失败）→ `correction_absent` → **`joint_gate_fit_face_pass`**。

## 4. 实测定价与硬帽

在丢弃模型上实测（threads=1，一次性前后向、不保留优化器状态）：单臂 256 步训练 **5.59 CPU**；单面 24 项完整 decode 矩阵 **40.44 CPU**；按 3 子种子 ×（3 训练臂 + 3 切断 + 1 覆盖探针）布局：

| 项 | CPU |
|---|---|
| run 合计（3 seed，含 preflight） | 975.19 |
| verify 复算 | 100.52 |
| recover（3 次独立进程重载） | 12.00 |
| 工程余量 | 15.00 |
| **包价** | **1,102.71** |
| **硬帽（×1.25）** | **1,378.38 CPU／45 min wall** |

优化器步上限 `3 seed × 3 臂 × 256 = 2,304`（`frozen_update` 为 `lr=0` 步，计入算力）；`check/final` 读取 `0`。

## 5. 授权状态与唯一后继

- **当前**：合同 `status=frozen`、`training_authority=null`；smoke 全过、`0` 优化器步后呈报 S3。S3 将按 owner 常设授权（“按推荐项先推进、后续回溯修订”）记录 `owner:standing-authorization:2026-11-20#recommendation-default` 并按本卡实价执行 S4，同时显式登记「fit 面、非正式 check② 裁决、架构门未跑」三项边界。
- **唯一后继**：S3→S4 执行 → `--closeout` → S5 结果卡与主线/台账同步；负结果按 §3 具名出口收束，不自动追加剂量/换父/改判据。
