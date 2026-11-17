# 22.5.0 A 父门收束：`closed_limited_exit`（等算力配平不可达）

> 卡号：`22.5.0`｜日期：2026-11-17｜状态：**`closed_limited_exit`**｜判据：**`compute_parity_unavailable`**｜性质：**零训练收束**，无架构优劣结论、无能力、无训练授权。
> 机器回执：[taiji_u2_22_5_0_closeout_20261117.json](../manifests/taiji_u2_22_5_0_closeout_20261117.json)（sha256 `da526b7fc52ded1c8ffac1881aee99e6c0b42de0e76dec6dc9e18e313b3f3fde`）。
> 依据：[时间配平结果](RESULT-TAIJI_U2_22_5_0_COMPUTE_PARITY_20261117.md)、[FLOPs 复核](../../reports/taiji_u2_22_5_0_flops_parity_20261117.json)、[22.4.4 价格](RESULT-TAIJI_U2_22_4_4_PRICING_20261117.md)、[计划 §22.5 前置](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)。

## 1 关闭范围

- 关闭的是 **22.5.0 A 父门**，不是 N 主线、不是 22.4 工程父、不是单位核心方向。
- 未开 A 父门、未进入 22.5.1 B、未解锁 22.6／22.7；**0 训练新增、0 能力、0 新父**。
- 22.4 工程父仍是 `accepted_engineering_parent_for_22_5`、`authorized=false`、`capability_qualifications=[]`。

## 2 为什么必须停

| 允许的配平量 | 读数 | 256/24 包络 | 结论 |
|---|---|---|---|
| 时间（wall） | U2 对 transformer **57.99×**、对 GRU **64.91×** | 同时需基线 **1,392／1,558** 次 > 256；基线满 256 次则 U2 **4.41／3.94** 次 < 24 | 不可达 |
| FLOPs | U2 对 transformer **26.31×**、对 GRU **29.92×** | 上限 **10.67×** | 不可达 |

- 时间读数来自 22.4.4 价格（U2 0.444676 wall；transformer 0.007668；GRU 0.006851）。
- FLOPs 计数：U2 **30,894,768**、transformer **1,174,400**、GRU **1,032,464**（同片段、同线程、前向+反向）。
- 首次 A 清单交叉核对：Native **1.373**、Transformer **0.1816**、GRU **0.1406**，更新数 24／188／243，已是 `compute_parity_unavailable`。
- 两个允许量都失败。**不再换第三个指标、不放大便宜臂、不改 256 上限、不删基线**；这些都会把测量 regime 改掉，而不是得到架构证据。

## 3 保留不变

- 22.4 七项交付与工程父（`authorized=false`）。
- 22.4.3 分区：`data_recipe_digest=a96cf26e…`、`partition_digest=8f16e764…`。
- 22.4.4 价格：`price_digest=25d0e692…`；U2 单次学习 0.453125 CPU。
- 全局 check **8 次未用**；final 仍 `reserved_not_generated`、`R-SEAL-1`、未读取。
- 未改任何门值、剂量、价格、check 曝光或 final seal。

## 4 不签

A 父、B 学习、check②、22.6、22.7、架构优越/低劣、任何能力、训练授权。**也不写"U2 架构被证否"**；本卡只说明当前 256 上限与两种配平量下无法做等算力对照。

## 5 唯一后继：owner 决定 C

当前窗口按 D 收束；若要继续架构对照，唯一上限更高的路径是 **C：换一条同成本基线或 U2 变体，走新身份版本与重冻**。C 必须由 owner 明确授权，**不是自动下一步**；在 C 之前不得开训练。

## 6 成本

- 时间配平检查：0.0 CPU／0.000967 wall；FLOPs 复核：7.0 CPU／7.397 wall；合计约 **7.0 CPU**。
- 0 optimizer、0 项目数据、0 check、0 final；FLOPs 复核 3 次前向+反向。
- 本收束脚本只读 JSON、0 模型前向。
