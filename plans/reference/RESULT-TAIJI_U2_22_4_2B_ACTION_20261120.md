# 结果卡：22.4.2b 动作片段（`action_consequence_pass`）——动作方向因果贡献成立

> 文档状态：[结果卡/通过] — 2026-11-20。合同：[`taiji_u2_22_4_2b_action_20261120.json`](../manifests/taiji_u2_22_4_2b_action_20261120.json)（`status=frozen`、`training_authority=owner:standing-authorization:2026-11-20#recommendation-default`、`result.closed=true`）。回执：[run](../../reports/taiji_u2_22_4_2b_action_run_20261120.json)／smoke。**0 check/final。**

## 1. 协议（修正 22.4.2 两缺陷后读前冻结）

- 任务：N=3 同向推，预测被推体最终位置桶（左/中/右，chance 0.333，不动基线 0.406）。
- 结构：`groups=6`（全部 6 通道均路由，修正 D-20261119-54）＋严格 split（训练仅 256，fit 32 只评测，修正 22.4.2 fit 混训）。
- 四臂：`full`／`action_masked`（动作两维置零）／`direction_permuted`（方向维打乱）／`proxy_removed`（当前桶代理维置零）；30 单位、1,200 步、3 seed。
- 门：完整 ≥max(基线+0.05,0.45)；完整−反事实 ≥0.05 且 3 seed 同向；配对单侧 95% LB>0；恢复 exact。

## 2. 读数（`105.19 ≤ 帽 150 CPU`）

| 臂 | seed 0 | seed 1 | seed 2 | 均值 |
|---|---|---|---|---|
| `full` | 0.750 | 0.813 | 0.781 | **0.781** |
| `action_masked` | 0.406 | 0.406 | 0.469 | 0.427 |
| `direction_permuted` | 0.469 | 0.406 | 0.469 | 0.448 |
| `proxy_removed` | 0.750 | 0.844 | 0.781 | 0.792 |

- **动作边际** `[0.281, 0.406, 0.313]`（均 ≥0.05、3 seed 同向）；**配对单侧 95% LB = 0.0625 > 0**；恢复 exact；全部 6 项 check 通过 ⇒ **`action_consequence_pass`**。
- `proxy_removed` 与 full 持平 ⇒ 当前桶代理维非必需（颜色分桶已携带位置信息）；该发现不影响动作因果结论，记入设计备注。

## 3. 判读（有界证据）

1. **动作方向对后果预测有真实因果贡献**：去掉或置换动作维，准确率掉回不动基线（0.41–0.45 vs 0.78）；不是「特征增宽」代签。
2. 范围：**仿真物理动作片段**（`advance` 力/质量/阻力/碰撞）、N=3 推、位置桶标签、单任务单主体；**不等于真机迁移**，也不等于完整「物理因果理解」。`physical_transfer` 轴记为 **verified（有界：仿真动作片段）＋ 真机/多动力学族仍 uncovered**。
3. 本卡为新图、无父模型，旧能力保持门 N/A（已披露）；特征覆盖用 `groups=6` 全通道路由。

## 4. 后继

- 扩到多动力学族（推/碰撞/遮挡声响）或真机回放需新卡与 owner 裁定（27 阶段）。
- 22.8 及 23–30 仍锁定；本卡只签该片段。
