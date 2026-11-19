# 结果卡：22.4.2 动作片段（`engineering_failure`——fit 集混入训练，判词作废）

> 文档状态：[结果卡/工程失败] — 2026-11-20。合同：[`taiji_u2_22_4_2_action_fragment_20261120.json`](../manifests/taiji_u2_22_4_2_action_fragment_20261120.json)（`status=frozen`、`training_authority=owner:standing-authorization:2026-11-20#recommendation-default`、`result.closed=true`）。**0 check。**

## 1. 协议

三臂反事实动作片段：`full`（动作方向＋当前桶代理）／`action_masked`（两维置零）／`direction_permuted`（方向维打乱）；32 单位 1,000 步、3 seed；门＝完整 ≥ max(基线+0.05, 0.45)、完整−反事实 ≥0.05、3 seed 同向、配对 LB>0、恢复 exact。

## 2. 读数与问题

- 三臂准确率**完全相同**（full/masked/permuted 均 `[0.5, 0.5625, 0.4688]`），margin 全 0；判词原为 `action_contribution_absent`（68.36 CPU）。
- **工程失败定位**：`train_arm(train_rows + fit_rows, ...)` 把 **32 条 fit 样本混入训练**，三臂在同一污染数据上过拟合，导致三臂输出一致、反事实失效 ⇒ **该判词作废**，按预注册出口记 **`engineering_failure`**。

## 3. 判读与后继

1. 本次不签「动作贡献存在/不存在」；`physical_transfer` 仍 `uncovered`，但动作后果任务面本身已由可行性探针（D-20261119-52）证明可构造可学。
2. 修复口径：训练仅用 256 train，fit 32 只用于评测；另需检查动作维（dims 37–38）是否真的被模型使用（若三臂仍一致，说明特征打包或通道切片有误，须先做特征消融诊断）。
3. 后继 **22.4.2b 新卡**：固定 split＋特征消融诊断（动作维单独输入/置零/置换、当前桶代理去留），帽 150 CPU，零 check；通过后才签动作因果贡献。
