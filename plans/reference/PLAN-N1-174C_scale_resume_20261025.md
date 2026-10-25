# N1-174C：从 epoch 30 续训到 40（利用耐久 checkpoint 完成规模判定）

日期：2026-10-25，Asia/Shanghai。状态：**实施前冻结**。承接[PLAN-N1-174B](PLAN-N1-174B_scale_comparison_rerun_20261025.md)。

## 1. 事实与修正理由

- N1-174（阈值 3,200）在 epoch 23–24 被截断；N1-174B（阈值 5,000）在 epoch 37–38 被截断（实际每 run 约 5,012 CPU，六 run 并行时单步成本比单独试跑高约 50%，我的估计仍偏低）。
- N1-174B 在 epoch 20、30 保存了耐久 checkpoint，并已在**新进程**恢复核对（digest、revision/steps/曝光、下一步 logits、15 byte 继续生成）。
- 因此不必第三次从头重跑：从 epoch 30 的 checkpoint 续训 10 个 epoch 即可到预注册终点 40。checkpoint 保存了参数、AdamW、RNG、steps/曝光与 cursor，批次顺序只由 `(seed, epoch)` 决定，续训与原始运行的后半段是同一路径。

## 2. 续训不变量（工程门）

1. 载入的 checkpoint 必须是 N1-174B 的 `epoch30.pt`，cursor 必须为 `{epoch:30, batch:125}`，`steps=3,750`、`parameter_revision=3,750`、曝光等于 30×每 epoch target byte 数。
2. 续训 epoch 31–40 使用与 N1-174B 完全相同的 `epoch_batches(seed, epoch)` 与训练器；不换数据、不换种子、不改学习率。
3. **复现校验**：N1-174B 的 PARTIAL 报告保留了 epoch 31–37/38 的 epoch 日志；续训得到的相同 epoch 平均损失必须与之一致（平均损失差 ≤1e-5）。若不一致，判定续训路径不等价，记工程失败，不判 R/G。
4. 40 步完成后测同一 16 训练样本与 16 留出题，epoch 40 加对照 A/B 与错配 TF；判据逐字沿用 N1-174：R（训练自由中位数 ≥0.50，≥2/3 seed）与 G（48 留出读数中 ≥24 命中且比另一臂多 ≥8）。

## 3. 判据与出口

- G 成立 `PASS_BINDING`；R 成立而 G 不成立 `FAIL_NO_GENERALIZATION`；R 不成立 `INCONCLUSIVE_ROLLOUT`；任一 run 未 complete 或复现校验失败记 `PARTIAL`/工程失败，不判通过。

## 4. 费用评估

- 每 run 续训 10 epochs：10×125 步×实测约 1.1 CPU/步 ≈ 1,375 CPU，加评估/恢复约 50 → **约 1,425 CPU/run**；六 run ≈ **8,550 CPU**。
- 六进程并行，预计 wall **30–40 分钟**；单 run 防失控阈值 2,600 CPU / 4,200 wall（按实测 1,400 留 85% 余量）。
- N1-174 已消耗约 49,200 CPU（两次截断）；本次续训后 N1-174 家族合计约 5.8 万 CPU，仍在本项目合理量级内。

## 5. 落地顺序

1. `scripts/training/calibrate_taiji_n1_plan174c_resume.py`：`--arm/--seed` 续训、`--all` 并行、`--aggregate` 判 R/G；
2. 先单 run 试跑核对复现校验，再跑六 run；
3. 写 [N1-174C 结案](PLAN-N1-174C_execution_closeout_20261025.md)（含 174/174B 截断事实），同步 03/10/黑板，跑门禁并提交。
