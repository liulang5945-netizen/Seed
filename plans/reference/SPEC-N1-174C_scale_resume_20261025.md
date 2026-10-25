# SPEC-N1-174C：续训的载入、复现校验与判据

日期：2026-10-25。真实更新前冻结，承接[PLAN-N1-174C](PLAN-N1-174C_scale_resume_20261025.md)。除「从 epoch 30 续训」外，数据、训练器、批次顺序、评估与判据与 [SPEC-N1-174B](SPEC-N1-174B_scale_comparison_rerun_20261025.md) 完全相同。

## 1. 载入与身份

1. 只接受 `data/checkpoints/n1_plan174b_scale_20261025/<arm>_seed<seed>/epoch30.pt`；格式必须与 arm 匹配（tf→v2 trainer，ss→rollout trainer）。
2. 载入后断言：`cursor == {"epoch":30,"batch":125}`、`steps == 3750`、`parameter_revision == 3750`、`target_byte_exposures == 30 × Σ min(128, len(target))`。
3. 同时读取同 run 的 PARTIAL 报告，取其 `milestones`（epoch 0/10/20/30）与 `epoch_log`（1..停止 epoch）用于拼接与复现校验；PARTIAL 报告与 checkpoint 均只读。

## 2. 续训

- epoch 31..40，每 epoch 用 `epoch_batches(train, seed, epoch)` 的 125 个 4 题批次；训练器、lr、clip 与原 run 相同。
- 复现校验：对 PARTIAL 报告中存在的 epoch 31..last，比较其 `loss_nats_mean` 与续训同 epoch 的均值；最大差 ≤1e-5 才继续。超过即记 `resume_mismatch` 并停止该 run。
- 非有限 loss/gradient 立即停止该 run。

## 3. 评估与存档

- epoch 40 测同一 16 训练样本与 16 留出题的 TF/自由 128 byte；加留出对照 A（同折下一道留出）、B（训练 target `(rank*37+11)%500`）与错配 TF。
- 保存 `final.pt` 到 `data/checkpoints/n1_plan174c_scale_complete_20261025/<arm>_seed<seed>/`，在新进程恢复并核对 digest、revision/steps/曝光、下一步 logits 与 15 byte 继续生成。

## 4. 判据与出口

- R：epoch 40 训练自由匹配中位数在至少一臂的 ≥2/3 seed ≥0.50。
- G：至少一臂 48 留出读数中 ≥24 命中（own ≥0.20 且 ≥A+0.15、≥B+0.15），且比另一臂多 ≥8。
- 出口：`PASS_BINDING` / `FAIL_NO_GENERALIZATION` / `INCONCLUSIVE_ROLLOUT`；任一 run 未 complete 或复现校验失败 → 不判通过。

## 5. 费用与保护

- 单 run 防失控阈值 2,600 CPU / 4,200 wall；不设整体硬上限，六 run 预计合计约 8,550 CPU、wall 30–40 分钟。
- 父进程汇总每 run 真实 CPU/wall；异常 run 单独记录，不影响其他 run。
