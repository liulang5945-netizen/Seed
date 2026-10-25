# SPEC-N1-174B：修正重跑的数据、训练、评估与判据

日期：2026-10-25。真实更新前冻结，承接[PLAN-N1-174B](PLAN-N1-174B_scale_comparison_rerun_20261025.md)。除阈值与中间 checkpoint 外，与 [SPEC-N1-174](SPEC-N1-174_scale_comparison_20261025.md) 逐条相同。

## 1. 数据与隔离

1. 训练集＝全部 train episode 剔除 N1-171 的 16 个 held-out family，断言 500 episode / 324 family；不满足即工程失败。
2. 留出集为 N1-171 的同一 16 题，selection digest 必须等于 `5db33945cd22ce71d70d8d97f6af72462d4871aa239559876e79cd1e1edd6b40`；留出与训练 side 不相交。
3. target 取前 `min(128, len)` byte；prompt 不截断（`max_observation_bytes=4096`）；空 prompt/target 拒绝。
4. `dev_queries=0`；默认模型不改；`can_promote=false`。

## 2. 训练

- 模型 `ConditionalLanguageCandidateV2`，参数必须 970,945；两臂 `tf`（v2 trainer）与 `ss`（rollout trainer，`max_probability=0.5`、`schedule_steps=256`）。
- 种子 169/170/171；每步 4 题，顺序由 `random.Random(seed*1000003 ^ epoch)` 生成；每 epoch 125 步；40 epochs＝5,000 步。
- AdamW lr0.001、wd0、clip1；`ss` 采样不回传；非有限 loss/gradient 立即停止该 run 并保存 partial。

## 3. 评估

- 0 步、epoch 10/20/40 测 16 个确定性训练样本与 16 个留出题的 TF/自由 128 byte。
- epoch 40 加留出对照 A（同折下一道留出 target）、B（训练 target `(rank*37+11)%500`）与错配 TF。
- 自由生成贪心 UTF-8 合法字节、128 byte 上限、遇 `<|end|>` 停止。

## 4. 判据与出口

- R：epoch 40 训练样本自由匹配中位数在至少一臂的 ≥2/3 seed ≥0.50。
- G：至少一臂 48 个留出读数中 ≥24 个 own ≥0.20 且 ≥A+0.15、≥B+0.15，且比另一臂多 ≥8。
- 出口：`PASS_BINDING` / `FAIL_NO_GENERALIZATION` / `INCONCLUSIVE_ROLLOUT`；任一 run 未 complete → `PARTIAL`，不判通过。

## 5. 存档与恢复

- 每 run 保存 `zero`、`epoch20`、`epoch30`、`final` 四份 checkpoint；每份在新进程恢复并核对 digest、revision/steps/曝光、下一步 logits 与 15 byte 继续生成；全部 write-once。
- 单 run 防失控阈值 5,000 CPU / 7,000 wall；不设整体硬上限，按实测成本给量。

## 6. 费用

- 预计 3,640 CPU/run（训练）＋约 50（评估/恢复）＝约 3,700；六 run 约 22,500 CPU；wall 70–90 分钟。
- 父进程汇总每 run 真实 CPU/wall；异常 run 单独记录，不影响其他 run。
