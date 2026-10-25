# SPEC-N1-174：规模对照的数据、训练、评估与判据

日期：2026-10-25。真实更新前冻结，承接[PLAN-N1-174](PLAN-N1-174_scale_comparison_20261025.md)。本包只验证「500 题规模下两臂是否抬起自由滚动与留出绑定」，不是默认采用或 L2 判定。

## 1. 数据与隔离

1. 语料仍来自冻结的 assistant train 加载函数；训练集为全部 train episode 中**剔除 N1-171 的 16 个 held-out family**后的 500 题（断言 500 episode / 324 family；不满足即工程失败）。
2. 留出集为 N1-171 的同一 16 题，selection digest 必须等于 `5db33945cd22ce71d70d8d97f6af72462d4871aa239559876e79cd1e1edd6b40`；留出与训练 side 在 episode/family/prompt SHA/target 前缀 SHA 上不得相交。
3. 训练 target 取前 `min(128, len)` byte；prompt 不截断（`max_observation_bytes=4096`，逐 byte 证据窗口按既有 512 byte 滚动规则）。空 prompt/target 直接拒绝。
4. `dev_queries=0`；默认模型、旧 parent、已消费 dev 面不改；`can_promote=false`。

## 2. 训练协议

- 模型 `ConditionalLanguageCandidateV2(TaskCoreConfigV2())`，参数必须 970,945；每个 run 新建，bundle 名含 arm/seed。
- 三种子 169/170/171；两臂 `tf`（`ConditionalLanguageTrainerV2`）与 `ss`（`ConditionalLanguageRolloutTrainer`，`max_probability=0.5`、`schedule_steps=256`）。
- 每步 4 题、固定顺序由 `random.Random(seed*1000003 ^ epoch)` 生成，每 epoch 125 步；共 **40 epochs＝5,000 步**；每 epoch 顺序哈希写入报告。
- AdamW lr0.001、wd0、betas/eps 默认、clip1；损失按 batch 内 target byte 平均；`ss` 的采样决定不回传。
- 任何非有限 loss/gradient 立即停止该 run 并保存 partial，不继续。

## 3. 评估

- 每 run 在 0 步、epoch 10/20/40 后测：
  - 16 个确定性训练样本（按 `sha256(episode_id)` 排序后等距取 16）；
  - 16 个留出题。
- 每点测 TF top-1/NLL/完整汉字、自由 128 byte 与前缀匹配；epoch 40 加留出对照：
  - A＝同折下一道留出 target（4 题一折内回绕）；
  - B＝训练集按 `(rank*37+11) % 500` 取的 target；
  - 错配 TF＝留出 prompt × A target 的教师强制。
- 自由生成固定贪心 UTF-8 合法字节、128 byte 上限、遇 `<|end|>` 停止。

## 4. 判据与出口

- R：epoch 40 训练样本自由匹配中位数在至少一臂的 ≥2/3 seed ≥0.50。
- G：至少一臂在 48 个留出读数中 ≥24 个 own ≥0.20 且 ≥A+0.15、≥B+0.15，且命中数比另一臂高 ≥8。
- 出口：G 成立 `PASS_BINDING`；R 成立而 G 不成立 `FAIL_NO_GENERALIZATION`；R 不成立 `INCONCLUSIVE_ROLLOUT`。
- 保持描述：N1-170 的 4 题自由匹配在 epoch 40 的报告值，只描述，不作门。
- 6 个 run 必须全部 complete 才判 R/G；任一 run partial/failed 时聚合只报告可得读数，不判通过。

## 5. 存档、恢复与费用

- 每 run 在更新前保存 `zero.pt`、结束保存 `final.pt`，均在新进程恢复并核对 digest、revision/steps/曝光、下一步 logits 与 15 byte 继续生成；write-once，不覆盖 N1-171/172/173 存档。
- 费用不硬性封顶：按实测 1.2 ms/target byte 估算 2,600 CPU/run、6 run ≈16,000 CPU；单 run 保护阈值 3,200 CPU/5,200 wall，仅用于阻断失控，不用于中途截断正常训练。
- 6 个单线程进程并行；父进程汇总每 run 的真实 CPU/wall，异常 run 单独记录，不影响其他 run。
