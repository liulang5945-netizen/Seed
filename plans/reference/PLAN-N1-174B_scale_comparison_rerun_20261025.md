# N1-174B：规模对照的修正重跑（同一设计，修正防失控阈值）

日期：2026-10-25，Asia/Shanghai。状态：**实施前冻结**。承接[PLAN-N1-174](PLAN-N1-174_scale_comparison_20261025.md)。N1-174 的六个 run 全部在 epoch 23–24 被脚本内的 3,200 CPU「防失控阈值」截断为 PARTIAL，而试跑实测 40 epochs 约需 3,640 CPU/run——阈值设得低于正常成本，属本包工程错误，不是模型结论。N1-174B 保持数据、模型、两臂、种子、批次、40 epochs 与判据完全不变，只修正阈值并增加 epoch 20/30 的耐久 checkpoint。

## 1. VISION 条款对账

与 [N1-174 §1](PLAN-N1-174_scale_comparison_20261025.md) 相同：§89 区分 TF 与自由生成，§81 用错配对照检验证据读取，§93 保持面独立。本包不新增结构、不改数据或判据。

## 2. N1-174 截断事实（保留为证据，不重写）

- 六 run 均在 3,200 CPU / 约 3,250 wall 处停止，状态 `partial`，无 final checkpoint；
- 已留下的 epoch 0/10/20 读数（六 run）：epoch 20 时 tf 臂训练样本 TF 中位数 .849/.867/.877、训练自由匹配中位数 **0/0/.006**；ss 臂 TF .642/.613/.604、自由 **.004/0/0**；留出自由均在 0–.003。
- 这些读数说明「到 20 个梯度贡献/题为止，500 题规模仍未抬起自由滚动」，但不足以按预注册判 R/G；正式判定由 N1-174B 的完整 40 epochs 给出。

## 3. 修正项（仅工程）

1. 单 run 防失控阈值改为 **5,000 CPU / 7,000 wall**（实测预算约 3,700，留 35% 余量），不再中途截断正常训练。
2. 在 epoch 20、30 各保存一份耐久 checkpoint 并在新进程恢复核对；zero、epoch20、epoch30、final 四份存档均 write-once。
3. 命名空间独立：报告 `reports/taiji_n1_plan174b_scale_20261025_*`，checkpoint `data/checkpoints/n1_plan174b_scale_20261025/`；不改 N1-174 的 PARTIAL 报告与 checkpoint。
4. 其余训练/评估/判据逐字沿用 N1-174：500 题（324 family）、16 留出题（`5db33945…`）、4 题/步、40 epochs、两臂 `tf`/`ss`、种子 169/170/171、epoch 10/20/40 评估、A/B 错配对照、R/G 判据。

## 4. 判据

- R：epoch 40 训练样本自由匹配中位数在至少一臂的 ≥2/3 seed ≥0.50。
- G：至少一臂在 48 个留出读数中 ≥24 个 own ≥0.20 且 ≥A+0.15、≥B+0.15，且命中数比另一臂高 ≥8。
- 出口：G 成立 `PASS_BINDING`；R 成立而 G 不成立 `FAIL_NO_GENERALIZATION`；R 不成立 `INCONCLUSIVE_ROLLOUT`；任一 run 未 complete 记 `PARTIAL`。

## 5. 费用评估

- 按试跑实测 0.72 CPU/步、125 步/epoch：40 epochs ≈ 3,640 CPU/run；6 run ≈ 21,800 CPU，加评估/四份恢复 ≈ **22,500 CPU**。
- 六进程并行，预计 wall **70–90 分钟**；不设整体硬上限，只保留单 run 防失控阈值；工程异常只影响对应 run，其余继续。
- N1-174 已消耗约 19,224 CPU（六个 partial run）；本次重跑是修正工程阈值的必要成本，合计约 4.2 万 CPU。相对本项目既有训练量级仍属合理，并换取一次完整的规模判定。

## 6. 落地顺序

1. `scripts/training/calibrate_taiji_n1_plan174b_scale.py`（由 174 脚本复制，仅改阈值/路径/中间存档）；
2. `--all` 跑六 run，`--aggregate` 判 R/G；
3. 写 [N1-174B 结案](PLAN-N1-174B_execution_closeout_20261025.md)，同步 03/10/黑板，跑门禁并提交。
