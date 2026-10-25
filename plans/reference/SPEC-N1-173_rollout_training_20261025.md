# SPEC-N1-173：scheduled sampling 的实现、存档与判据

日期：2026-10-25。真实更新前冻结，承接[PLAN-N1-173](PLAN-N1-173_rollout_training_20261025.md)。只验证「自喂入训练能否让自由滚动可判、并暴露留出绑定」；不是正式算法胜负、不是 L2、不是默认采用。

## 1. 前向与采样不变量

1. 前向与 N1-170/171/172 共用同一个 `ConditionalLanguageCandidateV2`（参数 970,945），不新增可学习参数、不改结构。
2. 每个目标位置先算 `next_logits`；以 `p = min(0.5, (已更新步+1)/256)` 决定输入：掷骰小于 p 时取**当前 UTF-8 合法字节**中 logits 最大者，否则取目标字节；损失只对目标字节计算。
3. 采样决定不回传梯度；被选字节经既有 embedding/decoder 进入图。`p=0` 时必须与 `teacher_forced` 的 logits 完全一致。
4. 采样使用模型自带 `generator`（seed 169），RNG 状态随 checkpoint 保存；同一存档恢复后 120 步之外的下一次更新必须逐位一致。

## 2. 存档

- 新格式 `taiji-conditional-language-rollout-trainer-v1`：config、bundle、parameter revision、参数、AdamW、RNG、steps/曝光、TaskStateV2、cursor、`max_probability`、`schedule_steps`、digest。
- 与 v2 trainer 格式互不接受（v2 拒绝 rollout 格式，rollout 拒绝 v2 格式），不得伪装兼容。
- 每份存档在新进程恢复并核对 checkpoint digest、revision/steps/曝光、下一步 logits digest 与 15 byte 继续生成。

## 3. 训练与评估口径

- 同一 16 题与折 0/1；selection digest 必须等于 `5db33945cd22ce71…`；每折 12 训 4 留出、target 前 128 byte、AdamW lr0.001、wd0、clip1。
- 两折各固定 120 次整批更新（无早停），每折 147,456 target byte。
- 0 步读数共享；120 步测该折 16 题 TF/自由 128 byte，留出题加对照 A（同折下一道留出 target）、B（训练 target `(3i+1)%12`）与错配 target TF。
- 自由生成仍用贪心 UTF-8 合法字节，128 byte 上限，遇 `<|end|>` 提前停止。

## 4. 判据与出口

- R：两折训练题自由前缀匹配中位数均 ≥0.50。
- G：R 达成且 8 个留出题中 ≥4 题 own ≥0.20 且 ≥A+0.15、≥B+0.15。
- 出口：G 成立 `PASS_BINDING`；R 成立但 G 不成立 `FAIL_NO_GENERALIZATION`；R 不成立 `INCONCLUSIVE_ROLLOUT`。
- S：训练/留出自由匹配分布、own/错配 TF、A/B 均值、不同自由输出数、TF 与损失（只报告）。

## 5. 失败与资源出口

- 工程失败（选择 digest、抽样不变量、存档/恢复、来源变化）先修可逆实现；修复不得改判据或选择。
- 费用上限 500 CPU / 1,000 wall（内部 480/940）；触顶保存 partial/failure 与真实费用，不续跑、不换配方。
- 报告 write-once；已存在即拒绝重跑。0 dev、0 reference、0 N1-166；`can_promote=false`。
