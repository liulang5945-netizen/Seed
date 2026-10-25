# SPEC-N1-170：逐字节证据候选 v2 的工程与标定判据

日期：2026-10-25。真实更新前冻结，承接[PLAN-N1-170](PLAN-N1-170_byte_evidence_architecture_20261025.md)。本包只验证工程与同一 64 batch 口径下的结构效果，不是泛化、正式算法胜负或默认采用。

## 1. 状态与图不变量

1. `TaskStateV2` 在 v1 字段外新增 `byte_memory`（N×width，N≤512）与 `byte_refs`（逐 byte 来源跨度）；`workspace` 16 槽继续保留，原始 byte 仍在 `observations`，不得静默截断或改写原文。
2. 逐 byte 记忆按观测顺序追加，超过 512 byte 按固定滚动窗口丢弃最旧 byte，并在 refs 中保持一致；观察总量仍受 `max_observation_bytes=4096` 约束。
3. `next_logits` 只看到 prompt 与本步之前的 target/已生成 byte；TF 与自由生成调用同一个入口；无 renderer 补答案。
4. 位置表为 512×64，key 侧加源 byte 位置、query 侧加已生成步数位置；注意力只对有效源位置归一化；位置越界显式拒绝。
5. 状态交换（换入另一题的 byte memory）必须改变实际 logits；仅改元数据 ID 不得改变 logits；交换后恢复原状态必须复原。
6. 训练路径不 detach；编码、keys/values、位置表、query、decoder、输出头都必须有可观测梯度；对编码路径 detach 后对应梯度消失。

## 2. 存档与恢复

- checkpoint 格式 `taiji-conditional-language-trainer-v2`，含 config、bundle、参数 revision、参数、AdamW、RNG、steps/曝光、TaskStateV2、课程 cursor 与 digest。
- 零步与一次更新后的存档都在**新进程**恢复，完整 digest、下一生成步与下一次合法更新一致。
- v1 候选的 state/checkpoint（`task-state-v1`/trainer v1）必须被显式拒绝，不得按 v2 载入，也不得修改磁盘上的 N1-169 存档。

## 3. 参数与容量

- 参数恰为 **970,945**，≤1,000,000；新增仅 512×64 位置表（+32,768），不增加槽、隐藏宽度或 decoder 层数。
- 默认模型、旧 parent checkpoint、旧 native 入口与 `adapter` 默认路径的字节不变。

## 4. 标定口径与判据

- seed169、同 4 个 train episode（N1-169 选择顺序）、prompt 不截断、target 前 128 byte、每 batch 495 byte、64 batch、31,680 暴露；AdamW lr0.001、wd0、betas 默认、clip1。
- 0/1/8/32/64 步记录：逐题 TF top-1、NLL、完整汉字正确数、自由 128 byte 生成与前缀匹配、TF 与自由生成使用的不同源 byte 位置数。
- 主判据：TF 均值 ≥0.681；自由匹配 ≥2/4 题 ≥0.10；TF 使用 ≥8 个不同位置 ≥2/4 题。任一不达按 PLAN §5 分支记录，不降门、不加步数。
- 费用 ≤1,200 CPU、≤1,800 wall；触顶保留 partial、failure 与真实费用。

## 5. 失败出口

- 工程失败（schema/恢复/身份/梯度/默认不变）先修可逆实现；修复不得刷新 N1-169 证据。
- 资源/数值失败保存 failure/partial 与费用，不自动换种子、学习率或结构续跑。
- 0 dev、0 reference、0 N1-166；`can_promote=false`；不称语言能力通过。
