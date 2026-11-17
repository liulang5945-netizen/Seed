# 22.4.2.4 C/D/J 零训练 fixture 认证结果（C 12/12、J 54/54、D 未定义）

> 卡号：`22.4.2.4`｜日期：2026-11-17｜判定：**`fixture_invalid_d_undefined`**（C/J oracle 通过；D capacity pressure 无冻结定义，阻塞 22.4.2.4 完成）｜性质：**零训练 fixture 认证**，不是模型能力/父结果。
> 依据：[缺口审计](REVIEW-TAIJI_U2_22_4_PARENT_ADMISSION_GAP_20261117.md)（D-20261117-21／24）、[计划 §22.4.2.4](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)、[世界规格](SPEC-TAIJI-U2_WORLD_22_4_2_20261112.md)。
> 纪律：**0 模型前向、0 optimizer step、0 项目数据读取、0 封存读取**；实耗 **13.375 CPU**。

## 1 读数

| 族 | fixture | 正确/分母 | 说明 |
|---|---|---|---|
| A | `temporal_fixture` initial/current | **32/32** | 引用既有[报告](../../reports/taiji_u2_temporal_world_22_4_2_20261112.json)，不重跑 |
| B | `temporal_fixture` previous | **32/32** | 同上 |
| U | `temporal_fixture` unknown | **32/32** | 同上；空白 previous 天花板 0.5、错时间 0.5 |
| C | `acquisition_fixture` inspect/wait | **12/12** | 3 seed×2 initial_x×2 known；oracle 最优动作与真值答案同时正确 |
| J | `joint_fixture` public oracle | **54/54** | 3 seed×18 checks（3 会话×3 实体×2 时间意图） |
| D | capacity pressure | **未定义** | 规格/计划无冻结定义，阻塞 22.4.2.4 |

**C 明细**：seed 124／125／126 × initial_x ±0.55 × known True/False；每个 fixture 的 `best` 动作等于 utility 最大者，且该动作答案等于真值；12/12。
**J 明细**：seed 124／125／126 各 18 项（3 会话 48 轮，3 实体 × current/previous × 3 个检查轮），public oracle 54/54；冷模型 0/18 **不是**可学性证据（D-23），本卡不引用为负证据。

## 2 判定

- **C/J 的合法历史 oracle 已达到 ≥0.95 门**（C 1.0、J 1.0），fixture 配对与分母已登记；A/B/unknown 32/32 与空白/错时间控制 0.5 沿用既有报告。
- **D 仍缺冻结定义**：`22.4.2.4` 要求 A/B/C/D/J 全族；缺 D 即 `fixture_invalid`，**不能签 22.4.2.4 完成**，也不能进入 22.4.2.5／22.4.3／22.4.4／`accepted_parent`。
- **不签**任何父/能力/check②/22.6；C/J 通过只说明 fixture 可辨，不是模型可学。

## 3 唯一后继（需 owner 定义 D）

给 **D capacity pressure** 一个冻结定义，再重跑 C/D/J 认证。执行者建议的最小定义：

- **任务**：6 个实体、48 轮真实世界、只释放 **2 帧公开 RGB**（首帧与末帧）、查询全部 6 个实体的 current/previous（12 检查/seed）；
- **容量压力**：声明工作记忆容量 **4 槽**（少于 6 实体），要求选择性保留/压缩；查询时对未保留实体允许回答"未知"，但全分母计分；
- **Oracle**：只读合法全历史（公开帧＋执行动作），current/previous 目标 ≥0.95；空白历史控制 ≤0.60；
- **反事实**：同外观不同动力学、遮挡/缺失各一臂；未过即 `fixture_invalid`；
- **预算**：同 18 题/64 字节范围，120 CPU／240 wall 硬帽；0 训练。

owner 可批准此定义、修改容量/实体数/可见帧数，或指定 D 的另一种含义。定义冻结后，我写 D fixture＋重跑 A/B/C/D/J 全分母认证；在 D 定义前 **0 训练新增**、0 能力、0 父；check② 不读、22.6／22.7 不解锁。
