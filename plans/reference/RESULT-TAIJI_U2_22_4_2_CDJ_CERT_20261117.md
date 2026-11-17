# 22.4.2.4 C/D/J 零训练 fixture 认证结果（C 12/12、J 54/54、D 36/36）

> 卡号：`22.4.2.4`｜日期：2026-11-17｜判定：**`c_d_j_fixtures_valid`**（C/J/D 全族合法历史 oracle 均 ≥0.95；D 空白历史控制 0.0）｜性质：**零训练 fixture 认证**，不是模型能力/父结果。
> 依据：[缺口审计](REVIEW-TAIJI_U2_22_4_PARENT_ADMISSION_GAP_20261117.md)（D-20261117-21／24）、[计划 §22.4.2.4](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)、[世界规格](SPEC-TAIJI-U2_WORLD_22_4_2_20261112.md)。
> 纪律：**0 模型前向、0 optimizer step、0 项目数据读取、0 封存读取**；本次（v2）实耗 **52.5469 CPU**。首份 C/J-only 报告与结果卡保留为 v1 历史，本卡的 v2 读数取代 v1 的"D 未定义"判定。

## 1 读数

| 族 | fixture | 正确/分母 | 说明 |
|---|---|---|---|
| A | `temporal_fixture` initial/current | **32/32** | 引用既有[报告](../../reports/taiji_u2_temporal_world_22_4_2_20261112.json)，不重跑 |
| B | `temporal_fixture` previous | **32/32** | 同上 |
| U | `temporal_fixture` unknown | **32/32** | 同上；空白 previous 天花板 0.5、错时间 0.5 |
| C | `acquisition_fixture` inspect/wait | **12/12** | 3 seed×2 initial_x×2 known；oracle 最优动作与真值答案同时正确 |
| J | `joint_fixture` public oracle | **54/54** | 3 seed×18 checks（3 会话×3 实体×2 时间意图） |
| D | `capacity_arms` capacity pressure | **36/36** | 3 seed×12 交错检查；遮挡臂 **36/36**；空白控制 **0.0**；动力学臂 3/3 seed 通过 |

**D 明细（owner 2026-11-17 批准定义）**：
- 6 个实体观测、48 轮真实世界、只释放 **2 帧公开 RGB**、容量 **4 槽**、12 检查/seed；`observed_facts=12`（6 实体×2 帧），压力比 **3.0**。
- **合同适配（必须披露）**：世界合同 `physics.objects=[1,3]` 最多允许 **3 个可见物体/世界**，`PhysicalWorld.reset` 对 6 bodies 抛 `invalid_body_count`。因此 D 用 **2 个独立三体子世界**（seed 与 seed+500）合成 6 个实体观测，而不是单个六体世界；两个子世界的检查 **交错排列**，容量 4 少于 12 个观测事实，保留容量压力语义。这是一次由世界合同强制的等价适配，不是把 6 实体要求降为 3 实体。
- standard 臂：两个子世界各 3 实体 × current/previous，共 12 检查/seed；3 seed 全对 **36/36**。
- occlusion 臂：第二个子世界遮挡 1 个实体的末帧，expected/answer 均走"未知"；3 seed 全对 **36/36**。
- blank 控制：标准臂检查全部为可判定 expected；对空历史调用 public oracle 一律回答"未知"，命中 **0/36 = 0.0 ≤ 0.60**。
- dynamics 臂：同 seed、同首帧外观，mass/drag `1.0/1.0` 对 `2.0/2.0`；首帧逐位相等、末帧答案出现差异（中间实体跨过"左边/中间"边界），3/3 seed 通过。
- 课程：每个子世界 48 tick（1 次 elapsed 首帧 + 47 次单目标 gentle act），三相位推动，避免三体接触求解不收敛。

**C 明细**：seed 124／125／126 × initial_x ±0.55 × known True/False；每个 fixture 的 `best` 动作等于 utility 最大者，且该动作答案等于真值；12/12。
**J 明细**：seed 124／125／126 各 18 项（3 会话 48 轮，3 实体 × current/previous × 3 个检查轮），public oracle 54/54；冷模型 0/18 **不是**可学性证据（D-23），本卡不引用为负证据。

## 2 判定

- **C/D/J 全族的合法历史 oracle 均达到 ≥0.95 门**（C 1.0、J 1.0、D standard 1.0、D occlusion 1.0），fixture 配对与分母已登记；A/B/unknown 32/32 与空白/错时间控制 0.5 沿用既有报告。
- **D 空白历史控制 0.0 ≤ 0.60**；动力学臂证明"同外观不同动力学"配对有效。
- **能签**：22.4.2.4 的 C/D/J fixture 合法性、oracle 可辨性、分母登记完成；可进入 22.4.2.5。
- **不签**：任何父/能力/check②/22.6；fixture 通过只说明任务可辨，不是模型可学，更不是 `accepted_parent`。22.4 的七项交付仍缺第 2（真实任务族数据 recipe 及分区 digest）与第 7（accepted_parent），22.4.3/22.4.4 未完成。

## 3 唯一后继

**22.4.2.5 干预矩阵**：按计划做动作因果／经历必要／跨模态干预的零训练核验，并登记分母；之后 22.4.2.6 生成器入口 → 22.4.3 分区 → 22.4.4 完整定价 → 22.4 准入 `accepted_parent`。在 22.4 准入前：0 训练新增、0 能力、0 父；check② 不读、22.6／22.7 不解锁。
