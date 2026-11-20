# 23.1 数据合同草案：中文组合世界、三轴留出与 oracle 接口（设计·待冻，零训练）

> 文档状态：[草案] — 2026-11-20（D-20261119-164）。上游 [23 S0](DESIGN-TAIJI_U2_23_S0_COMPOSITION_20261120.md)、[23.0 前置](DESIGN-TAIJI_U2_23_0_PRECONDITION_20261120.md)、现有生成器 [`first_a_support_22_5_v2.generate_examples`](../../scripts/training/first_a_support_22_5_v2.py)。**未准入、未冻结、未运行、零训练、零 check。**

## 1. 目标与范围

把 23 的数据合同定义到**可执行级**：世界生成、中文模板语法、答案空间、三轴交叉留出算法、分区 digest 与 oracle 接口。本卡只写合同，不生成 world、不训练、不读 check/final。

## 2. 新 fixture：`taiji-23-composition-v1`

- 世界：复用 `PhysicalWorld.reset(seed, {"body_count": 3})` 与 `NAMES`（红/绿/蓝三色物体）＋位置词（左/中/右，前/后由渲染深度提供）；
- 每个 world 生成 **4 类问题**（各 1 条，读同一世界快照，允许同世界多问）：

| 类型 | 模板 | 答案空间 | 说明 |
|---|---|---|---|
| Q1 单跳位置 | `{X}现在在哪里？` | 左/中/右/未知 | 复用 first-A v2 已验证面 |
| Q2 单跳关系 | `{X}的{方向}边是谁？` | 物体名集合∪{未知} | 方向 ∈ {左,右,前,后} |
| Q3 两跳关系 | `{X}{方向1}边物体的{方向2}边是谁？` | 物体名集合∪{未知} | 组合阶 ≤2；方向对有序 |
| Q4 来源/时间 | `根据{来源}，{X}在{时刻}时在哪里？` | 左/中/右/未知 | 来源 ∈ {画面,之前记录}；时刻 ∈ {开始时,结束时} |

- **措辞轴**：每个模板 3 组同义改写（如「在哪」/「位于何处」/「在什么位置」），训练只暴露 1–2 组，fit/check 暴露未见组；
- **关系方向轴**：左/右/前/后四方向成对；训练暴露 {左,右}，fit/check 暴露 {前,后}（方向反转留出）；
- **组合轴**：两跳有序方向对（如 左→右 vs 右→左）；训练暴露一组顺序，fit/check 暴露反向与未见对。

## 3. 三轴交叉留出算法（读前冻结）

```
worlds = 2688 个固定 seed 的世界（seed = 23_0000 + index）
每个 world 产出 (q_type, template_group, direction, pair_order) 四元组
泄漏禁止：任何 fit/check/final world 的 (template_group, direction, pair_order)
          不得与 train 中任一条完全相同
切分：
  train: 每 world 的 Q1+Q2(方向∈{左,右}) 各 1 条（512 world）
  fit:   每 world 的 Q2(方向∈{前,后}) + Q3(未见对) + Q4(未见时刻) 各 1 条（128 world）
  check: 每 world 的 Q3(反向对) + Q4(未见来源) 各 1 条（128 world，封存）
  final: 每 world 的 Q3(反向对)+Q4(未见来源)+Q2(未见方向) 各 1 条（128 world，封存）
```

- 每 world 的 4 类问题**共用同一世界状态**（可比性：同一场景下不同读取难度）；
- 分区生成后计算 `partition_digest = sha256(sorted(world_digest, question_digest))`，写入合同；
- 合同冻结后任何模板/方向/顺序改动即作废，须新卡。

## 4. Oracle 接口（23.1 可辨性门）

- `oracle_answer(world_snapshot, question)`：直接读符号世界状态（物体坐标、深度、时间戳），**不经过模型/媒体**，返回答案字符串；
- 可辨性检查：fit 128 的 oracle 准确率必须 **=1.0**（符号真值自洽），训练/推断路径的「全历史 oracle」（模型可读到全部观测）≥0.95；
- 若 oracle <1.0 ⇒ 数据合同 bug（`engineering_failure`），不得用模型调参弥补。

## 5. 与现有资产的关系

| 资产 | 用途 | 是否改动 |
|---|---|---|
| `first_a_support_22_5_v2.py` | Q1 单跳位置模板与 oracle 逻辑 | 不改（历史复现） |
| `taiji/u2_task_episode.py`（NAMES/position_word） | 名称与位置词 | 只读复用 |
| `taiji/native_composition_fixture.py` | 组合结构参考 | 只读参考 |
| 拟 `taiji/native_composition.py` | 23 新 fixture（Q2–Q4 模板与切分） | **新增**（23.1 冻结后） |
| 拟 `scripts/training/run_s2_23_1_20261120.py` | 生成、digest、oracle 检查 | **新增** |

## 6. 验收（23.1 出口）

| 编号 | 条件 | 失败出口 |
|---|---|---|
| C1 | 分区规模：train 512 / fit 128 / check 128 / final 128（以 world 计） | `data_contract_invalid` |
| C2 | 三轴无泄漏：`(template_group, direction, pair_order)` 集合不交 | `data_contract_invalid` |
| C3 | oracle 符号真值 1.0；全历史 oracle ≥0.95 | `engineering_failure` |
| C4 | 4 分区 digest 齐全、可复现（同 seed 两次一致） | `data_contract_invalid` |
| C5 | 中文模板 UTF-8 无乱码、答案空间枚举完整 | `data_contract_invalid` |

- 只读 fit；check/final 不生成到可读位置（生成到封存路径并只记 digest），首次读取即扣；
- 通过 ⇒ `data_contract_frozen`，可进 23.2；失败 ⇒ 按上表出口，不进入训练。

## 7. 不签

零训练、零 check；不签 23 通过/final/C/D/J/模型成型；依赖 23 准入与 23.0 前置；未准入不运行。
