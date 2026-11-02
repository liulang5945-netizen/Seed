# 10.1：T3 自然目标与有限行动·范围冻结（不授予训练）

> 文档状态：[合同] — owner 已于 2026-11-01 批准立项。本合同只冻结 10.1 的环境、动作接口、终态评分与三条自检判据；**不批准训练、不涉及模型能力**。

日期：2026-11-01。对象为 R021 参考图之外的**独立任务环境**；`taiji/living_core.py` 字节保持不变。

## 1. 环境（声明）

- **世界**：两个对象（蓝色杯子、红色印章）分布在三个位置（木桌、窗台、左盒）。
- **可观察状态**：每个对象当前所在位置（`object → place` 的映射）＋已用步数。宿主不向 runtime 提供目标、正确性标签或评分。
- **自然 Goal**：一句中文，例如「把蓝色杯子放到窗台」或「红色印章不要留在左盒」。Goal 只声明世界应满足什么，不声明怎么做。
- **动作集**：`move(object, place)`，对每个对象、每个「非当前位置」生成一个动作。2 对象 × 3 位置 ⇒ 每步至少 2 个合法动作。
- **动作 ID 不含正确性**：每步用声明种子把动作打乱后编号为 `a0…an`；同一 (object, place) 在不同步拿到不同 ID。
- **终态评分（独立于动作序列）**：`score = 1[所有 Goal 条件满足] − 0.01 × 步数`。只读终态与步数，不读路径。
- **成本**：每步 1 单位；评分已含线性代价。

## 2. 三条自检判据（先于任何模型；全部可机检）

1. **换 Goal 改变最优行为**：在**同一初态**上，两个不同 Goal 的最优动作序列（以 `(object, place)` 表示，不用被打乱的 ID）必须不同。若相同 ⇒ `goal_insensitive`。
2. **分区完整**：夹具集至少覆盖 3 组不同 (初态, Goal) 组合，其中至少一组共享初态、至少两组共享 Goal；缺任一项 ⇒ `partition_incomplete`。
3. **捷径不能满分**：声明四条捷径策略——`first_legal`（总取第一个合法动作）、`lowest_id`（总取编号最小）、`fixed_object`（只动第一个对象）、`cycle`（固定两步循环）。每条策略在**至少一个**夹具上必须**拿不到**最优分；且没有任何一条在**全部**夹具上达到最优 ⇒ 否则 `shortcut_can_solve`。

## 3. 预算、出口与交付

- 预算：≤300 CPU 秒 / 600 墙钟秒；超限 `budget_stop`。状态空间只有 9 个状态，穷举最优是精确解，不用采样。
- 出口：`passed_10_1_environment_only`／`goal_insensitive`／`partition_incomplete`／`shortcut_can_solve`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/native_goal_world.py`、`tests/taiji_native/test_goal_world.py`、`scripts/training/verify_taiji_goal_world_10_1.py`、`reports/taiji_core_goal_world_10_1_20261101.json`、MAINLINE/03 同步。
- **通过只表示**这个任务环境本身满足 10.1 的三条设计判据；**不表示**任何模型能解它，也不表示 10.2/10.3 准入。
