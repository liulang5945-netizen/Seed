# 10.1b：任务加深——位置容量与阻挡（不授予训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「加做 10.1b 加深任务」。本合同给 10.1 的世界加一条容量规则，使"往目标挪一格"的贪心不再最优；**不涉及模型、不授予训练**。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。

## 1. 改动（一条规则 + 两条夹具）

- **位置容量 1**：一个位置只能放一个对象；目标位被占时，必须先把它腾开才能放入。实现为 `legal_moves(..., occupy=True)`／`step(..., occupy=True)`，**默认 `False` 保持 10.1 的旧路径逐位不变**。
- **两条新夹具**：`g1`（杯子在木桌、印章在窗台，目标杯子→窗台）与 `g2`（杯子在窗台、印章在左盒，目标杯子→窗台）。g1 必须先腾出窗台。

## 2. 判据

1. **旧路径不变**：`occupy=False` 下，四条原夹具的穷举最优仍为 f1/f2/f4=0.99、f3=1.0，四条捷径仍都不能通吃 ⇒ 否则 `default_path_changed`。
2. **10.1 三判据在容量规则下仍成立**：换 Goal 改变最优动作、分区完整（≥3 组、≥1 共享初态、≥2 共享 Goal）、四条捷径都不能通吃（`occupy=True`）⇒ 否则沿用 10.1 的出口名。
3. **规划能分开**：`occupy=True` 下，深度 2 的 rollout 规划器在**全部夹具上不低于**贪心，且**至少一条严格更高** ⇒ 否则 `no_planner_separation`。这一条是 10.1b 存在的理由。
4. **冲突确实发生**：g1 上贪心的终态分必须低于最优 ⇒ 否则 `blocking_not_effective`。

## 3. 预算、出口与交付

- 预算：≤300 CPU 秒 / 600 墙钟秒；超限 `budget_stop`。
- 出口：`passed_10_1b_environment_deepened`／`default_path_changed`／`goal_insensitive`／`partition_incomplete`／`shortcut_can_solve`／`no_planner_separation`／`blocking_not_effective`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/native_goal_world.py` 的 `occupy` 参数与 g1/g2、`scripts/training/verify_taiji_goal_world_10_1b.py`、`reports/taiji_core_goal_world_10_1b_20261101.json`、MAINLINE/03 同步。
- **通过只表示**加深后的环境满足三条 10.1 判据、且规划与贪心在该环境下可分；**不表示**任何模型能规划，也不追认 10.3 通过——10.3 需在加深后的环境上重测。
