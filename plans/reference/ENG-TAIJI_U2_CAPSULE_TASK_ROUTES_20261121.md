# 工程卡：capsule_v2 加法式 task_routes（任务保护可继承）

> 文档状态：[工程] — 2026-11-21（D-20261121-250）。上游 [参数保护 3 seed](RESULT-TAIJI_U2_S1RT_PROTECTION_20261121.md)。**零 check/final。**

## 1. 改动（`taiji/u2_selective_units.py`）

- `SelectivePersistentGraph` 新增 `task_routes: dict[str, list[int]]` 与：
  - `set_task_route(name, indices)`（越界/空/重复拒绝）；
  - `task_route_mask(name)` 返回 `[n_units]` 0/1 掩码；
  - `export_task_routes()`；
- `build_capsule_v2` 载荷新增 `"task_routes"`；`validate_capsule_v2` 校验类型/范围/重复；`apply_capsule_v2` 恢复路由；
- **旧胶囊兼容**：缺少该字段时按空路由载入（测试覆盖）。

## 2. 测试（`tests/taiji_native/test_u2_selective_units.py`）

新增 3 项：路由往返＋掩码正确、旧胶囊无字段可载、非法索引拒绝；**全量 14 passed**。

## 3. 判读

- 参数级保护机制现在可随 `capsule_v2` **继承**（任务→单位子群映射可恢复）；
- 仍不构成协同/生长收益证据；零 check/final；不签模型成型/final/C/D/J。
