# Seed / Taiji 当前执行状态

> 快照日期：2026-08-30。本文件是“现在做什么”的唯一事实源；历史过程位于 `plans/archive/`，详细阶段定义见 [04_EXECUTION_PLAN.md](04_EXECUTION_PLAN.md)。

## 1. 当前能力状态

| 范围 | 状态 | 可以声明 | 仍不能声明 |
|---|---|---|---|
| W0–W3 Workbench 闭环 | 已完成基线 | 原生 workspace 证据、语言识别/IDE 切换、受控工具执行、Outcome、recovery、checkpoint continuation | 默认写入自治、无限循环、开放域自然语言工具选择 |
| W4–W6 产品边界 | 已完成基线 | Legacy/HF/GGUF/Transformer 退出 cognition 和前端主语义；native facade/客户端真实性 | Legacy 已物理删除、provider 是认知主体 |
| W7-R1 provider watchdog | S0/S1/S2 已完成 | provider artifact digest、健康隔离、checkpoint replay、native-readable packaged 观测 | 外部 provider artifact 的真实客户端轮换已完成 |
| W7-R2 interaction-group | S0/S1/S2 已完成 | 从真实 trace 归因互补/冲突/恢复，不硬编码神经元角色 | 已自动改写 executive、memory 或结构 |
| W7-R3 visual/desktop | S0/S1 + 页面证据完成 | 生命雷达、窄布局、前端/包字节一致、客户端真实状态投影 | Windows 任务栏、托盘、通知、高 DPI 已现场通过 |
| W7-R4 CUDA | `hardware-blocked` | CPU 基线与设备/checkpoint 合同仍有效 | CUDA 性能、数值一致性或自定义 kernel 已验证 |
| W7-R5-S0 学习通道 | 已完成 | 真实 Workbench Outcome 可进入 `record_executive_outcome()`；`learn=False` 冻结，`learn=True` 在线更新；checkpoint 保留计数与选择 | 知识已内化、外挂可删、效应器可注册、开放域自进化 |
| W7-R5A/R5B | G1 合同已冻结、实现未开始 | 两份独立 manifest 已定义 owner、边界、checkpoint、证据、可删性/注册生命周期与回滚 | 任何生产转换器、注册表或效应器能力已开始或完成 |
| W7-R5C 开放域成长 | 合同已冻结、实现未开始 | 结构成长的输入/证据/回滚边界已版本化 | 长期真实任务会自行扩容或进化 |

最新证据数字与报告只在 [IMPLEMENTATION_STATUS_2026_08.md](../../reference/IMPLEMENTATION_STATUS_2026_08.md) 维护。

## 2. 当前阻塞与非阻塞边界

- **R3 Windows shell 为 `tool-blocked`。** Chrome 页面和窄布局证据已通过，但 Computer Use 无法激活 Seed 窗口，不能把桌面背景截图当任务栏/托盘/通知/DPI 证据。工具恢复后补证，不返工已通过页面层。
- **R4 CUDA 为 `hardware-blocked`。** 当前主机没有可用 CUDA；不写自定义 fused/sparse kernel，不用 CPU 结果代替 GPU 结论。
- 两条阻塞线仍影响对应发布声明，但不再冻结无依赖的 R5 CPU/native 合同与实现。R5 的任何进展也不能反向把 R3/R4 标为通过。
- 训练或结构试验开始前必须先通过 checkpoint 保存→关闭→恢复→继续的阻塞 Gate；只在内存中成立的结果不进入路线。

## 3. 仓库收敛状态

- 当前 checkout 为 `main`；`output/` 是未跟踪的现场证据目录，本轮不暂存、不删除。
- `backup-local-20260828` 与干净的 `codex/interaction-group-credit` 已收束并删除；`codex/interaction-group-incremental` 仍附着含 5 个未提交文件的 worktree，未强行删除或混入主线。
- CI 修复已提交为 `21c16b4` 并推送到 `origin/main`；R5-G1 合同与门禁本轮单独提交，推送后复核远端状态。
- `plans/` 没有空目录或 0 字节文件。核心架构讨论留在 active/reference；已完成 Gate 过程和旧执行蓝图已移到 archive。

## 4. 当前唯一下一步

执行 **W7-R5A-S0：纯 DTO 内化转换与确定性 replay 红/绿 Gate**。

R5-G1 已完成合同分离并交付：

1. `plans/manifests/taiji_w7_r5_internalization_v1.json`；
2. `plans/manifests/taiji_w7_r5_effector_registry_v1.json`；
3. 扩展 `tests/test_w7_gate_manifests.py`，先证明缺文件、owner 混合、缺 checkpoint、错误删除执行器或 provider 取得 cognition ownership 时会红，再验证合法合同；
4. 明确两份合同与现有 `taiji_w7_r5_open_domain_growth_v1.json` 的依赖，三者状态互不代替；
5. 合同测试包含合法合同与缺 owner、混合 owner、缺 checkpoint、认知越权、错误删除边界的 red contract。

为什么下一项是 R5A-S0：R5-S0 已接通真实学习输入，R5-G1 已冻结“哪些知识可删、哪些执行器不可删、状态归谁、失败如何恢复”；现在可以只实现内化 DTO 和确定性 replay，不提前改效应器注册表或结构成长。

## 5. 本 slice 明确不做

- 不实现 `taiji/internalization.py`；
- 不重构 `seed_platform/workbench.py` 的硬编码分派；
- 不删除 skill/MCP、Legacy、`codex/interaction-group-incremental` 或 `output/`；
- 不启动训练、不改模型权重、不做 CUDA；
- 不继续视觉美化或用模拟截图关闭 R3；
- 不在同一提交推进 R5A/R5B 的生产代码。

完成 G1 后唯一后继为 [04_EXECUTION_PLAN.md §4](04_EXECUTION_PLAN.md) 的 **R5A-S0：纯 DTO 内化转换与确定性 replay 红/绿 Gate**。

## 6. 更新规则

- 每个 slice 完成后先更新 manifest、实现事实和本文件，再提交；不得并列维护第二个“当前唯一下一步”。
- 若实现与架构合同冲突、checkpoint 不能恢复、red proof 不会红或 CI 新增失败，立即停止功能推进，先修该错误。
- 已完成过程进入 archive；仍影响设计和后续开发的核心需求、所有权、接口合同与未关闭缺口必须留在 active/reference。
