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
| W7-R5A/R5B | R5A-S2-A 已完成；R5B 仍为 G1 合同基线 | R5A 已具备 DTO、内容寻址 replay、原生学习器、holdout/lesion、生命周期和 checkpoint；真实 Workbench Outcome 只能经当前 evidence 与重投影 affordance 进入 Taiji-owned 候选；R5B 仅有独立合同 | 真实 Workbench 纵向可删性、效应器注册表、开放域成长 |
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
- CI 基线由 `b6d1bf2` / 远端运行 `33295880356` 完成过全量验收；本轮 `9f30eb4` 接入 S1 canary 后，运行 `33298105636` 暴露了 Linux/Windows 最小 torch 环境缺 NumPy 时 `tensor.numpy()` 的跨平台失败。`513cb1f` 已改为纯 PyTorch 字节视图并补充无 NumPy contract；远端运行 `33298754868` 的 7 个 job 已全部成功。
- `plans/` 没有空目录或 0 字节文件。核心架构讨论留在 active/reference；已完成 Gate 过程和旧执行蓝图已移到 archive。

## 4. 当前唯一下一步

执行 **W7-R5A-S2-B：真实 Workbench 纵向 holdout、lesion、checkpoint recovery 与可删候选 Gate**。

R5-G1 与 R5A-S0 已完成并交付：

1. `plans/manifests/taiji_w7_r5_internalization_v1.json` 与 `plans/manifests/taiji_w7_r5_effector_registry_v1.json`；
2. `taiji/internalization.py`：不依赖 `seed_platform` 的 Outcome/evidence DTO、内容 digest、train-only replay、生命周期和五项因果门控；
3. `tests/taiji_native/test_internalization_contract.py`：未 grounding、缺 reward、越界、provider/capability 文本、holdout 写穿、重复 evidence、未通过 causal gate、checkpoint resurrection 的 red proof；
4. `tests/test_w7_gate_manifests.py`：合同边界与 R5A/R5B 分离关系；
5. R5A S0/S1 证据为 S0 定向测试 `14 passed`、S1 定向测试 `5 passed`、原生 canary `gate.passed=true`，以及本地 checkpoint/holdout/lesion 检查；R5B 与 R5C 仍未实现。

R5A-S2-A 已完成：`api/seed_runtime.py` 只会在当前、校验过的 `workbench.evidence`、同一 capability snapshot 和由该 evidence 重投影出的 grounded successor affordance 同时成立时，创建 `GroundedOutcomeEvidence`。运行时不拥有 replay、learner 或 lifecycle 的写权；缺失/陈旧 snapshot 或非当前 affordance 全部 fail-closed。定向用例在真实只读 workspace 上通过，并已回归完整 Workbench 合同 `44 passed`。

为什么下一项是 R5A-S2-B：S1 已在父 checkpoint 的原生 trial 上证明 grounded feature 能改善独立 holdout，并且 feature、grounding、retention、checkpoint 控制成立；S2-A 现在把输入接到真实 Outcome。下一步必须用不参与训练的真实 Workbench 任务组合完成纵向 Gate，才讨论外挂的可删候选；不提前改效应器注册表或结构成长。

## 5. 本 slice 明确不做

- 不执行物理删除或外部 artifact tombstone 提交；S2-B 只产生可恢复的候选与 Gate 证据；
- 不重构 `seed_platform/workbench.py` 的硬编码分派；
- 不删除 skill/MCP、Legacy、`codex/interaction-group-incremental` 或 `output/`；
- 不启动训练、不改模型权重、不做 CUDA；
- 不继续视觉美化或用模拟截图关闭 R3；
- 不实现 `seed_platform/capability_registry.py`，不在同一提交推进 R5B/R5C 的生产代码。

完成 S1 后唯一后继为 [04_EXECUTION_PLAN.md §4](04_EXECUTION_PLAN.md) 的 **R5A-S2：真实 Workbench 纵向证据与可删性边界**。

## 6. 更新规则

- 每个 slice 完成后先更新 manifest、实现事实和本文件，再提交；不得并列维护第二个“当前唯一下一步”。
- 若实现与架构合同冲突、checkpoint 不能恢复、red proof 不会红或 CI 新增失败，立即停止功能推进，先修该错误。
- 已完成过程进入 archive；仍影响设计和后续开发的核心需求、所有权、接口合同与未关闭缺口必须留在 active/reference。
