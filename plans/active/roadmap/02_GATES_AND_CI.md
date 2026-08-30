# Seed / Taiji 当前门禁与 CI 纪律

> 本文件只保留仍然生效的门禁规则，不记录逐次修复日志，也不决定当前下一步。2026-08-29 及以前的完整事故、修复与 Gate 证据已归档到 [SEED_GATE_CI_HISTORY_2026_08.md](../../archive/history/SEED_GATE_CI_HISTORY_2026_08.md)。当前执行入口只看 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md)。

## 1. 证据分层

所有能力必须按同一层级声明，低层证据不能替代高层证据：

| 层级 | 证明什么 | 不能证明什么 |
|---|---|---|
| S0 | 确定性小型机制、反例和边界能够成立 | 真实运行时、真实任务或产品可用 |
| S1 | native adapter / checkpoint replay / sandbox 能恢复并复现 | packaged client、长期开放域行为 |
| S2 | 真实 Workbench、真实 provider 或 packaged client 的现场闭环 | 超出测试任务和声明边界的通用智能 |

每个 Gate 必须有 owner、真实输入、结构化输出、内容或版本 lineage、资源预算、checkpoint、red proof、holdout、lesion、失败隔离和 rollback。缺少其中任一项时只能记为实验，不能写成能力完成。

## 2. 阻塞级不变量

### G1：唯一事实源

- 当前唯一下一步只允许出现在 `03_CURRENT_EXECUTION.md`。
- 代码事实与验证数字只允许在 `plans/reference/IMPLEMENTATION_STATUS_2026_08.md` 维护一份。
- 架构使命和所有权分别由 active 根目录的四份架构合同负责；历史“下一步”不得从 archive 恢复。
- 前端不得推断或补造 Taiji/runtime/provider 状态；缺测字段必须缺失或显式 unavailable。
- `RuntimeEvidenceStrip` 的产品权威入口只有生命状态页；聊天、知识库、训练、Agent、工作台和设置页不得复制该证据条。E2E 路由巡检必须同时验证生命状态页展示、其他页面不展示。

### G2：checkpoint 往返与继续执行

任何训练、在线学习、结构变化、provider 轮换或能力装配开始前，必须先通过：

1. 保存父 checkpoint；
2. 关闭并重新构造运行时；
3. 恢复 checkpoint；
4. 继续至少一步；
5. 对状态、结构 revision、parent/child lineage、预算、tombstone、provider artifact、capability snapshot 和学习计数做等价性断言。

只在内存中成立、恢复后丢 lineage、或最后一步才发现无法保存的结果一律失败。新增可变状态时，`checkpoint()`、`restore()`、旧版本兼容分支和往返测试必须同一提交完成。

### G3：学习与因果证据

- 训练集、holdout 和 lesion 数据必须隔离；holdout 不得产生在线更新。
- 红测必须先证明旧实现会失败，不能只添加永远为真的断言。
- 成功率提升必须同时报告遗忘、恢复时间、资源、跨 seed 稳定性和回滚结果。
- 结构成长只能由真实失败簇、容量压力、遗忘或恢复不足触发，不能由目标规模、UI 按钮或人工神经元角色表触发。

### G4：所有权与边界

- `taiji/` 不依赖 `seed`、`seed_platform`、`neuroplex` 或 Transformers。
- provider 是语言效应器，只实现已经形成的内容，不选择工具、不拥有 cognition。
- Workbench capability 必须来自内容寻址 snapshot；prompt、前端和 provider 不得维护第二份能力表。
- Legacy 保持冻结对照；`SEED_ENABLE_LEGACY=0` 时不得注册 Legacy router 或把旧 HF/GGUF/Transformer 选择器重新暴露到产品。

### G5：CI 必须真的执行

- 依赖版本必须真实存在并固定；本地钩子和 CI 使用同版本。
- 阻塞 job 不得因无真实产物依赖的 `needs:` 被上游失败隐藏为 skipped。
- 每个阈值型门禁必须固定工具版本，解析失败时直接失败，不能静默放行。
- 本地 Windows 通过不替代 Linux/Python 矩阵；CI 红时先核对提交、job 是否执行和平台分支，再判断环境差异。
- 新增代码不得提高核心 mypy、Ruff、API/OpenAPI、native boundary 或安全门禁的债务基线。
- Taiji 核心的内容寻址必须只依赖声明的核心运行时；例如 tensor digest 不得隐式要求可选 NumPy，S1 canary 必须在 3.10、3.12 和 Windows 矩阵真实执行。

## 3. 按改动范围执行的验证矩阵

| 改动 | 最小阻塞检查 | 追加检查 |
|---|---|---|
| 计划/manifest | JSON 解析、manifest contract、计划链接/身份、`git diff --check` | 核对唯一下一步和归档边界 |
| Taiji 核心 | 定向 pytest、Ruff、核心 mypy、native boundary、checkpoint roundtrip | 对应 S0/S1 eval、holdout、lesion |
| Seed/API/Workbench | 定向 pytest、API contract/OpenAPI、Legacy-off、checkpoint continuation | 真实 Workbench S2、失败恢复 |
| 前端 | Vitest、ESLint、Vite build、缺测/错误态 | 真实浏览器窄布局与可访问性 |
| 桌面打包 | 前端字节一致性、PyInstaller clean build、健康 canary | Windows 窗口/任务栏/托盘/通知/DPI 现场证据 |
| 训练 | 先过 G2，再跑短训练保存/恢复/继续 | 长训、跨 seed、遗忘与资源报告 |
| CUDA | CPU 基线仅作参考 | 必须在真实 CUDA 主机做 profiler、数值容差和跨设备 checkpoint |

验证数字只写入 [IMPLEMENTATION_STATUS_2026_08.md](../../reference/IMPLEMENTATION_STATUS_2026_08.md)，本文件不复制会过期的通过数。

## 4. 当前开放门禁

| 工作包 | 当前状态 | 关闭条件 |
|---|---|---|
| W7-R3 Windows shell | `tool-blocked` | 能激活 Seed 窗口后，补齐任务栏、托盘、通知和高 DPI 真实证据 |
| W7-R4 CUDA | `hardware-blocked` | 在真实 CUDA 主机完成 profiler、CPU↔CUDA checkpoint 和数值一致性 |
| W7-R5A 内化 | `contract_frozen / S1-implemented` | S0 DTO/replay 与 S1 原生学习器、holdout、lesion、checkpoint continuation canary 已通过；跨平台 CI 已全绿，仍需真实 Workbench 纵向与可删性 Gate |
| W7-R5B 效应器成长 | `contract_frozen / implementation-not-started` | `taiji_w7_r5_effector_registry_v1.json` + 注册生命周期/快照/回滚 Gate |
| W7-R5C 开放域结构成长 | `contract_frozen / implementation-not-started` | R5A/R5B 真实 evidence 可供触发，且 S0→S1→S2 全通过 |

R3、R4 是独立验证线：它们未通过就不能发布对应声明，但工具或硬件受阻不再冻结与其无依赖的 R5 合同和 CPU/native 实现。任何 R5 结果也不能反向冒充 R3/R4 已完成。

## 5. 停止条件

出现以下任一情况时停止自动推进，先修复并单独提交：

- checkpoint 无法保存、恢复或继续；
- owner、依赖方向或删除边界不清；
- red proof 不会红、holdout 被训练污染、或 lesion 不影响结果；
- CI 新增失败、job 被跳过、测试只在单平台假绿；
- 前端或 provider 成为第二认知源；
- 需要改变“语言 provider 不参与 cognition”或“Legacy 只作冻结对照”的架构边界；
- 需要物理删除外部知识、效应器或历史数据，但尚无可恢复证据与明确授权。
