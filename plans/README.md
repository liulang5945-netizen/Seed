# Seed 计划与架构入口

本项目和模型是 **Seed**。**Taiji** 是 Seed 的原生计算基底，承担输入表示、时间状态、上下文、学习、输出、生成和 substrate checkpoint；`seed/` 拥有模型级组合与身份，`neuroplex/` 只作为冻结的 Transformer 基线保留。Seed 只通过 Taiji 公共 API 组合基底，Taiji forward 不调用 `seed/`、`neuroplex/` 或 `transformers`。

命名口径（Seed / Taiji / Legacy NeuroPlex / 历史 `taiji.*` 别名）见 [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) §0 规范词表。

## 当前权威文档

| 文档 | 权威范围 |
|---|---|
| [SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) | 当前唯一执行路线：工程基线、CUDA、容量、训练、机制、产品原生化与公开测试版 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed 模型与 Taiji 基底的所有权、云端实现吸收判定和当前构建上限 |
| [TAIJI_SUBSTRATE_ARCHITECTURE.md](active/TAIJI_SUBSTRATE_ARCHITECTURE.md) | 完整算法：张量、状态方程、tick、局部学习、训练、生成、复杂度、代码映射和反证门槛 |
| [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) | 当前实现状态、实测结果和唯一下一步 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 规范词表、“全面替代而非补丁”的不可回退决策与命名边界 |
| [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](archive/authored/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) | 冻结 Legacy NeuroPlex 的源码事实基线 |

其余 active 文档是记忆、自举、生物类比等专项参考；若冲突，以 Seed 总架构、Taiji 算法规格和当前实现计划为准。

## 当前代码事实

- 正式模型包：顶层 `seed/`；当前明确组合一个 `Taiji` substrate，并拥有 `seed-native-v1` checkpoint envelope。
- 正式基底包：顶层 `taiji/`；不导入 `seed`、`neuroplex` 或 `transformers`。
- 被替代的 Transformer 底层：`neuroplex/layers.py::TransformerBlock`，live 消费点 3 处（`neuroplex/resonance/neuron.py:25`、`scripts/training/train_tinystories.py:26`、`scripts/training/train_tinystories_field.py:32`），由 `tests/taiji_native/test_naming_boundary_contract.py` 强制封闭。
- 原生链：raw-byte sensor → hierarchical predictive fabric ↔ distributed episodic field → 全皮层覆盖稀疏感受器组 → byte motor → action feedback。
- 原生学习：区域预测误差、递归状态误差、运动结果误差和情景 cue→event/readout 的真实边局部 delta；无 optimizer/BPTT。
- 原生训练入口：`seed.datasets` 定义 UTF-8 text/raw-byte 合同；`api/training/recommend.py` 只返回 Taiji 参数预算与容量画像，`/api/train/native` 提供 SSE 在线训练，前端 Seed 模式不再调用 Legacy 睡眠训练。
- 当前可复现实验：Native v7 为 83,841 active learned parameters，byte-cycle accuracy `0 → 94.12%`；N7/N8 上下文与 trace 因果门槛通过；N9 自反馈 128/128；N10 真实按边等价；N11 主动环境末 40 次成功率 `100%`，随机 `50%`，action-lesion `57.5%`；M5 one-shot action recall `87.5%`；M6 12/12 seed 均为 4/4，control 均为 25%，mean gain `+0.75`。
- 旧 `neuroplex.taiji` K/V 原型及 T4/T5 活动文件已删除；Git 历史仍可恢复。
- 现有 9 个 Transformer 成员（含 5 个对话成员）未被改写，只作为离线对照。
- `scripts/archive/` 内 `from taiji.<legacy>` 是历史别名（含义＝`neuroplex`），已确认不重写；判定见 `scripts/archive/README.md`。
- CI 与本地一致性：`cryptography` 已在 `pyproject.toml` legacy extra 与 ci.yml 两处安装行声明（缺失会让 `SecureStorage` 构造失败）；`AuthManager.__new__` 只在初始化成功后写入 `cls._instance`，避免半构造单例被永久缓存成 `AttributeError`；SPA 兜底路由 `include_in_schema=False`，使 OpenAPI 快照不再依赖被 gitignore 的 `frontend/dist` 构建产物。三者共同消除“本地绿、CI 红”的环境漂移。
- CI 门禁约束（踩坑记录）：`black --check .` 是 `test (3.10)/(3.12)` 的早期步骤，一旦失败其后 mypy、pip-audit、8 个 verify 脚本、契约与全量回归**全部跳过**——任何新增文件未过 black 会让整条验证通路失效。覆盖率阈值 `fail_under = 17` 按「全量 tests/ + 全量 source」标定，只跑子集的 job 必须用 `--cov=<pkg>` 显式收窄度量面；`test-windows` 曾因沿用裸 `--cov` 把 `neuroplex/`、`api/` 计入分母而得出 9.19% 假低值，收窄为 `seed/taiji/seed_platform` 后实测 71.27%。
- 编码卫生（已闭环）：`api/__init__.py` 与 `frontend/shoot-fe.cjs` 的 UTF-8 BOM 已清除。BOM 是隐形炸弹——black 走 `tokenize.open` 会静默剥离，CI 因此长绿，但任何 `ast.parse(read_text(encoding="utf-8"))` 都会抛 `invalid non-printable character U+FEFF`。已加 `tests/seed/test_python_sources_have_no_utf8_bom` 守卫（经反向验证：塞回 BOM 立即失败），并要求扫描面 >100 文件以防守卫被悄悄收窄。`tests/seed/test_platform_boundary.py::_imports` 保持 `utf-8-sig` 容错不变——它的职责是查 import，编码问题由专职守卫报错。
- 遗留技术债（不阻塞，明确不修）：`scripts/archive/legacy_convert_dense_model_format.py` 同时存在 BOM 与不可逆 mojibake（第 83 行 `[绯荤粺]` ＝ GBK 的 `[系统]` 被当 UTF-8 写回），在 `utf-8`/`utf-8-sig`/`gbk` 下均无法解析，本已是死文件；它被 black `extend-exclude` 与 coverage `omit` 排除，也排除在 BOM 守卫扫描外。
- 分支策略：仓库**只保留 `main`**。历史上 Dependabot 版本更新累积过 31 条长期分支，已全部删除（对应 12 个 open PR 随之关闭，均为依赖升级，无业务提交丢失）。`.github/dependabot.yml` 三个生态系统的 `open-pull-requests-limit` 统一设为 `0`：该选项只关版本更新，安全更新走独立通道（内部上限 10）不受影响；`groups`/`ignore`/`labels` 全部保留，恢复只需把 `0` 改回 `3`。

## 当前状态与唯一下一步

2026-08-25 路线复核确认：GitHub `main` 远程读写已经恢复；Taiji 核心没有 Transformer/tokenizer/autograd 依赖。容量硬编码第三阶段已完成：`CapacityPolicy` + 参数预算现在同时规划区域/突触结构与 episodic memory 的时间、episode 编码维度，训练画像也会同步放大，旧 policy JSON 保持兼容。Transformer 清理边界第十二阶段已完成：训练推荐、数据集检查、检查点续训和前端 Seed 训练均已迁到 raw-byte Taiji 路径；桌面核心仍不自动安装 Transformer/RAG/Agent 依赖，只有显式 `SEED_ENABLE_LEGACY=1` 才启用 Legacy 清单；运行状态的 life/tools fallback 也已门控；真实 API 的 Legacy 开关矩阵与原生训练 SSE 有通过记录。

R0 的工程收敛改动已经落到 `main`，当前进入统一路线的 S1 产品断点验收：知识库、生命状态、设置、聊天 chip、工作区重命名必须全部有真实行为和对应测试。S1 关闭后进入 S2 质量稳定性，再进入 S3 桌面发布。禁止直接续跑 100M、凭猜测开发 CUDA kernel 或删除 `neuroplex/` 造成产品壳断裂。

M7 已闭合（七项判据全过）：accepted replay 用内生 `cortical_projection` 重建 cue 基底、把 action mode 写入慢通路，`act()` 显著高于 no-replay/content-lesion。阶段 1/2 完成：800K raw-byte 重训（byte_ppl 23.1，面板三组排序正确）、`seed/judge.py` 原生自我评估、A1 同判据验证通过。阶段 3 完成：原生 sleep 调度 + 主题探索环境，A2–A5/B1 五项判据在 800K 成熟检查点上全部 PASS（报告落盘 `reports/seed_a2/a3/a4_a5/b1_*.json`）；机制：`_development_ticks` 生命周期成熟门控、观察性夜晚（零漂移自我维持睡眠）、经验清醒预算封顶。阶段 4/5 完成：产品接入（api/前端/桌面端/移动端远程接入）全仓 108 项绿；超越证据报告见 `reports/seed_phase5_transcendence_20260823.md`。当前诚实边界：byte-level 生成尚未到人工可读。判据见 [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) §6 与 [BOOTSTRAP_CRITERIA.md](archive/authored/BOOTSTRAP_CRITERIA.md)。

## 归档

`archive/` 保存旧架构、审计、实施历史和参考资料。旧 Taiji-0 补丁原型见 [TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md](archive/architecture_design/TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md)，本次云端 Transformer 壳审计见 [TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md](archive/audits/TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md)。归档中的下一步不再有效。
