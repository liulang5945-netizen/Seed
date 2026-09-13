# Seed / Taiji 技术债登记册

## 最新状态补充（2026-09-13，B0 完成后）

- DEBT-A1/A2 已结项，见下方修复记录；原“只登记不修复”是建册时范围，不应把已完成修复写回未解决。
- 30 失败/788 用例是 aa124f52 的历史基线；28 个 SystemExit 仍待定位，不是当前 HEAD 重测计数。
- 下文引用图论证仅能缩小直接依赖范围，不能证明间接状态、动态导入、文件和环境污染不存在；失败归属须结合可复现顺序、父提交对照与栈证据。
- DEBT-G1/G2/G3 的数量和路径是旧快照；后续 Git 修复见[路线 A 报告](../../../reports/M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_RESULT_20260913.md)。本轮未做 fsck 或清理；继续禁止未经确认 gc/prune、删除备份。
- 新研究阻塞（已由 B0 审查并给出结论）：路线 B 的收益参照不一致、任务协作上界与门禁语义差异。B0 复算 32/32 一致、三种候选参照全部不可达 ⇒ 暂停正式训练，先改任务与估计目标。见[B0 设计包](../../reference/M5_B0_MEASUREMENT_AND_REACHABILITY_AUDIT_20260913.md)。
- **CI 命令级基线（2026-09-13，B0 建立）**：本地 `ruff check .` 原有 1 项 `I001`
  （`tests/taiji_native/test_p5_2c_triple_prime_representation_repair_gate.py` 的导入顺序），
  即两条 Linux CI 的 Ruff 失败原因；B0 已修复，现为 **All checks passed**。
  B0 新增测试 14 passed、目标集八个文件 95 passed。**全量套件本轮未跑**，28 项 SystemExit 未定性；
  不把局部通过当全仓绿。远端 workflow 本轮未查询（`gh` 未认证）。

以下保留建册时的观察、命令和修复记录；现行顺序由[当前推进方案](03_CURRENT_EXECUTION.md)决定。

> 建立：2026-09-13。登记基线：`aa124f52`。
> **本文只登记与量化，不修复。** 修复顺序由 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) 决定；
> 主线（P5.2c 未见组合迁移及其后续）收尾后才进入本文的处置阶段。
> 纪律：登记项不得在未修复的情况下被当作「已解决」；每项必须保留可复现命令与观测基线。

## 0. 为什么单独建册

2026-09-13 修复 P5.2b 零步缺陷时发现，`tests/taiji_native/` 全量运行有大量失败，但这些失败
**无法归因于当次改动**（论证见 §1）。这类既有债务与主线实验证据混在一起时，会污染
「某次改动是否引入回归」的判断。因此把它们与主线解耦，单独建册、单独排期。

**登记册的用途**：任何一次改动后，若全量套件出现失败，可对照本册判断「是否属既有债务」，
而不必重新做一次归属论证。

## 1. 归属论证方法（可复用，重要）

判定「全量失败是否由本次改动引入」，**不要靠反复重跑或手感**。用引用图检查：

```bash
# 列出被改动的模块名，然后全仓扫描谁引用它
python -c "
import os
targets=['<changed_module_a>','<changed_module_b>']
for root,dirs,files in os.walk('.'):
    if any(s in root for s in ['.git','dist','node_modules','__pycache__']): continue
    for f in files:
        if not f.endswith('.py'): continue
        p=os.path.join(root,f)
        src=open(p,encoding='utf-8',errors='ignore').read()
        for t in targets:
            if t in src and t not in os.path.basename(p):
                print(t,'<-',p)
"
```

**2026-09-13 实例**：提交 `aa124f52` 改动了
`scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py` 与
`scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py`。
全仓扫描结果：**这两个模块只被 `tests/taiji_native/test_intervention_reality_gate.py`
（本次新增）引用**。其余测试在 import 层无法受影响 → 全部失败可证为既有。
这比「跑两次对比」更快、更硬。

配套观察（同一事件的旁证）：
- 全量套件的失败集合**每次运行都不同**（两次运行分别在不同进度点多出失败）。
- 失败用例单独运行（或小批组合）**全部通过**。

## 2. 量化基线（2026-09-13，`aa124f52`）

采集命令（**必须用 JUnit XML**，`-rf`/`--tb` 的文本输出会被工具截断丢失）：

```bash
python -m pytest tests/taiji_native/ -q --no-header --tb=no -p no:cacheprovider \
  --junitxml=<tmp>/taiji.xml
# 全量约 15 分钟，建议 run_in_background
```

| 指标 | 数值 |
|---|---|
| 用例总数 | 788 |
| 失败 | **30** |
| 错误（error） | 0 |
| 跳过（skipped） | 1 |
| 全量墙钟 | ~15 分钟 |
| 采集日期 | 2026-09-13 |
| 提交 | `aa124f52` |

失败分为三类（**分类很重要**：类别决定了严重性和修法）：

| 类别 | 数量 | 特征 | 严重性 |
|---|---|---|---|
| A. 架构边界违反 | 2 | 断言失败，非 `SystemExit` | **高**（真实契约违规，非环境问题） |
| B. `SystemExit: 1` 级联 | 28 | 调用 gate `evaluate()` 前即退出 | 中（环境/状态污染，非逻辑错误） |

## 3. 类别 A：架构边界违反（2 项，高）

这两项**不是**环境污染，是真实的边界破坏，需要单独定性。

### DEBT-A1 · 原生核心引入 legacy/序列模型依赖

- 用例：`tests.taiji_native.test_architecture_contract::test_native_core_has_no_legacy_or_sequence_model_dependency`
- 失败形态：`AssertionError: {'__future__','adapter','adaptive_residual_bridge','adaptive_residual_candidate','adaptive_residual_growth',...}`
- 含义：原生核心的依赖集合里出现了不该出现的符号/模块，测试白名单未覆盖。
- 待查：是**新增了真实违规依赖**，还是**白名单未随重构更新**。两者修法完全相反，必须先定性再动手。
- **【已解决 2026-09-13】** 定性结论：**真违规，且与 DEBT-A2 同根因**。A1 的失败信息是整个
  `imported` 集合的噪声转储，实际触发项是 `taiji/document_embedding.py` 贡献的 `transformers`
  （命名边界测试递归证明 taiji/ 内仅此一个文件含违禁 top-level 导入）。修复见 DEBT-A2：
  文件迁出 taiji/ 后本用例恢复通过。

### DEBT-A2 · `taiji/document_embedding.py` 引入 transformers

- 用例：`tests.taiji_native.test_naming_boundary_contract::test_taiji_substrate_never_imports_legacy_or_transformers`
- 失败形态：`AssertionError: Taiji 是自足认知架构，不得依赖 Seed 运行时、seed_platform 治理层、Legacy NeuroPlex 或 HuggingFace transformers：{'taiji/document_embedding.py': {'transformers'...}}`
- 含义：**`taiji/document_embedding.py` 直接依赖 HuggingFace transformers**。
- 影响面较大：`DocumentEmbedder` 是 P5.2b/P5.2c 群体语料的 cue 来源（`embedder.embed([goal_text])[0]`，384 维）。
  这是「原生基底自足性」的核心边界，不是可以随手放宽的白名单。
- 待查：该依赖是历史遗留还是为 P5.1d 语义 encoder 注入所必需；若是后者，需在设计层决定
  「锚定 encoder 是否允许外部依赖」，而不是改测试。
- **【已解决 2026-09-13】** 定性结论（回应原「待查」）：该依赖是 P5.1d 语义 encoder 的
  **功能性必需**（锚定 embedder），不是历史遗留；因此按设计层决策处理——**taiji/ 不再持有
  该依赖**，而不是改测试。修复内容（随本提交落地）：
  1. `taiji/document_embedding.py` → 顶层新包 `instruments/document_embedding.py`；
     checkpoint payload 格式 `taiji-document-embedder-v1` 不变，digest 锚与既有预注册兼容；
     依赖方向 instruments → taiji 单向（仅 `content_digest`），taiji/ 对 instruments 零引用。
  2. `taiji/artifact_internalization.py` 依赖倒置：删除 `from .document_embedding import ...`；
     `SemanticArtifactKnowledgeEncoder(embedder=...)` 改必选注入（None 即 ValueError）；
     `from_checkpoint(..., *, embedder)` 锚校验语义保留；
     `ArtifactInternalizationTrainer.from_checkpoint(..., *, embedder=None)` 对语义 payload
     无注入即 ValueError（fail closed）。
  3. scripts/training 17 个导入行批量更新；8 处 `from_checkpoint` 语义调用点注入 embedder。
  4. pyproject packages.find 增加 `instruments*`；新增 3 个回归测试钉住 fail-closed 与锚漂移。
- 验证：两个契约测试转绿；目标集（architecture/naming/artifact_internalization/
  intervention/P5.2c'''/project_identity）**64 passed**；21 文件 py_compile + ruff 0 错误；
  冒烟确认 `import taiji` 后 `sys.modules` 无 instruments/transformers。
- 残余：p5_1b/1d/1e/1f/1g 等 gate 的完整 `evaluate()`（15 分钟级）未重跑，由 §8(4)
  处置阶段统一重采基线覆盖。

## 4. 类别 B：`SystemExit: 1` 级联（28 项，中）

### 现象

28 个用例在**全量套件上下文**中以 `SystemExit: 1` 失败，单独运行或小批组合**全部通过**。

代表用例（完整清单见 §6）：

- `test_semantic_grounding::test_semantic_grounding_gate_passes`
- `test_terminal_three_domain_governance::test_terminal_three_domain_governance_gate`
- `test_runtime_artifact_store_*`（4 项）
- `test_runtime_structural_artifact_*`（5 项）
- `test_structural_artifact_store` / `test_structural_artifact_measurement_*`（3 项）
- `test_structural_lineage_artifact_*` / `test_structural_lineage_restart_*`（6 项）

### 已验证的事实

- 这些测试**在进程内**调用 gate 的 `evaluate()`（如 `from scripts.training.eval_taiji_semantic_grounding import evaluate`），
  不是 subprocess。
- 单独调用 `evaluate()` → 正常返回，`gate.passed = True`。
- 失败时**没有可用的 traceback**：pytest 的 JUnit writer 在此配置下对 `SystemExit` 不写栈，
  `failure.text` 仅有 `E SystemExit: 1`。**这是采集障碍，处置时需先解决可观测性。**

### 已排除

- 不是 `eval_taiji_semantic_grounding.py` 自身调用 `sys.exit`（模块内无该调用）。
- 不是两文件组合可复现（已试 `test_sequence_learning + test_semantic_grounding`、
  `test_semantic_provider* + test_semantic_grounding`，均通过）。
- 不属于本次改动（§1 引用图已证）。

### 首要嫌疑（待验证，**未确认**）

`tests/conftest.py` 的会话级 fixture `_reset_global_app_state` **只在 session teardown 重置**
`seed_platform.app_state` 单例：

```python
@pytest.fixture(autouse=True, scope="session")
def _reset_global_app_state() -> None:
    yield                      # <-- 整个会话期间不重置
    ...                        # <-- 只在会话结束时重置一次
```

即：**788 个用例共享同一个 `app_state` 单例与全部模块级全局**。若有任一用例留下不可逆状态
（单例缓存、torch 全局开关/随机种子、`reports/` artifact、环境变量、`sys.path` 变更），
后续依赖该状态的 gate 就会真实地返回失败 → `sys.exit(1)`。

这与「失败集合每次不同」和「单独跑就过」两个观测一致。

### 处置时需先建立的观测能力

1. 让 `SystemExit` 带栈：给相关 gate runner 的 `sys.exit` 路径加包装，或在测试内捕获并打印栈。
2. 二分定位污染源：用 `pytest -p no:randomly --lf` 或按文件二分（`--deselect`）缩小到引入污染的用例。
3. 记录触发顺序：污染通常依赖特定前序用例，需把顺序一起记入回归。

## 5. 类别 C：既有 CI 口径差异（旁证，未计入 30）

- `.workbuddy`、`output/manual-r5-canary` 等路径此前已加 `.gitignore` 规则；
  `output/manual-r5-canary` 仍在盘上但被忽略，属预期。
- 全量测试运行中出现沙箱批量删除守卫提示
  （`[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED] {"count":85,...pytest-of-...garbage-...}`），
  为 pytest tmp 目录清理触发，**未观察到影响测试结果**，仅记录以备后续排查。

## 6. 类别 B 完整失败清单（28 项，基线 `aa124f52`）

```
tests.taiji_native.test_natural_language_workbench::test_natural_language_workbench_gate_passes
tests.taiji_native.test_natural_language_workbench_api::test_natural_language_workbench_api_gate_passes
tests.taiji_native.test_natural_language_write::test_natural_language_write_gate_passes
tests.taiji_native.test_runtime_artifact_store_audit_projection::test_runtime_projects_external_store_audit_without_mutation
tests.taiji_native.test_runtime_artifact_store_bridge::test_runtime_artifact_store_bridge_validates_before_native_mutation
tests.taiji_native.test_runtime_artifact_store_preflight::test_runtime_artifact_store_preflights_all_candidates_before_mutation
tests.taiji_native.test_runtime_artifact_store_runtime_reconciliation::test_runtime_store_reconciliation_distinguishes_missing_and_orphan
tests.taiji_native.test_runtime_multi_artifact_store_batch::test_runtime_multi_artifact_store_batch_preserves_parent_order
tests.taiji_native.test_runtime_retention_store_audit::test_store_audit_is_read_only_and_reports_runtime_orphans
tests.taiji_native.test_runtime_retention_store_separation::test_runtime_retention_does_not_delete_or_resurrect_external_artifacts
tests.taiji_native.test_runtime_structural_artifact_batch::test_seed_runtime_restarts_and_consumes_measured_artifact_batch
tests.taiji_native.test_runtime_structural_artifact_failure_concurrency::test_runtime_artifact_failures_are_isolated_and_concurrent_submit_is_idempotent
tests.taiji_native.test_runtime_structural_artifact_multi_round::test_runtime_artifact_multi_round_lifecycle_and_retention
tests.taiji_native.test_runtime_structural_artifact_post_retention::test_runtime_artifact_continues_after_retention_and_restart
tests.taiji_native.test_runtime_structural_artifact_repeated_retention::test_runtime_artifact_repeated_retention_stays_bounded
tests.taiji_native.test_runtime_verified_measurement_bridge::test_verified_measurement_bridge_is_opt_in_and_all_or_nothing
tests.taiji_native.test_semantic_grounding::test_semantic_grounding_gate_passes
tests.taiji_native.test_structural_artifact_measurement_bundle_recovery::test_partial_measurement_bundle_fails_closed_then_recovers_explicitly
tests.taiji_native.test_structural_artifact_measurement_sidecar::test_measured_artifact_sidecar_is_verified_and_legacy_is_explicit
tests.taiji_native.test_structural_artifact_store::test_external_artifact_store_is_immutable_and_runtime_consumable
tests.taiji_native.test_structural_lineage_artifact_batch_isolation::test_artifact_batch_rejects_unknown_keys_and_isolates_partial_failure
tests.taiji_native.test_structural_lineage_artifact_rollback::test_artifact_provenance_survives_rollback_and_terminal_compaction
tests.taiji_native.test_structural_lineage_disk_checkpoint::test_seed_runtime_disk_checkpoint_preserves_migration_and_rollback
tests.taiji_native.test_structural_lineage_multi_batch_artifact::test_multi_batch_artifact_retention_preserves_active_lineage
tests.taiji_native.test_structural_lineage_restart_admission::test_restart_candidate_admission_and_rollback_continue_from_checkpoint
tests.taiji_native.test_structural_lineage_restart_artifact::test_restart_replay_bound_artifact_continues_and_rejects_tampering
tests.taiji_native.test_structural_lineage_restart_continuation::test_restart_continuation_consumes_only_new_evidence
tests.taiji_native.test_terminal_three_domain_governance::test_terminal_three_domain_governance_gate
```

## 7. 其他已知遗留（非测试）

| 编号 | 内容 | 严重性 | 处置约束 |
|---|---|---|---|
| DEBT-G1 | `.git` 内 114 个 dangling 对象 | 低 | **确认无需回溯历史前不要跑 `git gc` / `git prune`** |
| DEBT-G2 | reflog 文本坏行（218 条 `error: invalid reflog entry`），`git fsck` exit 2 | 低 | 仅文本坏，不影响对象与 refs；`HEAD` 可达缺失对象 = 0 |
| DEBT-G3 | `.git/index.corrupt-backup`、`.git/logs/*.corrupt-backup` 备份文件 | 低 | 内含仅存的历史哈希线索，**不要删** |
| DEBT-G4 | 全量测试约 15 分钟且会 SIGTERM，需后台运行或分批 | 低 | 采集一律用 `--junitxml` + `run_in_background` |

## 8. 处置阶段入口条件

进入本册处置阶段**必须**满足：

1. 主线 P5.2c 及其直接下游（P5.2d 在线结果回写）已收尾或明确暂停；
2. 已按 §4 建立 `SystemExit` 可观测性（能拿到栈）；
3. 类别 A 两项已完成定性（真违规 vs 白名单过期），并各自给出「修代码」或「改契约」的结论；
4. 处置前后各采一次 §2 基线，形成可对比的量化。

**禁止**：在未定性前直接放宽 `test_architecture_contract` / `test_naming_boundary_contract` 的断言
来「让套件变绿」——这两项保护的是原生基底自足性契约。
