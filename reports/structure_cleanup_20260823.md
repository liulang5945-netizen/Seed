# Seed 仓库结构整理与清洁审计报告（2026-08-23）

> 目标：脚本资产审计、文件命名与归属核对、plans 同步、死代码清理、依赖方向验证。
> 验证：全仓回归 `108 passed, 3 skipped`（零回退）；`python -m compileall scripts/` 通过（3 个乱码遗留文件除外，属既有状态）；依赖方向测试在回归内强制通过。

---

## 一、移动（182 个文件，全部 `git mv` 保留历史）

### 1. `scripts/training/` → `scripts/archive/diagnostics/`（46 个）
`diag_dialogue_*`（30）、`diag_micro_*`（14）、`diag_c25e_leader_quality_gap`、`diag_runtime_mechanism_trace`。
**理由**：全部为服务于已冻结 transformer 种群管线的一次性诊断脚本；引用图分析确认无当前训练/产品调用，仅 `tests/test_micro_data_ab.py` 导入其中常量（已同步改引用）。

### 2. `scripts/training/` → `scripts/archive/`（77 个 py + 4 个运行脚本 `_run_d1_*` / `run_parallel_aug.ps1`）
遗留训练/评估/验证脚本，含：
- 旧管线训练器：`train_neurons_from_scratch`、`train_multi_domain_foundation`、`train_domain_target_sft`、`train_compact_*`、`train_round_level_quality`、`train_encodec`、`train_video`、`train_vqvae`、`train_field_memory_components`
- 旧评估/对比：`eval_aug_joint`、`eval_single_dialogue`、`eval_std_neuron`、`evaluate_ablation`、`compare_sparse_dense*`、`analyze_side_channels`、`finetune_side_channels`
- c25/c26/c27/c28、hub、play engine、a3/a4 遗留验证簇（60+ 个 `verify_*`）
- 词表/数据工具：`hot_swap_vocab`、`upgrade_tokenizer`、`tokenize_sft_p7`、`download_tinystories`、`download_zh_data`、`build_domain_tokenizers` 等

**理由**：与当前 taiji/seed 原生管线无接口；api/main.py 产品训练入口仅依赖保留的 4 个脚本（`finetune_neuron_dialogue`、`finetune_cross_spec`、`train_cross_domain_collab`、`train_hub_neuron`）。CI 与测试钉住的脚本全部保留。

### 3. `scripts/training/` → `scripts/data_prep/`（0 个净增）
`download_simple_zh`、`split_simple_zh`、`split_alpaca_sft`、`convert_alpaca_to_sft`、`inspect_alpaca_zh`、`build_dialogue_extended`、`clean_dialogue_data` 先归入 `data_prep/` 复审，随后与其余遗留语料脚本一并归档（见下）。

### 4. `scripts/data_prep/` → `scripts/archive/data_prep/`（51 个）
含 `download_sft_data`、`download_training_data`、`build_perfect_dataset`、`merge_for_4090d`、`generate_taiji_native_dedup` 等。
**理由**：逐一审计输出路径——除 `download_hf_dialogue_candidates.py`（当前对话语料候选下载器，保留在 `scripts/data_prep/`）外，全部指向已废弃的 `taiji_data/` 旧管线，与当前 `data/simple_zh` + `data/sft` 需求不匹配。

### 5. `scripts/` 根 → `scripts/archive/`（4 个）
`bootstrap_population_demo`、`count_data`、`migrate_ckpt_v3`、`verify_population_baseline`（后者仍被 `tests/test_population_baseline.py` 使用，引用已同步改为 `scripts.archive.*`）。

### 6. `scripts/utils/` 与 `scripts/maintenance/` → `scripts/archive/ops/`（3 个）
`check_endpoints`、`restart_server`（无任何在用引用）；`cleanup_redundant_artifacts`（2026-08-19 一次性存储清理，硬编码白名单已执行完毕）。两个目录清空后删除。

### 7. `PROJECT_ANALYSIS.md` → `docs/history/PROJECT_ANALYSIS_20260822.md`
一次性分析快照（含整理前脚本规模数字），加日期后缀归档，避免根目录残留。

## 二、引用与路径修复

- `tests/test_micro_data_ab.py`：8 处 `scripts.training.diag_*` → `scripts.archive.diagnostics.diag_*`。
- archive 内脚本互引 20+ 处统一改写为 `scripts.archive.*` / `scripts.archive.diagnostics.*` / `scripts.archive.data_prep.*`（含遗漏的 `verify_hub_collab_train`、`verify_feed_sleep_progressive`、`verify_population_baseline`）。
- 116 个被移动脚本的根路径计算（`os.path.dirname` 链 / `Path(__file__).parents[N]`）按新深度对齐；另修正 4 个遗留脚本原本就少一层的路径（`download_supplementary_data/v2`、`merge_for_4090d`、`download_training_data`）。
- 3 个既有乱码/缩进损坏的遗留文件（`legacy_convert_dense_model_format`、`legacy_expand_lifeform_data`、`legacy_fill_general_corpus`）确认属移动前既有状态，按"存档不修补"原则保留。

## 三、顶层清洁

- 删除 `_v9_run.log`（无跟踪的运行日志）。
- 删除空目录 `scripts/utils/`、`scripts/maintenance/`。
- `.gitignore`：更新过时的 `scripts/data_prep` 注释；新增 `_scratch/` 规则；一次性辅助脚本（`audit_helper.py`、`reorg_execute.py`、`fix_depths.py`）用后即删。

## 四、plans 同步

- `BIO_INSPIRED_ARCHITECTURE_PLAN.md`：§6.10 的"当前唯一下一步"（M7）已被 ac37c89 闭合，补记 §6.11（PASS，2026-08-23），声明本文件不再有活跃下一步。
- `NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md`（7 处）、`BOOTSTRAP_CRITERIA.md`（1 处）、`BIO_INSPIRED_ARCHITECTURE_PLAN.md`（1 处）：脚本路径更新到归档后位置。
- `scripts/archive/README.md` 与 `scripts/archive/diagnostics/README.md`：补充子目录划分、新归档簇说明、引用风险边界。
- `plans/archive/` 内容核对：均为带明确历史定位的记录（README 已声明"归档中的下一步不再有效"），无应删除项；历史文档中的旧路径属当时事实，不改写。

## 五、依赖方向验证

- `taiji/`、`seed/` 对 `neuroplex`/`transformers`/`sentencepiece` 的导入：**0 处**（另由 `tests/taiji_native/test_naming_boundary_contract.py` 等契约测试强制）。
- `frontend/src/`、`desktop/`：对 `neuroplex.taiji`、`taiji0`、已移除脚本路径的引用 **0 处**。
- api 对产品训练四脚本的依赖链（`scripts.training.finetune_*` 等）全部指向保留文件。

## 六、整理后结构

```
scripts/
├── training/    29 个：seed/taiji 原生验证 + 产品训练入口 + 共享库（全部被 CI/测试/api 钉住）
├── data_prep/    1 个：download_hf_dialogue_candidates.py（现役语料候选下载器）
└── archive/    337 个：根（遗留训练验证）+ diagnostics/90 + data_prep/51 + ops/3 + native_v6/7
```
