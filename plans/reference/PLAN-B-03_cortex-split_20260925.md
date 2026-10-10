# B-3 预注册：**Cortex 神对象完整拆分**（所有者已签字；分簇、顺序、判据与回滚点先冻结）

前件：`PLAN-B-03` 的依据 = 审计 §5.1（脚本/神对象债）+ P2 修复报告（`_cortex_helpers.py` 样板已建立、
"完整拆分需研究负责人签字 + 黄金向量等价测试"）；所有者 2026-09-25 **签字放行**。
冻结基线约束：**行为保持**——除 §4 声明的确定性兜底外，**不引入任何静默数值/语义漂移**。

## §1 现状盘点（2026-09-25 实测）

`neuroplex/brain/cortex.py`：**3162 行、56 个方法**。已迁出：`detect_modality`、`infer_domain`
（`_cortex_helpers.py`，等价守卫绿）。按职责分簇：

| 簇 | 成员 | 行数 |
|---|---|---|
| **C-1 质量/评分** | `_nll_quality_from_round1_logits`、`_rolling_nll_quality`、`_probe_inactive_fused`、`to_ngrams`、`_capture_field_memory` | ≈233 |
| **C-2 tokenizer/对齐** | `set_tokenizer`、`set_tokenizer_hub`、`set_general_tokenizer`、`_reencode_domain_generation_context`、`_get_domain_to_general_alignment`、`set_alignment_rules`、`invalidate_alignment_cache` | ≈156 |
| **C-3 路由** | `_executive_route`、`_fingerprint_route`、`_auto_topk_route`、`_instance_route_evolve`、`_select_best_candidate` | ≈385 |
| **C-4 生成编排** | `think`、`generate`、`generate_task_chain`、`generate_staged`、`_generate_p7`、`generate_multimodal`、`_generate_multimodal_p8` | ≈1255 |
| C-5 神经元件生命周期 | `_load_neurons`、`add/remove/isolate/revive_neuron` 等 | ≈600 |
| C-6 状态 setter/存取 | 各 `set_*`、`save_state`/`load_state` | ≈300 |
| C-7 `__init__` | — | ≈133 |

## §2 迁移顺序（**风险从低到高**；每簇一个提交 = 回滚点）

1. **C-1 质量/评分簇** ⇒ `neuroplex/brain/_cortex_quality.py`（评分是纯计算，依赖经参数提升）
2. **C-2 tokenizer/对齐簇** ⇒ `neuroplex/brain/_cortex_alignment.py`
3. **C-3 路由簇** ⇒ `neuroplex/brain/_cortex_routing.py`
4. **C-4 生成编排**：**算法抽离、编排留在 Cortex**（生成是 Cortex 的公共 API 与状态机所在，
   全部外移会切断状态生命周期）——每抽一段配等价测试
5. **C-5/C-6/C-7**：生命周期与 setter 是 Cortex 的**公共 API**，**留在原地**（拆出去反而破坏对外契约）

## §3 每簇的固定动作（**不绿不迁**）

1. **等价测试先行**：迁移前对代表性输入采集黄金向量（`Cortex.方法(x)` 的输出）；
2. 迁移（`self.X` 依赖提升为参数；`@staticmethod` 委托或直调 helper）；
3. 等价测试**逐位断言** + 全量测试回归；
4. **一个提交 = 一个回滚点**（`git revert <hash>` 即整簇回退）。

## §4 声明的行为变更（**唯一一处**，独立提交、独立证据）

`_infer_domain` 的兜底 `_first_domain()`（= `next(iter(set))`）**依赖 set 构造序** ⇒ 非确定语义
（实测：同文本同域集，键形不同 ⇒ 兜底返回 `code` 或 `zh`）。B-3 落地时改为**确定性兜底**
（按 `sorted(neuron_domains)[0]`）——**单独提交**，前后行为对照随提交落盘；不与其他簇的迁移混在一起。

## §5 停止线

* 迁移中发现任何**无法等价**的成员 ⇒ **停在该簇**，该簇标记"与状态耦合、不可纯迁移"，转下一簇；
* **不重构生成数学**（`_generate_p7` 的推理逻辑只做位置搬移，不改语句）；
* 与并行会话（A2/taiji-harness）的文件**零接触**。

## §6 收口判定（2026-09-26 实测；逐成员处置）

| 簇 | 成员 | 处置 | 证据 |
|---|---|---|---|
| C-1 | `_nll_quality_from_round1_logits` | ✅ 抽离 + 委托 | `tests/test_cortex_quality_extraction.py::test_nll_quality_delegate_matches_helper`（**原为 skip**，本轮改 neuroplex Cortex 真夹具后真的跑起来） |
| C-1 | `_rolling_nll_quality` | ✅ 抽离 + 委托（第二刀 `e3a6e0a4`） | 迁移前黄金 `reports/cortex_rolling_nll_golden_20260926.json`（10 格）；否证：改 helper 取位 ⇒ 2 条红 |
| C-1 | `to_ngrams` / `_select_best_candidate` | ✅ 抽离 | 同上文件，`Cortex.__new__` 活体对照 |
| C-1 | `_probe_inactive_fused` | ⛔ 留壳 | 依赖 `self.ensemble.forward` + `self._shared_embedding`（前向 = 状态 + 算力） |
| C-1 | `_capture_field_memory` | ⛔ 留壳 | 读 `self.get_last_field_state()`、写**全局 SleepEngine 单例**（副作用汇，非计算） |
| C-2 | 5 个 `set_*` / `invalidate_alignment_cache` | ⛔ 留壳 | 公共 API + 缓存失效（§2 第 5 条） |
| C-2 | `_reencode_domain_generation_context` / `_get_domain_to_general_alignment` | ✅ 抽离 | 黄金 50000 条全量一致 |
| C-3 | `_fingerprint_route` / `_select_best_candidate` | ✅ 抽离 | 黄金 15 组 |
| C-3 | `_executive_route` / `_auto_topk_route` / `_instance_route_evolve` | ⛔ 留壳 | 调 `think()` + EMA/streak 状态、依赖 `ensemble` |
| C-4 | 单步解码算法 `decode_step` | ✅ 抽离 | 黄金 15 组逐位等价（`0ef93b0b`） |
| C-4 | `think`/`generate`/`generate_task_chain`/`generate_staged`/`_generate_p7`/`generate_multimodal`/`_generate_multimodal_p8` | ⛔ 留壳（**编排即其职责**，§2 第 4 条） | 见 §7 的后续刀清单 |
| §4 | `_first_domain()` 确定性 | ✅ 落地 `1da2877e` | 改前 21/330 格随哈希种子翻转 ⇒ 改后 0 格；21 格确定值逐格等于原黄金 |

**结论**：§1 表列成员已**逐成员处置**（抽出并配等价守卫，或按 §5 判留壳并写明耦合点），
§4 唯一声明的行为变更单独入库 ⇒ **B-3 按预注册范围收口**。

## §7 C-4 后续刀清单（**不在 §1 成员表内**，登记不静默丢弃）

2026-09-26 通读 `_generate_p7` / `_generate_multimodal_p8` / `generate_task_chain` / `think` 得到的
**可抽纯段**（零 self 或依赖可全提升为参数），按性价比排序，逐段仍守"每抽一段配等价测试"：

1. `think` 的 `memory_vectors → seed_memories` 归一（2/3 元组、dict、phase 分支）——最干净的单点；
   **已被 B-4 第七刀（`67e89eec`）直测**（`tests/test_cortex_wiring_and_injection.py`，间谍接
   `ensemble.forward` 断透传协议）⇒ 抽取不再是当务之急，先抽的收益只剩可读性；真要抽时该件即回归网。
2. p8 的 codec 段 logits 采样（mask + top-k + 越界停止）——**RNG 消耗顺序必须逐位一致**，与 `decode_step` 同契约；
3. p8 的 模态→codec 段解析（只读 `MULTIMODAL_TOKENS` 全局配置）；
4. `generate_staged` 的 dict→`TaskSet` 升级（整段即纯映射）；
5. `generate_task_chain` 的阶段 prompt 组装 + `seed_memories` 构造；
6. `_generate_p7` 的 leader 词表→decode 空间解析、回退 logits 选择、终文本解码三段；
7. 退化重试温度 `min(t+0.15, 1.2)`（p7 与 task_chain 两处逐字重复）。

⚠️ 内联段**没有独立可调入口** ⇒ 迁移前黄金只能从"外层方法输出"采（采集即一次性锚点，同
`capture_taiji_cortex_rolling_nll_golden.py` 的教训：迁移后重跑 = 用被测件自采自证）。
这一段与 B-4（核心推理路径覆盖率）合做最省：抽出的纯函数直接可测，覆盖率与拆分同向。

## §8 判据载体在哪（2026-10-10 ㊵-655 带日期补注，只点名、不新增判据）

- 本件的可判内容住在 §3「每簇的固定动作（**不绿不迁**）」：每簇一个提交、门不绿则该簇不迁，这就是本件的验收动作；
- §5「停止线」与 §6「收口判定（2026-09-26 实测；逐成员处置）」是**已执行的出口记录**（1 处行为变更单列在 §4，独立提交、独立回滚），§7 是 C-4 后续刀清单且不在 §1 成员表内；
- 本件对措辞门不可见的原因是节名用了「现状盘点／迁移顺序／固定动作／行为变更／停止线／收口判定／后续刀」，不含门的 3 个标题词（判据／出口／验收）⇒ `structural_criterion_lines` 计 0；本条只点名载体，§3／§5／§6 的口径一字未改。
