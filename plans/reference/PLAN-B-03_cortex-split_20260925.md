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
