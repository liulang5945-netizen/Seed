# M4.V2 B3-K C-entry parity v3 预注册（readout 修订 + 全新 sealed）

> 冻结日期：2026-09-10。前置：[parity formal 预注册 §7](M4V2_B3_K_C_PARITY_FORMAL_PREREGISTRATION_20260910.md)（v1 formal honest fail + readout 归因）与 [caliber revision §6/§7](M4V2_B3_K_C_PARITY_CALIBER_REVISION_20260910.md)。本文冻结 v3 的唯一设计修订（readout）与全新 sealed 评分合同；冻结后 materialize sealed v2、实现 runner 并运行。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 设计修订（唯一变更：readout 语义）

v1 formal 的失败归因（validation-only 探针，sealed 未再读取）钉死：logit 相加 readout 把共享 parent 权重加倍（`logit_ch1 + logit_ch2 = 2·W_parent·x + (Δ1+Δ2)·x + …`），未见输入上概率过度自信 → sealed MSE +0.107/课程；同 artifacts 改权重平均即反转为 −0.0017（9/9 为负）。

**v3 readout（冻结）**：candidate = 两通道 state_dict 的**权重平均** `(W_ch1 + W_ch2) / 2 = W_parent + (Δ1+Δ2)/2`——「两个不同顺序训练运行」的校准保持合成，是 fixed-large「replica 概率平均」的权重域对偶。实现为 state_dict 相加后除 2，装入 parent 结构 learner 后走标准 predict。

**不变项（全部冻结，零重训）**：widened/fixed-large v2 artifacts 原样复用（digest 对 v2 build report 校验）；parent workers 不动；训练流、600 新增步、类平衡课程、排列守卫全部不变。v3 只改变推理期合成方式——这是设计修订的明确边界。

## 2. 全新 sealed test v2（当前 sealed 已被 v1 formal 读取，不可复用）

- Materializer：`scripts/training/materialize_taiji_m4v2_b3_k_c_sealed_test_v2.py`；`SEALED_TASK_SEED = 47`（v1 为 19）；3 episodes × 3 steps（`sealed2_` 前缀文件，语言轮换与 v1 不同）；同 v1 合同（observation/episode payloads embedded、`scores_or_targets_embedded=false`、status `materialized-unread`）；
- **不交性**：file digests 与 train/validation fixture（task_seed=0 基线）及 **v1 sealed artifact 的 digests** 均不交，materializer 强制校验；
- artifact 落盘 `plans/manifests/taiji_m4v2_b3_k_c_sealed_test_v2.json`，internal `artifact_digest` 与文件 sha256 在 runner 输入校验时核对。

## 3. 判据（与 v1 formal 同结构，全部在看 sealed v2 前冻结）

- **T1/T2 技术门**：sealed v2 digest 校验 + v2 artifacts 对 build report 校验 + 课程独立性门通过；评分键集无漂移、无 NaN；
- **G1 质量门（validation）**：candidate combined delta < 0 于 ≥ 2/3 课程；
- **G2 灾难界**：`epsilon_cat = max(0.01, 3 × population std(9 个 validation candidate combined deltas))`，pre-sealed 产物落盘后才读 sealed v2；每课程 sealed candidate delta ≤ +epsilon_cat；
- **G3 主判据**：sealed 上 candidate combined delta < fixed-large 于 ≥ 2/3 课程，且 3 课程均值更优；
- **G4 描述性**：per-cell/per-course 全量呈现。

**结果映射**：G1∧G2∧G3 过 → 「校准保持合成下，顺序分化学习规则 ≥ strong control」闭合，进 K 轴 scorecard v3；任一失败 → widened 路线关闭（readout 修订是最后一次设计尝试），fixed-large 概率平均保留为 C-entry strong arm，B 阶段继续。

## 4. 停止线

- sealed v2 materialization 的不交性校验失败 → 停，换 task seed 重materialize（不触碰模型）；
- 任何 digest/键集失败 → 停；
- **sealed v2 读取后禁止任何判据/阈值/readout 调整**——本预注册是 widened 路线的最后一次 sealed 机会；
- 不引入 provider/联网/真实客户端写入/default runtime；`can_promote=false` 固定。

## 5. 产物

- `scripts/training/materialize_taiji_m4v2_b3_k_c_sealed_test_v2.py` + `plans/manifests/taiji_m4v2_b3_k_c_sealed_test_v2.json`；
- `scripts/training/eval_taiji_m4v2_b3_k_c_parity_formal_v3.py`（复用 v1 formal 的评分机械，readout 换权重平均；先 py_compile/ruff）；
- `reports/taiji_m4v2_b3_k_c_parity_formal_v3_presealed_20260910.json` + `reports/taiji_m4v2_b3_k_c_parity_formal_v3_20260910.json`；
- 路线图执行记录 + 独立提交。

## 6. 执行记录（2026-09-10，formal v3 已运行，G3 失败——widened 路线按预注册关闭）

两阶段纪律执行（pre-sealed epsilon 产物先落盘，sealed_read_count=1）。首轮运行曾因 internal digest 校验把 `materializer_sha256` 误计入 unsigned 而中止（runner 校验 bug，修复后重跑；sealed 在修复前未被读取，纪律未破）。

| 门 | 结果 |
|---|---|
| G1 质量门 | **通过**——weight-averaged candidate 的 validation delta 在 ≥2/3 课程 < 0（readout 修复使学习信号恢复正常） |
| G2 灾难界 | **通过**——sealed candidate delta 每课程均为负（−0.00220/−0.00165/−0.00135），远离灾难 |
| G3 主判据 | **失败**——1/3 课程胜出（course0：candidate −0.00220 vs fixed-large −0.00188；course1/2 落后 −0.00165/−0.00135 vs −0.00188）；课程均值 −0.00173 vs −0.00188 |

### 6.1 判定

- **v1 formal 的 +0.107 灾难被 readout 修订完全消除**——G1/G2 通过证明权重平均是校准保持的正确合成，且顺序分化训练真实改善了 sealed 泛化（三课程全部为负）。
- **但主假设仍被否决**：在同等预算与同等容量下，顺序分化 + 权重平均 **不优于** 同构 replica 的概率平均（1/3 胜出、均值落后 0.00015）。两种合成方式落在同一表现带内，差异为二阶小量。
- 按 §3 冻结的结果映射：**widened 路线关闭**；`fixed-large`（同构 replica 概率平均）保留为 C-entry strong arm；B 阶段继续。

### 6.2 学习问题定位（widened 关闭后的收敛点）

证据链收敛到唯一学习问题：**K 课程 harness 的可见信号空间只有 3 类**（.py/.rs/.ts 首 file profile；recover 受 K2 词汇限制；.h 歧义受 fact 词汇限制）。在此空间内：任何课程/顺序/合成的变化都只能产生二阶差异（v3 的 ±0.0002 量级）；学习规则的差异需要更丰富的可见信号才能显现。诚实的后续路径是**扩展可见类空间的设计预注册**（新 fact 词汇或新 registry/schema 维度，允许 recover/.h 类进入），而非继续在 3 类空间内调课程/合成。
