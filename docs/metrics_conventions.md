# Seed 指标口径白皮书（Metrics Conventions）

> 目的：统一 `train` / `holdout` 分离、verify 脚本口径、指标命名、在线评估声明与
> 阈值标定记录，防止"训练即测试"式评估泄漏与对外宣称口径失真。
> 适用范围：`scripts/training/*verify*.py`、`scripts/training/eval_seed_corpus.py`
> 及所有对外发布的评估报告。

---

## 1. Train / Holdout 分离原则

**核心约束**：holdout 必须来自与训练语料**互补**的哈希分桶，禁止用训练文件顺序头部
作为评估集。

- 复用 `scripts/training/utils.split_train_eval` 做确定性分桶：
  - 使用 `hashlib.md5(f"{seed}:{text}")` 分桶，**不依赖 `PYTHONHASHSEED`**；
  - `eval_ratio` 与 `seed` 固定（当前默认 `eval_ratio=0.05`、`seed=42`），跨运行一致；
  - 同一文本始终落入同一桶，训练桶与评估桶**无交集**。
- **禁止**：`data[:N]`、`head -n`、顺序取训练文件前 N 行作为评估集——这会令
  holdout 与训练分布重叠，评估泄漏。
- 当训练语料与 holdout 同源（同一文件/同一分布）时，必须走 `hash_split` 取评估桶，
  而非头部。

---

## 2. Verify 脚本口径

verify 类脚本（如 `verify_taiji_native_v7.py`）必须**分离报告**两类指标：

| 字段 | 含义 | 用途 | 是否卡线 |
|------|------|------|----------|
| `train_fit_accuracy` | 在**训练序列**上打分（模型已 `learn_bytes` 该数据） | 信息性，展示拟合程度 | 否 |
| `heldout_accuracy` | 在**未参与训练**的确定性字节流上打分 | 通过条件 / 对外宣称依据 | **是** |

- 以 **`heldout_accuracy`** 作为脚本退出码（`status`）与对外宣称的唯一依据。
- `train_fit_accuracy` 仅用于内部对照，不得写入对外"模型准确率"宣称。
- 二者必须作为**独立字段**出现在 report 的 `metrics` 中（见回归测试
  `tests/seed/test_verify_metrics_contract.py`）。

---

## 3. 指标命名规范

- **禁止**用 `holdout` 命名指代"训练内进度探针"。原 `HOLDOUT_PROBE` 语义误导，
  它实际是训练内探针，必须重命名为 `progress_probe` / `train_progress_probe`。
- 字段级区分：
  - 训练内进度探针 → `progress_probe_accuracy` / `train_progress_probe`；
  - 真正的 held-out 评估 → `heldout_accuracy` / `heldout_*`。
- 报告字段出现 `heldout` 前缀即表示"未参与训练的数据"，不可用于训练内量。

---

## 4. 评估模式声明

- **在线评估**（homeostasis / 阈值适应仍在测试数据上持续学习）必须在报告中显式
  标注 `eval_mode: "online"`，并注明所适应的测试数据范围。
- 在线评估指标**不得**与静态 PPL（离线、固定权重）直接比较或并列宣称。
- 静态指标标注 `eval_mode: "static"`，二者分表呈现。

---

## 5. 阈值标定记录

所有卡线阈值必须在脚本/报告中记录**标定值、实测值与理论期望**：

- 示例（`verify_taiji_native_v7.py`）：
  - 阈值 `HELDOUT_ACCURACY_FLOOR = 0.60`；
  - 训练数据 `"abcd"`（周期 4），held-out 流 `"abc"`（周期 3，从未训练）；
  - 理论期望：`a→b`、`b→c` 命中、`c→d` 未命中 ⇒ 2/3 ≈ 0.667；
  - 实测（seed=7, epochs=200, 2026-08-23）：`heldout_accuracy = 0.692`；
  - 取 0.60 留出实现细节余量；纯位置记忆模型远低于此值。
- 任何阈值调整须同步更新本文件对应条目与脚本内注释。

---

## 6. 回归测试守护

- 防泄漏：`tests/seed/test_seed_corpus_eval.py::test_holdout_hash_split_uses_eval_bucket_not_head`
  断言 holdout 取自 hash 评估桶而非文件头部。
- 口径分离：`tests/seed/test_verify_metrics_contract.py` 断言 verify report 同时含
  `heldout_accuracy` 与 `train_fit_accuracy` 且二者非同源恒等。
- 新增 verify 脚本时，必须补齐上述两类守护，并在本报告登记阈值标定。
