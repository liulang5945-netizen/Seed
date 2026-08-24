# scripts/legacy — 历史诊断脚本隔离区

本目录用于隔离早期一次性诊断脚本，使其不再散落在 `scripts/archive/diagnostics/`
中与活跃脚本混在一起，也便于集中去重公共常量。

## 本次迁移了哪些文件（及为什么）

审计发现 `scripts/archive/diagnostics/` 下多个 `_diag_*.py` 互相复制同一段
硬编码绝对路径：

```python
torch.load(r"e:\Seed\checkpoints\seed_corpus.pt", weights_only=False)
```

全仓精确匹配该 raw 字符串的只有 **4 个**文件（审计原述「5 个」应为多算）：

| 文件 | 原位置 | 说明 |
|------|--------|------|
| `_diag_weight_mass.py` | `scripts/archive/diagnostics/` | 崩塌 vs 新生模型突触权重对比 |
| `_diag_param_health.py` | `scripts/archive/diagnostics/` | 参数健康度（NaN/Inf/范数） |
| `_diag_lateral_health.py` | `scripts/archive/diagnostics/` | 侧向竞争权重体检 |
| `_diag_dynamics_health.py` | `scripts/archive/diagnostics/` | 皮层活动率/误差动力学 |

迁移动作（均为**移动**，未删除任何文件）：
1. 新建 `scripts/legacy/__init__.py`，定义公共常量 `CHECKPOINT_DIR`。
2. 上述 4 个文件移入 `scripts/legacy/`，头部加 `# MIGRATED` 注释与
   「仅历史诊断、不再使用」说明。
3. 4 个文件内 `torch.load(r"e:\Seed\checkpoints\seed_corpus.pt", ...)`
   改为 `torch.load(CHECKPOINT_DIR / "seed_corpus.pt", ...)`，导入
   `from scripts.legacy import CHECKPOINT_DIR`。**加载语义不变**
   （仍是 `torch.load`，仍指向 `E:/Seed/checkpoints/seed_corpus.pt`）。

> 注：`__init__.py` 中 `CHECKPOINT_DIR` 用 `parents[2]`（仓库根），而非审计建议
> 原文的 `parents[1]`——`scripts/legacy/__init__.py` 的 `parents[1]` 会得到
> `scripts/checkpoints`（不存在），`parents[2]` 才指向真实 `checkpoints/`，
> 以保加载语义不被改变。

## 运行目录依赖（待修清单，本次仅记录、未改动）

以下脚本使用依赖运行目录的导入，无法作为包被 pytest/CI 直接 import。
它们**不在本次迁移集合**，且经排查未被任何自动调用方引用（多在 `archive/`，
属历史脚本），按保守原则**仅记录、不大规模改**：

- `from verify_taiji_m6_endogenous_replay import ...`（裸名、依赖运行目录）：
  - `scripts/training/verify_taiji_m7_cue_chain.py:22`
  - `scripts/archive/native_v6/_diag_m6_write_basis.py:55`
  - `scripts/archive/native_v6/_diag_m6_test_bite.py:20`
  - `scripts/archive/native_v6/_diag_m6_margin.py:33`
  - `scripts/archive/native_v6/_diag_m6_logits.py:31`
  - `scripts/archive/native_v6/_diag_m6_coverage.py:29`
  - `scripts/archive/native_v6/_diag_m6_churn_rate.py:20`
  - `scripts/archive/native_v6/verify_taiji_signed_opponent.py:39,40`
  - `scripts/archive/diagnostics/_diag_m7_motor.py:22`
  - `scripts/archive/diagnostics/_diag_m7_field.py:30`
  - `scripts/archive/diagnostics/_diag_m7_decode.py:23`
  - `scripts/archive/diagnostics/_diag_cortical_identity.py:18`

- `from _diag_m6_write_basis import ...`（兄弟文件裸名导入，运行目录依赖）：
  - `scripts/archive/native_v6/_diag_m6_test_bite.py:20`
  - `scripts/archive/native_v6/_diag_m6_churn_rate.py:20`
  - `scripts/archive/native_v6/_diag_m6_logits.py:31`

建议修法（待有自动调用方或统一整改时）：将上述裸导入改为相对项目根的
绝对包导入，例如
`from scripts.training.verify_taiji_m6_endogenous_replay import ...`
与 `from scripts.archive.native_v6._diag_m6_write_basis import ...`。

## 新脚本编写规范

- 新脚本应从项目根以 `from scripts.training.X import` 形式可被 import，
  **避免**运行目录相对导入（`from verify_xxx import`、`from _diag_xxx import`、
  以及 `sys.path.insert(0, r"e:\Seed")` + `from seed import Seed` 这类写法）。
- 需要检查点路径时统一使用 `from scripts.legacy import CHECKPOINT_DIR`，
  不要硬编码 `e:\Seed\...` 绝对路径。
- 不要再往 `scripts/archive/diagnostics/` 增加一次性诊断脚本；历史脚本统一
  收口到 `scripts/legacy/`。
