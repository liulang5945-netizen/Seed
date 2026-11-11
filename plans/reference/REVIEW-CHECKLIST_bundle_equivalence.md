# 审阅清单：重建件与历史发布件的读数比对

**状态**：常设审阅流程（台账 `D-20261111-03` 第 2 条）。
**不是** CI 门，**不产生**自动绿/红，**不维护**任何读数基线。

## 0. 为什么这份清单存在，以及它不是什么

CI 的门（`ci.yml` 的 `bundle-determinism` 步）判的只有一件事：**同一次 job 内两次独立重建的读数是否等价**。它不拿历史发布件当参照物，理由写在 `D-20261111-03` §1：发布行绑定的是**发布时的代码**，不是当前 HEAD；把它接成门，红的时候最省事的动作就是重生成基线，等于把判据交给"想不想让它绿"。

与历史发布件的比对因此移到这里：由人按本清单在**持有封存件的机器**上执行，结果连同运行条件与代码提交号一起归档。**不设自动门，也不把"这次不一致"自动升格为结论**——不一致先分类（见 §6）。

## 1. 前提

| 项 | 要求 |
|---|---|
| 机器 | 持有封存件 `output/taiji_core_09/subject_bundle.pt` 的维护机（该文件被 `*.pt` 忽略，不入库） |
| 封存身份 | sha256 `ce4c62685d55bb518d234e7407e19b05257d1d761a0430432f7d0f09641321cf`，77,609,636 B |
| 代码 | 当前 HEAD；**记录完整提交号**（`git rev-parse HEAD`）与工作区是否干净（`git status --short`） |
| 解释器 | `torch==2.13.0+cpu`、Python 3.12（与发布时的钉法一致） |

## 2. 必须一起记录的运行条件

否则"相等"不算证据（能力卡 §13）：

- `python`、`platform.platform()`、`torch.__version__`
- `torch.get_num_threads()`、`torch.get_num_interop_threads()`，以及 `OMP_NUM_THREADS`／`MKL_NUM_THREADS` 是否设置
- `torch.backends.cpu.get_cpu_capability()`（ISA）
- `torch.__config__.show()` 里的 BLAS 实现与版本、CPU capability 行
- CPU 型号、OS 版本、是否虚拟化

最快的取法：跑一次 `scripts/training/verify_taiji_bundle_rebuild_determinism.py`，它把上面这组字段原样写进报告的 `runtime_conditions`（那次的判定本身不是本清单的结论，只借它的条件记录）。

## 3. 步骤一：身份核对（记录，不判红）

```
python -c "from pathlib import Path; from taiji.core_09_bundle import BUNDLE_RELATIVE_PATH, sha256_of_file; print(sha256_of_file(Path(BUNDLE_RELATIVE_PATH)))"
```

- 等于封存值 ⇒ 走 §4 的 `published_rows` 支。
- 不等于 ⇒ 这条路走不下去：**不要**把重建件当发布件用。改用 `output/_scratch_*` 里的重建件时，按 §4 的 `structural_invariants` 支记录，并在归档里写明"本次不是封存件"。

## 4. 步骤二：重放四个消费者

意志统一：每个脚本都写一份**新报告到独立目录**（它们都是 write-once，不许覆盖已发布文件），再用发布件逐项比。

| 消费者 | 命令（相对仓库根） | 发布件 |
|---|---|---|
| 10.21.1 | `python scripts/training/verify_taiji_10_21_1.py --output <dir>/10_21_1.json` | `reports/taiji_core_10_21_1_preflight_20261101.json` |
| composition graph | `python scripts/training/verify_taiji_composition_10_21.py --stage graph --output <dir>/composition_graph.json` | 该脚本声明的发布报告 |
| 14.5 price | `python scripts/training/verify_taiji_record_relation_14_5.py --stage price --report <dir>/14_5.json --manifest <dir>/14_5_manifest.json --payload-dir <dir>/payload` | `reports/taiji_core_record_relation_14_5_20261102.json` |
| 14.6 preflight | `python scripts/training/verify_taiji_record_relation_14_6_preflight.py --stage preflight --report <dir>/14_6_preflight.json --manifest <dir>/14_6_preflight_manifest.json --payload-dir <dir>/payload_14_6` | `reports/taiji_core_record_relation_14_6_preflight_20261102.json` |

两套分支（互斥，**必须写明走的是哪一支**）：

- **`published_rows`**：封存件在位。断言发布行逐项相等，浮点按 `1e-12` 相对容差、其余精确。
- **`structural_invariants`**：非封存重建件。只断言不依赖位相等的不变量（行数、字段集合、枚举集合、`target_mass∈[0,1]`、顺序不变性、`null_mass` 恒定、来源 id 属于发布集合）。

`tests/taiji_native/test_composition_read_probe_anchor.py` 运行时会把实际走的分支写进 `output/ci_verify/composition_probe_branch.json`，归档时一并附上。该测试**不 skip**：skip 会让断言在 CI 里永远不执行。

## 5. 步骤三（按需）：14.6 全量 compare

≈10,898.6 CPU s / 11,341 s wall（≈3.15 h）。**不进 CI 门**；需要时：

- 本地：`python scripts/training/verify_taiji_record_relation_14_6.py --stage compare --report <dir>/14_6_compare.json --manifest <dir>/14_6_compare_manifest.json --payload-dir <dir>/payload_compare`
- 或手动触发 `.github/workflows/equivalence-deep.yml`（`workflow_dispatch`）。该工作流会在跑之前先验封存 sha，不等就直接失败——因为 `load_chain` 一定会在父身份上中止，跑满三小时没有意义。

## 6. 结果表（照抄归档）

| 消费者 | 分支 | 判定字段 | 整数/计数 | 浮点最大相对差 | 与发布件一致？ | 备注 |
|---|---|---|---|---|---|---|
| 10.21.1 | published_rows / structural_invariants | | | | | |
| composition graph | | | | | | |
| 14.5 price | | | | | | |
| 14.6 preflight | | | | | | |
| 14.6 compare（按需） | | | | | | |

附：`git rev-parse HEAD`、工作区状态、§2 的全部运行条件、`composition_probe_branch.json`。

## 7. 不一致时的判读（顺序固定）

1. **先查代码**：`git log <发布提交>..HEAD -- taiji/ scripts/training/` 里是否有改动会进读取路径。代码漂移不算"发布行不可复现"。
2. **再查运行条件**：§2 有哪一项与发布时不同（线程数、ISA、BLAS、OS）。
3. **最后才谈发布行**：前两项都排除后，仍不一致 ⇒ 记"该发布行在当前条件下不可逐位复现"，**报 owner**。
4. **禁止动作**：调容差、改发布值、重生成基线、把 `structural_invariants` 当成"通过了发布行比对"。

## 8. 归档位置

结果放 `output/<本次目的名>/`（中间产物，不入库），结论摘要写进台账的新 `O-` 条目或能力卡对应节，附 §6 表格与 §2 条件。没有运行条件的结论不要写进台账。

## 9. 撤回

本清单不改变任何判据，撤回即删本文件并删除 `D-20261111-03` §2 的引用；CI 门不受影响（它从不读本文件）。
