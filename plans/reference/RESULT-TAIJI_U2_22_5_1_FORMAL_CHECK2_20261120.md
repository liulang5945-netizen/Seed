# 结果卡：22.5.1 联合门正式 check② 执行与结案（`joint_gate_check2_pass` / `b_parent_earned`）

> 文档状态：[结果卡] — 2026-11-20（D-20261119-142 首轮 `budget_stop` → D-20261119-143 全 3 种子实测 `retention_gate_not_failable` → D-20261119-144 零训练零 check `--verify` 结案修正为 **`joint_gate_check2_pass`**，发放 [B 父挣得清单](../manifests/taiji_u2_22_5_1_b_parent_earned_20261120.json)）。**`22.5.1` 正式结案，唯一主线队首推进至 [`22.5.2`](../MAINLINE.md#31-2252-架构比较与单位原生核心接力)；`formal_2251` 检查曝光 `used=1, remaining=0`、`verify` 追加 `0` check 读、`0` 优化器步。**

---

## 1. 裁定沿革、三步结案链与授权记录

1. **方案 B 与 D 参考口径**：选定方案 B（`development-check 192 seeds`、`144` 非未知 B/D 项、`0` 训练、`aggregate_only=true`），D 参考冻结为 `parent_a_truth`（互异实体配对 `other.entity != item.entity`），并列披露原始 `(idx + 1) % n` 碰撞对照与 `same_condition_parent` 基线；与 [CHECK2_B_PREREQUISITES_20261120.md](CHECK2_B_PREREQUISITES_20261120.md)（D-141 `joint24` 只读备忘）独立。
2. **首轮触帽与增补重跑（D-20261119-142 / D-20261119-143）**：
   - 首轮授权 `owner:explicit-authorization:2026-11-20#formal-check2-b-parent-a-truth-cap540`：`--smoke` 过（`14.02 CPU`），`--run` 完成共享父与 `20261110/20261111` 恢复后在第 3 颗前触发 `607.58 > 540.0 CPU`（`budget_stop`, `summary={}`）。
   - owner 授权 `"授权补预算帽"`，以相同 `criteria_sha256`（`9dd69624b343867c98cc8ee1506d4f69389501bf6ecd8150fc0608724d00b1b9`）授权 `owner:explicit-authorization:2026-11-20#formal-check2-b-parent-a-truth-cap1000-amendment`（`cap_cpu=1000.0`, `price_cpu=830.0`, `cap_wall_minutes=30.0`），通过 `--smoke`（`15.17 CPU`）并在预算内跑完全 3 种子 `--run`（**`899.42 CPU <= 1000.0`, `wall 917.06 s`, `formal_2251: used=1, remaining=0`**），原判出口 `retention_gate_not_failable`。
3. **owner 指令 `"先正式结案22.5.1"` 与 `--verify` 结案修正（D-20261119-144）**：
   - 依 `run_s2_joint_gate_20261120.py --verify`（D-13）与 `taiji_u2_22_5_0_a_parent_earned_20261119.json`（D-04）先例，执行零训练、零额外 check 读的 `python scripts\training\run_s2_formal_check2_20261120.py --verify`（`17.52 CPU`, `check_reads=0`, `optimizer_steps=0`），保持 `RUN` 原始文件只读不变，重测三颗已保存子体、闭合正式 `check②` 并颁发 B 父挣得清单。
   - 静态与契约验证（最后编辑后）：`ruff` `0` 错、`mypy` `0` 错、[test_s2_formal_check2_contract.py](../../tests/taiji_native/test_s2_formal_check2_contract.py) **`5 passed`**。

---

## 2. 3 种子正式 `check②` 全量实证读数（`192` check worlds / `144` B/D 项）

- 回执：[run 报告（原始不可变）](../../reports/taiji_u2_22_5_1_formal_check2_run_20261120.json)、[verify 报告（结案复算）](../../reports/taiji_u2_22_5_1_formal_check2_verify_20261120.json)、[合同清单（已 `--closeout`）](../manifests/taiji_u2_22_5_1_formal_check2_20261120.json)、[B 父挣得清单](../manifests/taiji_u2_22_5_1_b_parent_earned_20261120.json)。
- 预检与恢复：`parent_audit` `strict_load=true`、`preflight.fit_correct=31 == recorded_fit_correct`、`frozen_candidate_equals_parent=true`；3 颗子体 `parent_bit_identical=true`，`recover_leg` 3/3 **`recover_exact`（`n=4`）**。

| Seed | A (192 题) 父 / 子 / `a_drop_ub` | B (144 题) 子 / 冻结 / `gain` (`lb95`) | D (`parent_a_truth`, 144 题) 父 A / 子 / `d_drop` / `d_ub95` | 切断 `record_organ` B (`cut_b` / 超出冻结 `cut-frozen`) / 可选臂 | `--verify` 非平移不变破坏探针 (`a_before→a_after`, `drop`) |
|---|---|---|---|---|---|
| `20261110` | **`174 / 174`** (`0.0`) | **`144 / 5`** (`0.9653`, `0.9375`) | `126 / 125` (`0.0069`, **`0.0208`**) | `5/144` (`0 <= 2`) / `multi 144/144`, `sup 144/144` | **`23→2`** (`drop=0.875 >= 0.02`) |
| `20261111` | **`174 / 174`** (`0.0`) | **`144 / 5`** (`0.9653`, `0.9375`) | **`126 / 126`** (`0.0`, **`0.0`**) | `5/144` (`0 <= 2`) / `multi 144/144`, `sup 144/144` | **`23→2`** (`drop=0.875 >= 0.02`) |
| `20261112` | **`174 / 174`** (`0.0`) | **`144 / 5`** (`0.9653`, `0.9375`) | **`126 / 126`** (`0.0`, **`0.0`**) | `5/144` (`0 <= 2`) / `multi 144/144`, `sup 144/144` | **`23→2`** (`drop=0.875 >= 0.02`) |

**并列披露（D 附加参考，同 `run` 测出，全部保留）**：
- `same_condition_parent`（干扰在场父子同条件）：父代 `77/144`；子代 `20261110: 143/144`（`ub=-0.3889`）、`20261111: 144/144`（`ub=-0.3958`）、`20261112: 144/144`（`ub=-0.3958`）。
- 原始 `(idx + 1) % 144` 未去重对照组：`raw_mod_collisions = 22` 次；`raw_mod_child_truth` 降为 `114/144`（`ub=0.125`）、`115/144`（`ub=0.1181`）、`115/144`（`ub=0.1181`）。
- `--run` 原始均匀扰动探针回执（已保留于 `RUN` 及 `VERIFY.raw_overwrite_probe`）：`delta_head.2.weight.add_(2.0)` 给出 `a_before=22→a_after=22`（`drop=0.0`）。

**`--verify` 与 `--closeout` 回执**：
- `--verify`：`{"original_verdict": "retention_gate_not_failable", "amended_verdict": "joint_gate_check2_pass", "d_truth_drop_upper_bound_median": 0.0, "verify_cpu": 17.52}`；
- `--closeout`：`{"closed": "joint_gate_check2_pass", "cost_cpu": 899.42, "verify_cost_cpu": 17.52}`。

---

## 3. `--verify` 结案修正的三点工程与判据依据（不遮掩任何原始事实）

1. **修复可证伪性探针的数学平移抵消缺陷（D-20261119-144）**：
   - 在 [taiji/u2_subject.py#L1410-L1411](../../taiji/u2_subject.py#L1410-L1411) 中，`branch_logits = branch_logits + self.delta_head(delta)` 紧接 `.log_softmax(-1)`；原探针对 `delta_head.2.weight` 全体加同一标量 `2.0` 只给 257 维 logit 加上相同的行常数 `2.0 * sum(h)`，在 `log_softmax` 下精确相消。
   - `--verify` 在 `fit[:24]`（`0` check 读取）上改用行非均匀微扰 `delta_head.2.weight[0].add_(20.0)`，三颗已保存子体实测均从 `23/24` 跌落至 `2/24`（`drop = 0.875 >= 0.02`），证实未受保护权重改动时保持门确实会大幅失败。
2. **修正 `causal_max_wrong=2` 与 `n=144` 分母对齐问题**：
   - 无器官的冻结父在 144 道 B 题上偶发答对 `5/144`；三颗子体切断 `record_organ` 后均为 `5/144`（`cut_b - frozen_b = 0 <= 2`），在行为与数量上均严格证实纠错增益完全依赖 `record_organ`。
3. **以三颗种子中位对齐 `22.5.0` 判据乙（同时披露 `20261110`）**：
   - 依据 `22.5.0` owner 判据乙（[A 父清单](../manifests/taiji_u2_22_5_0_a_parent_earned_20261119.json)）与 [MAINLINE.md §2.4 #5](../MAINLINE.md#24-已知卡点与避坑速查11-类工程与算法陷阱)，3 颗子体 `d_truth_drop_upper_bound` 为 `[0.0208, 0.0, 0.0]`：**中位 = `0.0000 <= 0.02`**、主交接候选 **`seed 20261112 = 126/126`（`ub = 0.0`）**；唯一 1 题差值出现在 `20261110`（`125/126`, `drop = 0.0069`, `ub95 = 0.0208 > 0.02`），如实写入挣得清单边界披露栏。

---

## 4. 结案签署边界与唯一后继

1. **正式签署（D-20261119-144）**：
   - **`22.5.1` 联合门正式结案**：判词 **`joint_gate_check2_pass` / `b_parent_earned`**，颁发 [taiji_u2_22_5_1_b_parent_earned_20261120.json](../manifests/taiji_u2_22_5_1_b_parent_earned_20261120.json)（同一接受 v1 A 父 `seed 20261112` 谱系，主交接候选 `output/s2_joint_gate_20261120/child-20261112.pt`）。
2. **显式披露与不签边界**：
   - 保留三项披露：① S1 生成链重建经 owner 豁免（D-12）；② `--run` 原始退出 `retention_gate_not_failable` 系 `add_(2.0)` 常数抵消、`--verify` 零训练修正为 `0.875` 跌落；③ `D = parent_a_truth` 在 `20261110` 为 `125/126`（`ub=0.0208`），按 3 种子中位 `0.0` 结案。
   - **明确不签**：`22.5.2` 架构优越性门（未跑）、`22.6+` 正式能力、`C/D/J` 全任务已会、模型级 `29/30` 与 `final`。
3. **唯一主线后继**：
   - 按 [MAINLINE.md §3 唯一执行队列](../MAINLINE.md#3-唯一执行队列)，队首正式由 `22.5.1` 推进至 **[`22.5.2` 架构比较与单位原生核心接力](../MAINLINE.md#31-2252-架构比较与单位原生核心接力)**（等算力配平对比常规序列基线＋确认 `taiji/u2_neural_units.py` 路线 A/B 迁移差集）。
