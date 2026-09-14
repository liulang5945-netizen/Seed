# M5 B0：机制修法反事实测量（D5）——五个失败、一个成功

> **规则版本**：本文全部数值只对 **`rule_revision = 0`**（`chosen = bindable[0]` + episode 级 cue 长度）有效。
> M4 落地后"冻结 vs 反事实"两列同时变成新规则、`gain_delta` 归 0，属预期的恒等退化，不得回改本文数字或重跑覆盖。

> 2026-09-13；基线 `f9825943`。
> 本文回答决策 **D5**（是否改组合机制）所依赖的唯一未知：**改了到底管不管用、会不会回归**。
> **M1/M4 均未实施**：没有改任何 gate、runner 或规则；本文全部结果是**反事实测量**。
> 证据：[反事实脚本](../../scripts/training/probe_taiji_b0_m1_counterfactual.py) /
> [测量报告](../../reports/taiji_b0_m1_counterfactual_20260913.json) /
> [契约测试](../../tests/taiji_native/test_b0_m1_counterfactual_contract.py)（13 passed，不跑训练）。

## §1 结论摘要

1. **测了六个修法变体，五个失败、一个成功。**
2. **失败的五个**（详见 §4）：`m1a`（无进展即让位）**安全但完全无效**（0 回归、0 改善、无交错）；
   `m1b`（轮转）**破坏基线**（3 回归）；`m2`（每成员自身进度）1 回归；`m2a` 2 回归；`m3`（轮次用尽即交接）1 回归。**五者均未产生任何交错轨迹。**
3. **成功的那个是 `m4_failure_handoff`**（+21 行、2 处替换）：
   - **冻结面：0 回归，反而改善 2 个 cell**（`a+b` 0.25→0.50、`a+c` 0.25→0.50）；
   - **候选面：`interleaved = 4/4`**（两种优先级顺序都成立）；
   - **同参照增益 `mean(P − max_i S_i) = +2.000`**，正是逃生通道的闭式上限 `(1−(−1))×4/4`；
   - **+2.000 > 1.65**（三种候选参照中最严的一条）⇒ **H2 协作主张第一次变得可达**。
4. **关键前提先被验证**：候选任务 `create_and_override` **可满足**（脚本化参考步骤能达标），且**顺序是强制的**——必须先 `create` 再 `set_language`；反序时 `editor.set_language` **绑定成功但执行失败**（`bound=True, success=False`）。没有这一步，所有负结果都不可解释。
5. **根因链至此完整**：失败动作**既不消耗该成员的回合、也不触发交接**，于是第一个成员在缺失文件上无限重试 `set_language`，独占 episode 直到 `STEP_CAP`；同时 cue 长度用**整集步数**，后加入的成员被查询在自己轮次之外的位置。**两处都要改**。
6. **D5 现在有实测可行的选项**，但它仍是**设计变更**（组合需要消费"执行失败"这一信号，并跟踪成员自身进度），不是调参。**未实施、未冻结。**

## §2 方法：反事实必须可审计

| 保障 | 做法 |
|---|---|
| 不改被测机制 | 取 `inspect.getsource(p5_2b._member_episode)`，**只做文档化的唯一锚点替换**；替换后的源码在冻结模块命名空间的**副本**中 `exec` |
| 锚点唯一性 | 每个锚点必须**恰好出现 1 次**，否则 `SystemExit` 拒绝出报告（冻结规则一变就大声失败） |
| 不 rebind | 断言 `frozen._member_episode` 身份不变；每个变体的 `delta.frozen_attribute_unchanged` 均为 `true` |
| 不泄漏到门禁 | 契约测试扫描四个 gate/runner 源码，确认无一引用反事实模块 |
| 只读 | 无 `torch.save` / `save_file` / `from_checkpoint(` / `optimizer` |

**Delta 规模**（可直接审阅）：

| 变体 | 替换数 | 增加行 | 锚点 |
|---|---|---|---|
| `m1a_no_progress` | 1 | +11 | `chosen = bindable[0]` |
| `m1b_tick_rotation` | 1 | +1 | `chosen = bindable[0]` |
| `m2_per_member_progress` | 1 | +3 | `cues = tuple([cue] * (len(steps) + 1))` |
| `m2a_progress_plus_handoff` | 2 | +14 | 两者 |
| `m3_repertoire_aware` | 2 | +27 | 两者（+ 注入 `REPERTOIRE`） |
| **`m4_failure_handoff`** | **2** | **+21** | 两者 |

## §3 候选面必须先可解释

| 检查 | 结果 |
|---|---|
| 脚本化参考步骤能达标（可满足） | **True**（4/4 context） |
| 参考顺序反转后能达标 | **False**（4/4 都不行） |
| 顺序是否强制 | **True** |
| 反序时 `editor.set_language` | `bound=True`、`success=False` —— 绑定成功但执行失败 |
| 成员轮次长度（train 证据） | `member-a=3, member-b=3, member-c=3, member-d=4` |

⇒ 候选任务**有效**（不是不可解），且**有强制顺序**。这使负结果可解释，也使"必须按状态/失败信号交接"成为可推导的要求，而不是猜测。

## §4 六个变体的实测

| 变体 | 冻结面回归 | 改善 | `interleaved` | 同参照增益 | 判定 |
|---|---|---|---|---|---|
| `m1a_no_progress` | **0** | 0 | 0 | +0.000 | 安全但**完全无效** |
| `m1b_tick_rotation` | 3 | 1 | 0 | +0.000 | **破坏基线，不可采纳** |
| `m2_per_member_progress` | 1 | 1 | 0 | +0.000 | 回归且无效 |
| `m2a_progress_plus_handoff` | 2 | 0 | 0 | +0.000 | 回归且无效 |
| `m3_repertoire_aware` | 1 | 1 | 0 | +0.000 | 回归且无效 |
| **`m4_failure_handoff`** | **0** | **2** | **4** | **+2.000** | **成功且不回归** |

**为什么前五个都不行**（每个都有独立证据）：

- `m1a` 只拦"种类重复"。但失败动作**不计入** `executed`，因此不进入"已执行种类"，首成员仍可反复选择它 ⇒ 无效。
- `m1b` 轮转把无效动作序列推进到合同层 ⇒ 新增 `contract_intercepted:preview_ValueError`，并压掉 3 个既有 cell。
- `m2` 只修"成员被查询的位置"，但交接仍不发生 ⇒ 1 个 cell 回归、无改善。
- `m3` 以"轮次用尽"为交接信号。信号确实触发了（`repertoire_exhausted` 出现 8 次），但**首成员那次失败的 `set_language` 已消耗掉**，后续 `create` 无法补回被丢弃的覆盖状态 ⇒ 仍是 `step_cap`。

## §5 M4 的两处改动

```python
# 1) 交接：最近一次尝试失败的成员先让位；一旦有人成功，解除封锁（世界变了）
last_success_index = max((i for i, s in enumerate(steps) if s.get("executed")), default=-1)
blocked = {
    s.get("chosen")
    for i, s in enumerate(steps)
    if s.get("chosen") and not s.get("executed") and i > last_success_index
}
chosen = next((c for c in bindable if c["member"] not in blocked), None)
if chosen is None:
    ... return finish("all_members_blocked")

# 2) 进度：按成员自身已执行步数构造 cue，而不是整集步数
own_steps = sum(1 for s in steps if s.get("executed") and s.get("chosen") == member_id)
cues = tuple([cue] * (own_steps + 1))
```

**两处缺一不可**：只改 (2)（即 `m2`）有 1 个回归且不产生交接；只改 (1) 无法让后加入的成员被查询在正确位置。`m4` = (1)+(2) 才同时做到"不回归"与"能交接"。

**为什么这不算调参**：`m1`–`m3` 都是在**已有信号**（绑定失败、种类重复、轮次用尽）上做代理；`m4` 让组合**消费"执行失败"这一此前被丢弃的信号**，并**跟踪成员自身进度**——后者是冻结规则从未拥有过的状态。

## §6 M4 的实测结果

### 6.1 冻结验证面（不回归，且有改善）

| cell | 冻结规则成功率 | M4 成功率 | Δ |
|---|---|---|---|
| `member-a+member-b` | 0.25 | **0.50** | **+0.25** |
| `member-a+member-c` | 0.25 | **0.50** | **+0.25** |
| 其余 9 个 cell | — | 完全一致 | 0 |

**0 回归**：M4 在既有任务面上不比冻结规则差，并回收了部分"冻结规则 headroom"（续篇 §4 测得的 +0.5 现象）。

### 6.2 候选面 `create_and_override`（协作主张首次可达）

| 顺序 | 最佳 pair | 该 pair 成功率 | `interleaved` | `mean(P − max_i S_i)` | 最严参照要求 |
|---|---|---|---|---|---|
| 冻结顺序 | `member-a+member-c` | **1.00** | **4 / 4** | **+2.000** | 1.65 |
| 反序 | `member-c+member-d` | **1.00** | **4 / 4** | **+2.000** | 1.65 |

- 四个单体**全部 0.00** ⇒ 四个 context **全部**为组合专属可解；
- 三种候选参照（0.5 / 1.0 / 1.5）**全部 feasible**，可支撑参照上限 1.85；
- **两种优先级顺序下都成立**（只是成功的 pair 不同）⇒ 不是顺序侥幸。

## §7 完整根因链

1. 任务需要"文件存在"**先于**"显式语言覆盖"（§3 实测强制顺序）。
2. 冻结组合规则把 pair 内第一个成员（`member-a`）排在最前，其政策是 `read → resolve → set_language`。
3. 文件缺失时 `editor.set_language` **绑定成功但执行失败**。
4. 失败动作**不消耗该成员回合、也不触发交接**（冻结代码 `"executed": bool(outcome.success)` 且交接只看绑定）⇒ `member-a` 无限重试，独占 episode 到 `STEP_CAP`。
5. 同时 cue 长度取**整集步数**，后加入成员被查询在自己轮次之外 ⇒ 即使让位也无法贡献。
6. ⇒ 24 个冻结单元 + 三个候选面全部 `interleaved = 0`。**这是机制缺陷，不是表征或任务缺陷。**

## §8 对 D5 与后续的影响

| 项 | 状态 |
|---|---|
| D5 是否有实测可行选项 | **有**：`m4_failure_handoff`（0 回归、2 改善、`interleaved 4/4`、增益 +2.0） |
| M4 是否已实施 | **否**。仍是反事实；落地需 D5 决策 + 单独预注册 |
| 落地后旧结果怎么办 | 旧规则下的 P5.2b/c 报告**只对旧规则有效**，必须重跑并单独报告，**不得改写** |
| M4 的残余风险 | 反序时成功的 pair 不同 ⇒ 需在预注册中冻结优先级定义；`all_members_blocked` 是新停止原因，须纳入门禁语义审查；本测量只覆盖 4 个候选 context，规模需扩大 |
| 与仲裁上界的关系 | 不矛盾。上界针对 `P ∈ {S_i}` 的机制；M4 使 `P ∉ {S_i}`（真实交错）成为可能，实测 4/4 交错即是证据 |

## §9 纪律与未做的事

- **未实施 M1/M2/M3/M4**：不改 gate、不改 runner、不冻结规则；全部为命名空间副本中的反事实。
- 未注册任何新任务；候选定义仍只存在于探针内（`partition="probe"`、`b0probe-` 前缀）。
- 未训练任何候选任务专用模型：复用冻结的四个成员。
- 未改动任何冻结报告/预注册/阈值；`growth_admitted=false`、`can_promote=false` 贯穿。
- 未声称"已实现协作"：M4 是**测量到的可行性**，不是已落地的能力。
- 契约测试不跑训练（<2s）；反事实脚本（≈50s）作为脚本化仪器单独运行，重跑结果一致。
