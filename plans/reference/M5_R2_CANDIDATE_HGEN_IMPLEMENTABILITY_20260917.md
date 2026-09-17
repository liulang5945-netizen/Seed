# M5 R2 候选 H-GEN 的实现可行性核验（规格 §2.2 的修正）

日期：2026-09-17。状态：**核验完成，含对已采用规格的一处实现修正**。
规格：[R2 下一候选规格](M5_R2_NEXT_CANDIDATE_GENERATION_STATE_TRAINING_20260917.md)（用户已采用）。
性质：**只读核验 + 规格修正说明**；未改任何训练代码、未启动任何训练。

## §1 核验结论：两个载体的"输入 / 目标可分性"不同

规格 §2.2 写的是「**只按比例 φ 让部分 episode 步改用模型自身生成的前缀**，
目标/表示/架构不变」。核验发现：**这句话在现有两个训练载体上的可实现性不同**。

| 载体 | 前向接口 | 输入与目标是否可分 | 结论 |
|---|---|---|---|
| `LanguageAlignmentTrainer`（R2 aligned，P3b-v2 用的就是这个） | `observe(symbol, learn=...)` —— **单个 `symbol`** | ❌ **不可分** | 无法在其上实现 scheduled sampling |
| `SequenceWorkspacePrototype` / `SequenceWorkspaceTrainer`（H3.8 隔离原型） | **`teacher_forced_logits(prefix, response)`** —— **prefix 与 response 分开**，返回 `torch.Tensor` | ✅ **可分且可微** | **本候选应在它上面实现** |

### 1.1 为什么 `LanguageAlignmentTrainer` 不可分（证据）

`taiji/model.py::Taiji.observe` 里，`symbol` **同时**承担两个角色：

- **输入**：`L1747 sensory = self.sensor.encode(symbol)`；
- **F1（predictive readout）的 next-byte 学习目标**：`L1777-1785` 用 `previous.last_symbol`
  与 `previous.motor_probabilities[symbol]` 对比。

`taiji/language_alignment.py::_target_pass`（L762）则逐字节 `for symbol in episode.target_bytes:
self._observe(symbol, learn=learn, ...)` —— **喂进去的就是真实目标字节本身**，
即标准 teacher forcing；**没有任何位置能插入"用预测符号当前缀"**。

⇒ 要在此载体上实现，必须**新增"输入符号 ≠ 目标符号"的 API**（改 `Taiji.observe` 签名或其内部时序），
这是**核心接口变更**，代价与风险都远大于规格 §2.2 的表述。

### 1.2 为什么隔离原型可以（证据）

`taiji/sequence_workspace.py`：

- `L360 def teacher_forced_logits(self, prefix: bytes, response: bytes) -> torch.Tensor`
  —— **前缀与回答是两个独立入参**；
- `L383 logits = self.teacher_forced_logits(prefix, response)`
  —— 由调用方决定喂什么；
- `L41` 的文档还专门说明「the teacher-forced fold only ever sees `y[:t]` when producing
  position `t`」—— **它对自己处在 TF 模式这一点是显式自知的**。

⇒ 在同一位置，把 `response` 的前半段换成**模型自己生成的前缀**、同时仍用**真实 `response`** 算损失，
是**局部可实现**的，不需要改任何核心接口。

## §2 规格修正（**供确认**）

- **§2.2「拟改动」的载体**：由"R2 aligned 训练入口"改为 **`SequenceWorkspaceTrainer`**；
- **改动内容不变**：按 φ 让部分步改用模型自身生成的前缀；目标 / 信用 / 表示 / 架构对**该原型**而言不变；
- **其余各节不变**：§1 事实与未分离因素、§3 四层区分、§4 基线/预算/停止线、§5 与 H3.8 的区别、
  §6 资源申请（仍只申请 pilot）。

**为什么这不违反 §5.5「不换名追加预算」**：H3.8 用同一原型检验的是
「**workspace 机制是否被承担**」（结论 `workspace_unused`）；
本候选检验的是「**train/inference 分布不一致是否是充分根因**」。
**被检验的假设不同、成功与失败的判据不同、失败的含义也不同** ——
H3.8 失败说明"机制没被用起来"，本候选失败说明"exposure bias 不是充分根因"。

## §3 残留风险（实现前必须处理）

1. **原型与 aligned 链路不同源**：隔离原型上的结论**不能直接外推到 `chat()` 链路**，
   必须在报告里声明（与 H3.8 同样的界限纪律）；
2. **自动微分**：原型返回 `torch.Tensor`，而项目此前不以自动微分训练为主线 ——
   需在实现前确认该原型的既有优化器与本候选一致（若需变更，属 §5.2 允许的"核心学习设计变更"，
   但**必须显式记录**）；
3. **φ 的调度**：φ 从 0 线性升到 φ_max；φ_max、升速属**冻结项**，须在跑之前写死。

## §4 下一步（实现顺序，待确认后执行）

1. 在 `SequenceWorkspaceTrainer` 上加 `generation_state_ratio`（**默认 0 ⇒ 既有行为逐字节不变**）；
2. 先跑 **φ=0 的对照**，确认与既有 H3.8 记录**可复现**（证明改动未引入偏移）；
3. 再跑 **φ>0 的处理臂**，同一语料、同一预算；
4. 判定：两臂 dev `exact` 是否可区分（即规格 §4 的停止线）。
