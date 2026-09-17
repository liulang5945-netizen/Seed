# R2-H3.9：六维有界只读归因总账

2026-09-17。只读复审，不改实验面。目标 sketch / slot phase / bridge credit 三维已在 [H3.8 复审](M5_R2_H3_8_SEQUENCE_CREDIT_REVIEW_20260917.md) §1 完成（哈希监督、固定 phase、信用截断、近似局部规则）；本账补齐剩余三维（renderer → prefix representation → capacity/data）并给出六维完整归因，作为 H3.8 隔离原型采用决策的完整输入。全部结论附代码级证据。

## 1. renderer（渲染路径）

证据：`_target_pass`（[language_alignment.py:738-780](../../taiji/language_alignment.py#L738-L780)）——`begin_response_plan()` 后 `learn_response_plan_target(...)` 一次性写入槽状态，随后逐 byte `_observe(symbol, learn=..., predictive_readout=response_plan_readout)` 并 `advance_response_plan_phase()`。

1. **计划状态在 episode 内凝固**：槽向量在 byte 循环前由 target sketch 一次性写入，生成/训练期间不因预测误差更新——renderer 没有逐 byte 的状态回写。
2. **phase 推进是确定性的字节计数**（16-byte stride），与内容无关。
3. **0.75/0.25 混合是固定局部规则**（H3.8 §1 已测：非链式梯度；四相有限差分测试现已钉住该映射本身正确——修复的价值是「实现与声明一致」，不是「实现是好的架构」）。
4. **renderer 与 readout 是同一对象**（`ResponsePlanReadout`，[organs.py:747](../../taiji/organs.py#L747)）：没有独立的可训练循环解码器；「渲染」= 槽混合向量喂给 byte 预测读出 + UTF-8 合法性表（`_utf8_allowed`）+ byte 损失监督结束标记。

归因：**renderer 维度的负结果根因 = 渲染器没有任何跨 byte 的可学习内容状态**——唯一可学习部件（byte 读出权重）的条件输入是凝固的 sketch 槽 + 确定性 phase，学到的只能是「该 sketch/phase 下的字节表」。

## 2. prefix representation（前缀表示）

证据：模块 docstring（[language_alignment.py:8](../../taiji/language_alignment.py#L8)）「The prefix is consumed without learning」+ `_prime`（[language_alignment.py:732-736](../../taiji/language_alignment.py#L732-L736)）逐 prefix byte `_observe(symbol, learn=False)` + `learn_response_plan_target(self._response_plan_target(episode))` 的目标是**对 response 文本本身**的 count-sketch（`_count_sketch_chunk`，[response_plan_target.py:501-515](../../taiji/response_plan_target.py#L501-L515)）。

1. **前缀零学习**：prompt byte 只推动冻结动力学状态，不产生任何参数更新。
2. **监督目标与 prefix 无关**：槽向量 = response 自身的 sketch——可学习映射是「response sketch → response bytes」，**prompt→response 的内容依赖在监督信号里结构性缺席**。
3. 因此 readout 可学习的最优策略是按 episode 记忆 sketch 键控的字节表（12 个家族 = 12 张表）；H3.7 的「无内容方向、boundary/因果消融门失败」正是该监督设计的必然输出，不是训练不足。

归因：**prefix 维度是最深的一层**——问题不在表示能力，而在**监督目标不包含需要学习的前缀依赖**。任何数据/容量扩充都不能修复一个不存在于 target 里的依赖。

## 3. capacity/data（容量与数据）

证据：有效参数 277,970/300k（验证报告）；语料 12 train / 8 dev / 4 final episodes，target 42-45 byte/episode，epoch = 537 byte 更新。

1. **数据处于记忆化区间**：12 个家族 × 45 byte，任何读出都能在此规模上零泛化地拟合；dev/final 的 8/4 条同分布样本无法区分记忆与泛化。
2. **容量非瓶颈**：参数预算未触顶，且 H3.7 的失败模式（排序/边界/因果门）是结构性的，不是欠拟合形态。
3. **数据与目标耦合**：即便扩数据，只要 target 仍是 response 自身的 sketch，扩的只是记忆表规模。

归因：capacity/data 维度**单独不构成根因**，但与 §2 耦合后使一切现有 dev/final 数字不可解释为内容能力。

## 4. 六维总账

| 维度 | 根因等级 | 一句话归因 |
|---|---|---|
| target sketch | **结构** | 监督目标是 response 自身的 count-sketch——prefix 依赖在 target 里缺席 |
| slot phase | 结构 | 固定 16-byte stride 相位与内容无对齐；UTF-8 边界与 byte schedule 错位 |
| bridge credit | 结构（已修） | 0.75/0.25 混合反传与凝固槽信用——修复使实现与声明一致，但该声明本身只是局部规则 |
| renderer | 结构 | 无跨 byte 可学习内容状态；渲染 = 凝固槽 + 确定性 phase + byte 读出 |
| prefix representation | **最深** | 前缀零学习 + 监督不含前缀依赖 ⇒ 无内容依赖可学 |
| capacity/data | 放大器 | 记忆化区间数据使全部既有 dev/final 数字不可解释为内容能力 |

## 5. 对 H3.8 采用决策的含义

六维归因收敛于同一点：**当前监督结构（response 自身 sketch + 前缀零学习）使「内容方向」在数学上不可形成**，H3.7/H3.7B 的全部负结果都是该结构的必然输出。H3.8 隔离原型（可训练 prefix 编码器 → 内容工作空间 → 可训练循环 renderer、真实下一 byte 交叉熵）恰好逐项反转这六维——这是支持采用的架构论据；决策权与授权在用户（H3.8 §7：不能从继续推进推断为默认核心迁移授权）。本账不替代该决策，只补全其依据。
