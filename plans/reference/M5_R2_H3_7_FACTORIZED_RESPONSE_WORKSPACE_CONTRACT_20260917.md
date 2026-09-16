# M5 R2-H3.7 分解式回答工作空间与因果信用合同

> 状态：设计复审、隔离实现、三 seed matched dev、三组只读消融与 aggregate 已完成；aggregate 在 final 前停止。训练权限已消耗完毕，本合同之外的训练、final 读取和默认入口变更不被授权。
>
> 日期：2026-09-17

## 1. 触发原因与判读边界

H3.5-A 与 H3.6-B 已经完成了同一回答计划候选的六次 matched dev 和三次只读 bridge ablation。机器边界全部通过，但能力出口没有形成：H3.6 control 的 dev sequence 为 `0/0.125/0.125`，treatment 为 `0/0.25/0`，三 seed 均值相同；exact response 与 required-term coverage 均为 0；paired prompt sensitivity 均为 1.0。移除 treatment bridge 后 sequence 反而恢复到 `0.25/0.25/0.25`，同时文本碰撞升高。

这组结果排除了以下解释：

1. 输入完全没有进入 native predictive state；
2. UTF-8 合法性或 end-marker 停止器是主要瓶颈；
3. 只要增加一个静态 plan 向量，renderer 就会自动获得回答级信用；
4. 再在 H3.6 同一 target 上追加 epoch 可以把失败变成能力。

H3.6 的实际结构仍有三个可定位缺口：

- 全部回答被压入一个 plan vector，回答头部、主体和尾部没有独立监督与消费阶段；
- `plan_bridge` 只被随机初始化和读取，byte prediction error 没有更新它；
- 计划目标更新与逐 byte renderer 更新是两个近似并列的局部规则，没有一条可观测的“字节错误 → 当前计划槽 → prefix planner”信用链。

因此 H3.7 只提出一个新机制假设：

> 在相同总参数上，把 response plan 改为多槽、分阶段消费的 native workspace，并让 renderer 的真实 byte error 通过解析的局部信用更新 bridge 与当前槽；如果该机制有效，它应在未见 dev 上产生稳定的内容/sequence 改善，并在 bridge 与 slot credit 消融时消失。

这不是 H3.6 的补训，也不是 L2、L3、Mini 或架构采用结论。

## 2. 唯一方案：factorized response workspace

### 2.1 原生运行时链

```text
v2 prefix (system/policy/context/history/user/assistant boundary)
        │
        ▼
native predictive motor_context
        │
        ▼ 只在 assistant boundary 创建一次
response_plan_state = [slot_0, slot_1, slot_2, slot_3]
        │                  │
        │                  └─ 每 16 个已生成 byte 切换消费槽
        ▼
tanh(motor_context + learned_plan_bridge(active_slot_mix))
        │
        ▼
isolated native byte renderer + UTF-8/end-marker constraint
```

固定参数如下：

| 项目 | H3.7 取值 | 约束 |
|---|---:|---|
| plan 总宽度 | 48 | 仍计入 active parameter budget |
| slot 数 | 4 | payload 与 target geometry 固定 |
| slot 宽度 | 12 | `48 = 4 × 12`，每槽独立目标 |
| phase stride | 16 byte | 只由已经生成的 byte 计数决定，不读答案长度 |
| phase mix | 当前槽 0.75 + slot0 0.25（phase 0 只用 slot0） | 保留头部结构信号，避免后续槽完全割裂 |
| plan 生命周期 | boundary 创建一次，reset 清空 | 不跨 episode 泄漏 |
| renderer 输入 | 当前 `motor_context` + active plan mix | 不接 task/family/split/评价字段 |
| 外部 provider | 禁止 | 原生 N-mode 唯一计分 |

slot0 是回答头部/全局风格的可消费锚点；slot1–3 是按固定 byte 窗口取得的后续回答片段。这里的“头部/片段”是目标编码布局，不是运行时标签，也不允许把 `unknown_policy` 或 `task_family` 直接写入槽。

### 2.2 训练目标

训练期可以读取 reference response 生成候选监督，但 reference 只进入 candidate update，不进入 generation：

- `L_plan[slot_i]`：每个槽对应 response 的固定 16-byte contiguous chunk 的确定性 count-sketch；每槽独立 L2 normalization，空槽使用版本化 empty token；target encoder 只在 train split 校准并绑定 corpus/parent digest。
- `L_render`：当前槽条件下的真实下一 byte prediction error；更新隔离 renderer synapses/bias。
- `L_bridge`：把 renderer 的 backprojected error 乘以当前 plan-conditioned `tanh` 的局部导数，更新 `plan_bridge`。
- `L_slot_credit`：将同一局部 error 经过 bridge 转回当前槽，只更新该槽对应的 planner rows；更新尺度固定为 bridge 学习率的四分之一，防止计划在回答中无界漂移。
- `L_boundary`：现有 UTF-8 legal mask、end-marker 和 boundary 率单独记账，不得用其高分掩盖内容收益。

H3.7 不使用自动微分，不把 `required_terms`、人工语义判断或 split 写入模型；上述局部投影是可审计的显式 credit 规则。slot target 更新发生在 boundary，byte credit 发生在真实 observed byte 之后的 prior error 上。

### 2.3 与 H3.6 的关键差异

| 维度 | H3.6 | H3.7 |
|---|---|---|
| 目标形态 | 单一 32 维 native+ngram 向量 | 4 个 12 维 response chunk 槽，总宽度 48 |
| 目标几何 | whitening 后与 compositional ngram 等权 | 每槽独立 count-sketch，保留回答阶段边界 |
| 消费方式 | 整个回答始终读取同一 bridge 向量 | 每 16 byte 消费当前槽并保留少量 slot0 全局信号 |
| bridge | 仅初始化/读取 | 接收 renderer error 的显式局部信用 |
| planner | 只在 boundary 拟合 target | boundary target + 当前槽 renderer credit |
| 失败解释 | 计划变化但未对齐未见 target | 可区分 target 不可学、bridge 未使用、slot credit 断路 |

H3.7 不复用 H3.6 target encoder、target map、checkpoint 或 final；新 variant 使用新的 payload format、seed offset 和 parent lineage。

## 3. 数据与公平对照

首轮继续使用已冻结的 `tests/fixtures/r2_h3_5a_response_plan_v3.jsonl`，digest 为
`0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691`。它只承担小规模机制验证，不扩大开放语言主张：

- train/dev/final 仍按完整 response 与 family 隔离；
- `answer`、`say_unknown`、`clarify`、`refuse` 仍有结构覆盖，但这些字段只用于分层报告；
- H3.3–H3.6 已曝光样本不成为新的最终集；H3.7 dev 只在完成所有 seed 后读取，final 始终延迟；
- control 使用同一初始化、同一训练顺序和同一 300k 总预算的 response-phase native owner；
- treatment 使用 H3.7 factorized workspace；
- 两臂不同时修改 shared fabric、memory、默认 Seed 入口、P3b 或 P5.2d。
- treatment 的只读 `slot_credit` 消融保留 phase 0 的 slot0 锚点和后续 phase 的
  0.25 全局混合，只去掉当前 phase 的 0.75 槽分量；它模拟“当前槽信用不可用”，
  不把训练尺度临时改成 0，也不向 checkpoint 写入消融状态。

公平账本至少包含：model seed、candidate initialization、parent digest、代码 revision、corpus digest、epoch/update/episode、core/candidate/effective parameters、checkpoint size、UTF-8/end-marker 配置和 evaluation code digest。

## 4. preflight 与最小实现门

任何正式 epoch 前，六个边界必须都通过：

1. zero-step atomic save → fresh-process restore → 同一 prefix 的 plan slots、phase、renderer probabilities 一致；
2. 一次 plan target update 只改变 H3.7 candidate，parent protected checkpoint digest 不变；
3. 一次真实 byte update 必须改变 `plan_bridge` 或当前 slot planner credit 的 digest，不能只改变 renderer；
4. phase advance 只改变 candidate 的消费游标和对应 probability surface，reset 后清零；
5. child checkpoint restore 后继续 generation 与 read-only score 可重复；
6. 缺失/错误 target format、slot 数、phase stride、corpus digest 或 parent digest 必须 fail closed。

总 active parameter 必须在建模前预留并满足 `effective <= 300000`；H3.7 的 48 维计划增加的参数不能在训练后追加。

## 5. dev 停止与晋级规则

H3.7 只允许从 preflight 进入一次 3-seed matched dev。每个 seed 先完成 control/treatment 的固定 10 epoch、最多 120 train episodes，再做一次只读 slot/bridge/sequence ablation；不读 final。

### 5.1 必须同时满足的继续条件

1. 三个 seed 的 checkpoint、恢复、parent 保护、预算、target lineage 和 native-only 均通过；
2. treatment 相对 control 在 dev 至少一个非代理内容指标（required-term coverage、sequence criterion 或 target response legal rank/likelihood）方向一致改善；
3. UTF-8/boundary 不退化，paired prompt sensitivity 不下降；
4. 消融 `plan_bridge` 和 `slot_credit` 至少撤销主要 treatment 内容收益，而不是只让 collision 增加；
5. treatment 的改善不是只在 train 或只在 end-marker 上出现；
6. 原始 bytes、目标概率、slot active phase、bridge/slot digest 变化和完整配置均可追溯。

所有条件成立才允许另立一次性 final capability read；即使 final 成功，也只进入 R2 机制复评，不自动晋级 S2/L2/Mini。

### 5.2 任一触发即停止

- runtime 读到 task/family/split/required/reference 等 oracle；
- checkpoint 保存恢复、atomic、parent 或 target lineage 失败；
- effective 参数超 300k；
- 只提升 train、surprise、UTF-8、boundary 或 collision；
- 三 seed 方向不一致且无执行前冻结的稳定性解释；
- bridge 或 slot credit 消融不撤销内容收益；
- 48 维 factorization 仍只产生静态 plan 变化但未见 response 内容信号。

停止后的归因顺序固定为：target sketch → slot phase schedule → bridge credit → renderer readout → native prefix representation → capacity/data。不得回到 H3.6 同质 epoch，也不得用改名新增阶段规避失败。

## 6. 交付物与当前不变项

本包完成后应提交：

- H3.7 机读合同与 target payload schema；
- factorized readout 的实现、单元测试和 preflight 报告；
- control/treatment 三 seed dev 原始报告、ablation 原始报告和 aggregate decision；
- checkpoint/report/target map 的 hash 与 lineage；
- 更新后的 `03_CURRENT_EXECUTION.md`、`01_SCOPE_AND_PHASES.md`、`02_GATES_AND_CI.md`、`07_MINI_MODEL_DELIVERY.md`、README 与唯一 VISION 文件。

保持不变：H3.5/H3.6 负结果只读保留；不读它们的 final、不追加旧 target epoch；不改默认 SeedRuntime.chat；不把 H3.7 candidate 当作产品模型；Mini 验收仍后置。

**执行结果与后续唯一下一步**：三 seed matched dev、只读消融和 aggregate 已完成，结果见`reports/taiji_r2_h3_7_matched_dev_result_20260917.json`。aggregate 判定`stopped_before_final`，因此不读 final、不追加 epoch、不切默认入口；下一步只做有界的只读归因复审，顺序固定为 target sketch → slot phase schedule → bridge credit → renderer readout → native prefix representation → capacity/data。该复审不得修改历史报告或将 H3.7 candidate 晋级为产品模型。
