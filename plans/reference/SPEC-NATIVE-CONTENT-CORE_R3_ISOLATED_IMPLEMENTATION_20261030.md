# 共同内容核心：重构-3 隔离实现与零步恢复记录

日期：2026-10-30，Asia/Shanghai。状态：**新核心 `new_core_v1` 已在隔离路径实现；结构探针 53/53 通过；未训练、未评价、未读新 sealed 面；不称已学会，不作能力主张。** 上级合同见[重构-2 候选/基线比较合同](SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md)、[重构-1 统一语义合同](SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)；当前队首只看[03](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 交付物

| 交付 | 路径 | 当前证据 |
|---|---|---|
| 新核心模块 | [native_content_core.py](../../taiji/native_content_core.py) | 版本化内容记录/存储、时间条件读取、逐步条件解码、credit 组、完整 bundle checkpoint |
| 定向测试 | [test_native_content_core.py](../../tests/taiji_native/test_native_content_core.py) | 13/13 通过；覆盖 schema、版本链、unknown、owner/scope、时间意图、逐步读取、首字节监督、credit、cut/rescue、query-only 隔离、零步恢复、编码器 revision |
| 结构探针 | [probe_native_content_core_r3.py](../../scripts/training/probe_native_content_core_r3.py) | 读取公开 R1 fixture 17 案；不训练、不评分正确性 |
| 机器报告 | [taiji_native_core_r3_probe_20261030.json](../../reports/taiji_native_core_r3_probe_20261030.json) | `r3_structure_passed`，53 项检查全过，0 optimizer updates，`sealed_text_read=0` |

探针配置：width 64、content 32、144,355 参数；CPU 6.03 秒、wall 4.55 秒（含一次独立子进程恢复）。这只说明结构成本很低，不构成训练预算。

## 2. 实现边界

| 接口 | 本实现 | 关键约束 |
|---|---|---|
| `ContentState` | `NativeContentRecordV1`：raw span、latent、来源、owner/scope/session、revision、valid_from/valid_to、superseded_by、status、encoder_revision | 更正保留旧版本；`resolved` 才写当前链；`unresolved` 可读但不进当前值 |
| `TimeIntent` | 存储按 `current/previous/unknown` 给合法候选集 | `previous` 只取当前版本的直接前驱；无前驱返回空集，不退回 current |
| `ContentReader` | 每步用 `(query latent + time embedding + decoder hidden)` 对候选打分并读出 | 不接收实体/属性/值标签；候选按 content_id 排序 |
| `DynamicDecoder` | 沿用 V3 byte 解码脊，逐步把 read 注入 evidence | 首字节由第 0 步预测；训练/生成共用同一 `next_logits` |
| `CreditMatrix` | `core/content_encoder/query_encoder/reader/attention/decoder/output` 七组 | 声明的 reader 与 decoder 接收任务梯度；store 无参数 |
| `Checkpoint` | bundle = 参数 + store + cursor + 可选 task_state | 任何记录 latent 的 `encoder_revision` 必须等于当前参数 revision |

### 2.1 解释器可读的三处关键语义

- **候选集不是答案**：宿主只按时间意图、owner/scope/session 和版本角色给出合法候选；选哪条、读出什么由模型计算，宿主不传 gold 实体/属性/值。
- **未解析内容不再一律丢弃**：`unresolved` 记录保留 raw span、来源与状态，可进入读取候选；但它不写当前值，因此不能冒充已确认事实。
- **cut 不是删库**：删除被选记录时，其直接前驱会重新打开为当前版本，store 不出现悬空 supersession；这样 cut 只移除被检验的内容，不破坏版本链自洽。

### 2.2 信用路径的实测事实

- 存储 latent 是持久化常量：`reencode=False` 时 `content_encoder` **收不到**任务梯度；这不是 bug，而是“持久内容不随当前任务回写”的边界。
- `reencode=True` 时，同一任务 loss 可到达 `content_encoder`；测试已分别验证两种情况。
- `set_credit_group("reader", False)` 后 reader 参数梯度为空；恢复后可用。credit on/stop 因而可在同一结构上单独切换。

## 3. 结构探针读数（零步、未训练）

| 检查组 | 读数 | 解释 |
|---|---|---|
| 17 案候选数（current/previous） | 全对 | 版本链、unknown、owner 边界的结构语义正确 |
| 同 query、不同当前值 | `delta 0.00283` | 当前内容确实改变读端；幅度小，因为未训练 |
| 同当前值、不同历史（current 条件） | `delta 0.0` | 当前候选身份一致时读出一致 |
| 同当前值、不同历史（previous 条件） | 有差异 | 历史版本确实进入旧值读取 |
| 同 query 文本、不同 time intent | 有差异 | 时间意图进入读取路径 |
| 相关交换 | `0.07253` | 被选内容改变会改变输出 |
| 无关交换 | `0.04608` | 未训练的读取尚未把无关内容压到零 |
| 相关/无关比 | `1.57` | **零步不构成判别力证据**；必须由训练后的 R5 判定 |
| cut | `0.00157` | 移除被选内容会改变输出 |
| 合法救援 | `0.0` | 恢复后逐位相同 |
| 第 0 步对 `target[0]` 的依赖 | `0.0` | 首字节未被“先看答案”泄露 |
| 第 1 步对第一字节前缀的依赖 | `0.25466` | 生成前缀实际推进 |
| 零步 bundle 恢复 | in-process 与独立进程 digest 相同 | 参数、store、cursor 完整恢复 |

结论边界：这些读数证明**机制接通**（时间、版本、读取、cut/rescue、恢复），不证明**学会内容选择**。零步的相关/无关比 1.57 明确低于可用判别力，不能当作新核心优势，也不能据此宣布旧图失败原因已排除。

## 4. 明确未实现

- `corrected_old_v1`（修正 byte 对齐/固定分母/时间后的旧图）与 `ordinary_v1`（同资源普通端到端基线）尚未实现。
- 训练循环、单位成本标定、数据分区求解器、数值门与预算尚未冻结（属重构-4）。
- 自然中文长上下文、真实多轮任务、跨会话过期/撤销/冲突合并尚未接入；当前 store 只覆盖顺序版本链与 owner/scope/session 边界。
- 答案正确性未评价：探针目标是任意字节，不是 fixture 的期望答案。

## 5. 出口与下一步

R3 的隔离实现与零步恢复门已通过；**下一唯一动作是实现 `corrected_old_v1` 基线**：按 R1 §2.1 先修 byte 对齐（第 t 步 logits 对第 t 个目标字节）、固定分母与 current/previous 选择，再用同一评分器与同一公开 fixture 产出基线读数。基线未接入前不做 Q1 比较，不进入训练。
