# M5 R2-H3.5 分层回答计划与渲染合同

> 状态：设计复审完成，等待进入隔离的 H3.5-A 最小可证伪实现；不代表 S2、L2、Mini 或正式 pilot 晋级。
>
> 日期：2026-09-16

## 1. 为什么必须改变表示合同

当前链路已经证明输入不是完全不可读：不同 prefix 会形成不同的 `motor_context` 和下一字节概率分布，保存恢复也可重复。但 H3.3-B/C 在相同 300k 总预算、相同数据和相同训练轮次下，无论只给首字节独立 owner，还是让同一 owner 承担完整 response，train/dev/final 的 exact 与 sequence 都为 0。H3.4 进一步把失败位置拆开：UTF-8 合法、无替换字符和 end-marker 边界均为 1.0，end-marker 位置 top1 也为 1.0；真正失效的是条件首字节和未见 continuation 的迁移。

这意味着当前 `prefix state -> 单步 byte 分布 -> 下一 byte 局部误差` 合同能够学习编码和停止，也能部分学习已见局部串，但没有一个在回答期间持续存在、用来约束“这次要回答什么”的原生状态。只增加 epoch、学习率或同构 readout，会继续把内容错误拆成互相竞争的局部 byte 误差，不能提供缺失的长程条件机制。

## 2. 选择与结论

| 选择 | 能解决什么 | 不能解决什么 | 决定 |
|---|---|---|---|
| A. 继续 byte 训练或扩大同构 readout | 可能提高已见串拟合和局部 margin | 没有新的回答级状态；H3.3/H3.4 已给出负证据 | 拒绝作为下一步 |
| B. 直接改成 token、字符或 codepoint 输出 | 缩短序列，减少 UTF-8 continuation 的无效权重 | 静态切分本身仍没有回答计划；会同时引入词表、分词与迁移变量 | 保留为以后渲染层对照，不先做 |
| C. 分层原生回答计划 + byte 渲染 | 把条件内容和表面实现分开；计划可持续、可消融、可恢复，渲染仍兼容现有原生链路 | 新增状态 owner、目标和预算，需要严格隔离验证 | **采用，进入 H3.5-A** |
| D. 先扩大共享 fabric | 可能增加表示容量 | 无法区分机制缺失与容量不足，且扩大回归面 | 仅当 H3.5-A 证明机制有效但容量受限时讨论 |

唯一推荐是 C。它不是把规则标签、`task_family` 或参考答案送进推理；运行时计划只能由模型读到的 system、policy、context、history 和 user prefix 产生。

## 3. v3 条件表示合同

### 3.1 三段原生因果链

1. **Prefix encoder**：沿用 v2 结构化 prefix 和 protected native predictive state，只读消费 system、policy、context、history、user 与 assistant boundary。
2. **Response planner**：在 assistant boundary 从 native prefix state 产生固定宽度 `response_plan_state`。它在一次回答内保持可读，可被后续生成状态更新或门控，但不得从 `task_family`、split、required terms、unknown markers 或参考 response 读取运行时 oracle。
3. **Byte renderer**：保留 byte 作为第一版表面实现和 end-marker owner。每一步分布同时读取当前 `motor_context` 与该回答的 plan state；renderer 的局部 byte 信用不得覆盖 plan owner。

编码/停止已经被证明可用，因此 H3.5-A 不重写 UTF-8 约束和 end-marker。它只检验“持久回答计划是否使未见条件 continuation 更可迁移”。

### 3.2 训练目标分账

训练时允许参考 response 生成监督目标，但监督只能用于更新 candidate，不能成为运行时输入。

- `L_plan`：回答级或片段级内容表示误差。首版采用确定性的、版本化的 response span 编码器生成训练目标；编码器只负责建立监督坐标，不参与推理生成，不作为能力来源。
- `L_render`：给定预测 plan state 的下一 byte 局部误差，继续由 renderer owner 承担。
- `L_boundary`：现有 end-marker/UTF-8 合法路径，保持独立观测，不用其高分掩盖内容失败。
- `L_retention`：candidate 相对 parent 的 protected 输出/状态保持项，只约束非目标漂移，不把 parent 错误答案当正确答案蒸馏。

必须分别报告四项，不允许只给合成 loss。正式实现前要冻结 span 编码格式、宽度、更新位置、学习率和权重；H3.5-A 不用人工语义标签，也不把 `required_terms` 反向写入模型。

### 3.3 状态生命周期与恢复

- plan state 只在显式 assistant boundary 创建；新 episode/reset 必须清空。
- plan state 属于 candidate checkpoint，保存其参数、配置、格式版本和必要的运行状态；parent checkpoint 不变。
- child 保存必须 atomic；执行训练前重跑 zero-save、fresh-process restore、一次更新、child restore、parent digest 不变和恢复后只读重复性。
- 生成、teacher forcing、保存恢复和只读评分必须走同一 plan owner，不允许评价时临时换 probe。
- plan candidate 与现有 response-start/response-phase candidate 首轮互斥，避免多个 owner 把收益归因混在一起。

## 4. 数据课程修订

现有 6/4/3 family-disjoint 集合可以继续做回归，但不足以单独判断回答计划。H3.5-A 新增一个小型、冻结、内容寻址的 v3 控制集，并满足：

1. train/dev/final 的 family 和完整 response 互斥；final 仅执行一次正式读取。
2. 每个 split 同时存在“共享首 byte/首 codepoint、不同内容”和“不同起点、同一意图结构”的配对，避免首字节支持集再次混淆结论。
3. `answer`、`say_unknown`、`clarify`、`refuse` 至少在 train/dev 有结构覆盖，但具体 response 和 family 未见；policy 仍通过 prefix 明文输入，不作 oracle。
4. 至少一组 history 改写和一组 context 置换；目标答案随有效证据变化，表面措辞扰动不应改变核心内容。
5. 记录 response byte 长度、codepoint 长度、span 数、首字节、首 codepoint 和内容重合度；不按 task label 把样本路由给不同 owner。
6. 当前已曝光 H3.3/H3.4 样本只作 regression/dev，不再承担新的最终泛化主张。

## 5. H3.5-A 最小可证伪对照

### 5.1 两臂与预算

- Control：现有 `response_phase` 合同。
- Treatment：`response_plan_state + plan-conditioned byte renderer`。
- 两臂使用同一父 checkpoint、数据顺序、种子集合、最大更新数和总 active 参数预算；首轮仍以 300k 为硬上限。candidate 参数必须在建模前由 core 预算中预留，不能训练后追加。
- 不改 shared fabric、默认 `SeedRuntime.chat`、P3b、Mini 或 P5.2d 文件。

### 5.2 必报指标

按 train/dev/final 分层报告：

- plan：同条件重复性、配对状态距离、正确目标相对负目标的 rank/margin、恢复前后 digest；
- content：首内容单位与 continuation 的 legal top1、rank、概率、累计似然；
- generation：原始 bytes、UTF-8、boundary、collision、exact、sequence criterion、required/forbidden/unknown policy；
- retention：parent 回归集状态/输出变化和 candidate 清除后的恢复；
- resources：core/candidate/effective 参数、峰值内存、墙钟、更新数和 checkpoint 大小。

### 5.3 通过、停止与失败归因

H3.5-A 不是 L2 门。它只在以下条件同时成立时证明新机制值得继续：

1. 全部 checkpoint preflight、parent 保护、预算、恢复和只读评价通过；
2. treatment 在 dev 的 plan target margin 与未见 continuation 指标相对 control 有方向一致改善，且不是只改善 end-marker；
3. final 不出现更高 collision，UTF-8/boundary 不退化，至少出现可归因的内容或 sequence 改善信号；
4. plan 消融会撤销该改善，证明收益来自持久计划而不是额外参数或评价规则。

出现以下任一项立即停止：checkpoint 无法完整保存恢复、parent digest 改变、effective 参数超预算、运行时读取标签/答案、只改善 train、只改善停止边界、消融不影响结果。失败按层回退：plan target 本身不可学回到目标编码；plan 可学但 renderer 不用它回到接口/信用；同机制随预算稳定改善才允许一次结构容量比较；没有机制信号则关闭该路线，不无限加宽。

## 6. 对主线和晋级的影响

H3.5 完成的是架构决策，不是模型能力交付。当前主线仍有意义：它已经建立了数据隔离、原生 owner、checkpoint、预算、恢复和可观测性，这些正是新架构能够被可信验证的底座。

下一步唯一执行项是 **H3.5-A 隔离 candidate 的实现前冻结包**：先把 v3 schema、plan target 编码、参数公式、两臂命令、数据 manifest、preflight 和停止门固化为可机读合同，再实现最小 candidate。冻结包通过后自动进入 plumbing smoke；只有 smoke 与 preflight 都通过，才允许小规模训练。Mini 继续后置，直到主线产生可加载且通过 L2 的 bundle。
