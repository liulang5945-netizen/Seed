# M4.V2.R6 9-cell resource validity / aggregate 预注册（2026-09-09）

> 本合同建立在 `model_seed=17 / course_seed=0` 的真实 paired single-cell ledger
> 之上。它只冻结资源字段、validity、聚合和停止线，不授权在合同冻结前运行
> 9-cell full formal，也不改变 candidate 的新能力阈值。

## 1. 已有 single-cell 基线

candidate 与 native fixed-large 已在同一真实 `typescript_05.ts` read 上完成
K1/K2→read-only planner→Workbench→K3→adapter exchange/stage/rollback。两臂都记录：

- `measurement_complete=true`；
- `process_rss_before_after_lower_bound` peak-RSS 方法；
- wall-clock、worker parameter count/bytes、checkpoint-write bytes、inference
  trace count、training update steps；
- `fixed-large - candidate` 的逐字段 paired delta。

这只证明字段合同在一个 cell 可执行，不把单 cell 当作 9-cell aggregate。

## 2. 每个 cell/arm 的必需字段

每个 `model_seed × course_seed × arm` 必须写入同一 `resource` 对象：

```text
device
resource_manifest_digest
wall_clock_seconds
peak_working_set_bytes
peak_working_set_method
training_update_steps
worker_parameter_count
worker_parameter_bytes
candidate_parameter_bytes
checkpoint_write_bytes
inference_trace_count
measurement_complete
```

`candidate_parameter_bytes` 在 fixed-large 上表示该 arm 的 native K 参数字节，保留
历史字段名以兼容现有 ledger；`worker_parameter_bytes` 是规范名称。任何缺失、
`None`、非有限值、设备不为 CPU 或 digest 不一致都使该 arm `resource_gate` 失败，
不得以 0、均值或另一 arm 的数值填补。

统一测量规则：

1. `peak_working_set_bytes = max(RSS_before, RSS_after)`，方法固定为
   `process_rss_before_after_lower_bound`；没有可用 RSS 时记录 environment blocker，
   不伪造峰值；
2. `checkpoint_write_bytes` 是该 arm 本次实际落盘的 checkpoint/artifact 文件字节
   总和，写入前冻结路径集合并在报告中列出；
3. `worker_parameter_bytes = worker_parameter_count × 4`，CPU float32 native
   参数按实际 owner checkpoint 统计，不把 runtime 对象或 JSON 报告大小算入参数；
4. `inference_trace_count` 只统计真实执行的 holdout trace；`training_update_steps`
   统计实际参数更新，不把 restore、projection 或 adapter stage 算成训练；
5. 所有 arm 使用同一 process、CPU、线程/worker 配置和 resource manifest；不能用
   CUDA、provider、MCP、client 或网络改变资源口径。

## 3. validity 与预算

输入 manifest 已冻结：`device=cpu`、`cuda_required=false`、peak multiplier cap
`1.25`、wall-clock multiplier cap `1.5`。R6 formal 沿用该预算：

- `matched-fixed-capacity` 是每个 cell 的资源 reference；candidate 的 peak RSS
  不得超过 matched reference 的 `1.25×`，wall-clock 不得超过 `1.5×`；
- fixed-large 是容量对照，必须完整记录其资源和参数，不允许隐藏其超出 candidate
  的容量；它的用途是 strongest-capacity paired comparison，不替代 matched budget；
- frozen-parent、matched、candidate、fixed-large 和 lesion 都必须有完整字段；
  lesion 不得因为不产出新能力而省略资源/side-effect ledger；
- 任一 arm 资源无效、超预算、checkpoint 写入集合漂移或 source/resource digest
  断裂，当前 cell 立即归类 `resource_gate`/`lineage` 并停止；不把失败 cell 删除后
  继续 aggregate。

## 4. paired 与 9-cell aggregate

每个 cell 计算 candidate 相对 frozen-parent、matched-fixed-capacity 和 fixed-large
的绝对能力/旧能力/资源 paired delta。资源报告必须同时输出：

- `fixed-large - candidate` 的 wall-clock、peak RSS、parameter bytes、checkpoint
  bytes、inference trace delta；
- candidate 相对 matched 的 peak/wall multiplier；
- `task_success_delta` 和资源归一化诊断值（只作诊断，不替换预注册主 Gate）。

9-cell aggregate 固定使用全部 `3×3` cell：输出 arithmetic mean、min、max、每个
cell 明细和一侧 95% Student-t lower confidence bound（`df=8`）；方差为零时 lower
bound 等于 mean。任何 cell 缺失、无效或提前停止时 aggregate 状态为
`blocked_aggregate`，不能用剩余 cell 的均值补齐。

新能力的既有 Gate 不变：candidate 每个 K holdout 至少 `0.75`，相对 frozen-parent
和 matched-fixed-capacity 的 paired delta 各至少 `0.25`，K3 lesion 必须破坏声明的
新增收益；fixed-large 只作为 strongest-capacity 对照和资源归一化诊断，不能事后
从结果选择性改成晋级阈值。

## 5. 停止线

- model23/31 没有对应的 fixed-large artifact、owner/source/resource manifest 或
  fresh restore：停止 full formal；
- 资源字段不完整、测量方法改变、RSS 不可得且未被记录为 environment blocker：停止；
- 任一 cell 资源超预算、checkpoint/parameter/inference 账本不一致：停止该 cell，
  保留失败 artifact，不放宽预算；
- 任何 full formal 结果试图在 `can_promote=false` 下接 default runtime、provider、
  MCP、client 或 CUDA：停止并归类 side-effect gate。

model17/23/31 的 fixed-large artifact、source/resource manifest、fresh restore 和
non-overlap 已全部纳入 content-addressed formal input registry，preflight 已通过。
当前唯一下一步：**实现 model17/course0 的单 cell execution contract**，在不扩大矩阵
的前提下逐臂落盘 S→G→K、causal/resource/side-effect/checkpoint/rollback 与失败归因；
该 cell 的五臂 resource ledger 已闭合，下一步改为**将这份执行合同抽为可复用 cell
executor，并用同一 model17/course0 重放做字段级一致性检查**；通过前不运行其余 8
cells，不接 default runtime、provider、MCP、client 或 CUDA。

可复用 executor 与 replay 已完成：两次 model17/course0 的结构化
`execution_contract_digest` 一致，五臂 resource Gate 全部通过。当前唯一下一步：**让
formal runner 消费一个 `not_started` ledger row 并写回 executor 结果与 failure
attribution**；该 row 已闭合。当前唯一下一步：**只执行下一个预注册
`model17/course1` row**，继续验证同一资源/失败合同；任何失败都保留并停止，不用其余
cell 的均值覆盖。course1 已闭合且 prior ledger 保持单调；当前唯一下一步：**只执行
`model17/course2` row**，完成 model17 的 course slice 后再评估是否开放 model23/31。
course2 已通过，model17 的三行均为 `executed_passed`；runner 已冻结完整的 9-cell
前序顺序并拒绝 predecessor 未通过时的跳跃。当前唯一下一步：**只执行
`model23/course0` row**，不计算 aggregate、不用其他 cell 抵销失败。
model23/course0 已通过且追加，当前唯一下一步：**只执行 `model23/course1` row**，继续
按同一 resource/side-effect/failure contract 记录，仍不计算 aggregate。
model23/course1 已通过且追加，当前唯一下一步：**只执行 `model23/course2` row**，完成
model23 的 course slice 后再推进 model31。
model23/course2 已通过且追加，当前唯一下一步：**只执行 `model31/course0` row**，继续
沿用相同 resource/side-effect/failure contract，仍不计算 aggregate。
model31/course0 已通过且追加，当前唯一下一步：**只执行 `model31/course1` row**，继续
沿用 prior ledger 和相同资源/失败合同。
model31/course1 已通过且追加，当前唯一下一步：**只执行最后的 `model31/course2` row**，
完成 9-cell ledger 后再按冻结合同计算 aggregate。
