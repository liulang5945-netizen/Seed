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

## 6. Aggregate 实测结果与 matched 控制修订边界（2026-09-10）

冻结 aggregate 已实际运行并输出
reports/taiji_m4v2_r6_formal_aggregate_20260909.json。全部 9 个 cell/5 个 arm 的
字段和 lineage 可复核，execution ledger 为 9/9 executed_passed；candidate 的
task success、lesion causal 和 peak resource 结果分别通过 1.0、1.0 和
1.1091× < 1.25×。但是 candidate/matched wall multiplier 的九个值全部超过
1.5×，mean 为 6.5769×，所以 resource aggregate 不能通过。

同时，当前 matched-fixed-capacity 并非真正的 same-capacity reference：它的 K worker
没有 attach，worker_parameter_count=0、worker_parameter_bytes=0、
inference_trace_count=0，与 frozen-parent 一样不能产生 task success。因而
candidate - frozen-parent 和 candidate - matched-fixed-capacity 的能力 delta
均为缺失，而不是 0；聚合器按合同生成 18 个 capability_gate blocker，不进行插值。

后续控制修订必须保留本合同的 peak=1.25×、wall=1.5×、完整 9-cell 和无效 cell
不可删除规则。唯一允许的修订方向是预注册真正的 matched K：加载相同 worker 参数和
checkpoint、运行同一输入与 inference trace、保持 no-update/no-external-side-effect，
仅屏蔽 K feedback/output admission，使它成为可计算能力差值的容量对照。修订完成前
不计算新 promotion 结果，也不接入 default runtime/provider/MCP/client/CUDA。

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
model31/course2 已通过，9-cell ledger 已闭合；当前唯一下一步：**运行冻结的 aggregate
calculation**，验证完整 3×3、五臂字段、资源 Gate 与 source/resource digest 后输出
mean/min/max 和一侧 95% Student-t lower bound；不改变 promotion 状态。

## 7. Matched-control revision first-cell measurement（2026-09-10）

revision manifest 将 matched 的 resource owner 从 parent 改为 worker，并显式声明
frozen-parent admission baseline 与 matched worker attachment。model17/course0
实测：candidate/matched wall multiplier=1.2724×，peak multiplier=1.0024×，均通过
原有 1.5×/1.25× cap；candidate、matched worker 参数均为 4833/19332 bytes，
inference trace 均为 1，training update steps 均为 0。

该结果只说明修订后的一个 cell 满足资源/能力字段合同；不允许把单 cell 外推成
9-cell aggregate。后续仍使用全部 3×3、不可删除失败 cell、一侧 95% Student-t
lower bound 和原有阈值，按 model17/course1 开始的固定顺序继续执行。

## 8. Revised ledger course1 closure（2026-09-10）

修订版 runner 已按新的 content-addressed manifest 重放并单调追加 course0/course1，
2/9 row 均为 executed_passed；两个 cell 的 candidate/matched 参数均为 4833/19332
bytes、trace 均为 1，candidate/matched wall 与 peak 均通过 1.5×/1.25×预算，
candidate-frozen 和 candidate-matched 能力差值均为 1.0。

这仍不是 aggregate。下一步只执行 model17/course2，所有前序 row 必须保持
executed_passed；失败 cell 必须保留，不能用其余 course 的均值替代。

## 9. Revised model17 slice closure（2026-09-10）

model17 的三个 revised course cell 已连续通过，3/9 row 为 executed_passed；每格
candidate/matched wall multiplier 均低于 1.5×、peak 均低于 1.25×，参数字节和
inference trace 相等，candidate-frozen 与 candidate-matched delta 均为 1.0。

slice closure 不改变 aggregate 合同。下一步只执行 model23/course0，保留固定前序、
全部 arm 字段和不可删除失败 cell 规则。

## 10. Revised model23/course0 resource closure（2026-09-10）

model23/course0 已追加为第 4/9 个 executed_passed row；candidate/matched 均为
4833 parameters、19332 parameter bytes、1 条 inference trace，candidate/matched 的
wall 与 peak multiplier 均通过 1.5×/1.25× cap，candidate-frozen 与
candidate-matched capability delta 均为 1.0。fixed-large K-task-equivalent control
也通过，未发生 side-effect 漂移。

该结果只关闭一个 cell，不能外推为 aggregate。下一步只执行 model23/course1；保留
固定 predecessor、失败 cell 不可删除、不得用均值填补的规则，9/9 前不运行 aggregate。

## 11. Revised model23/course1 resource closure（2026-09-10）

model23/course1 已追加为第 5/9 个 executed_passed row；candidate/matched 均为
4833 parameters、19332 parameter bytes、1 条 inference trace，wall multiplier=1.1905×、
peak multiplier=1.0040×，均通过 1.5×/1.25× cap，candidate-frozen 与
candidate-matched capability delta 均为 1.0。fixed-large K-task-equivalent control
也通过，未发生 side-effect 漂移。

该结果只关闭一个 cell，不能外推为 aggregate。下一步只执行 model23/course2；保留
固定 predecessor、失败 cell 不可删除、不得用均值填补的规则，9/9 前不运行 aggregate。

## 12. Revised model23 slice closure（2026-09-10）

model23/course2 已追加为第 6/9 个 executed_passed row；model23 三格的
candidate/matched 均为 4833 parameters、19332 parameter bytes、1 条 inference trace，
wall multiplier 均低于 1.5×、peak multiplier 均低于 1.25×，每格 candidate-frozen 与
candidate-matched capability delta 均为 1.0。model23 slice 的 fixed-large
K-task-equivalent control 和 side-effect Gate 均通过。

slice closure 不改变 aggregate 合同。下一步只执行 model31/course0；保留固定
predecessor、失败 cell 不可删除、不得用均值填补的规则，9/9 前不运行 aggregate。

## 13. Revised model31/course0 resource closure（2026-09-10）

model31/course0 已追加为第 7/9 个 executed_passed row；candidate/matched 均为
4833 parameters、19332 parameter bytes、1 条 inference trace，wall multiplier=1.1799×、
peak multiplier=1.0024×，均通过 1.5×/1.25× cap，candidate-frozen 与
candidate-matched capability delta 均为 1.0。fixed-large K-task-equivalent control
也通过，未发生 side-effect 漂移。

该结果只关闭一个 cell，不能外推为 aggregate。下一步只执行 model31/course1；保留
固定 predecessor、失败 cell 不可删除、不得用均值填补的规则，9/9 前不运行 aggregate。

## 14. Revised model31/course1 resource closure（2026-09-10）

model31/course1 已追加为第 8/9 个 executed_passed row；candidate/matched 均为
4833 parameters、19332 parameter bytes、1 条 inference trace，wall multiplier=1.1603×、
peak multiplier=1.0030×，均通过 1.5×/1.25× cap，candidate-frozen 与
candidate-matched capability delta 均为 1.0。fixed-large K-task-equivalent control
也通过，未发生 side-effect 漂移。

该结果只关闭一个 cell，不能外推为 aggregate。下一步只执行 model31/course2；保留
固定 predecessor、失败 cell 不可删除、不得用均值填补的规则，9/9 前不运行 aggregate。

## 15. Revised 9-cell execution closure（2026-09-10）

model31/course2 已追加为第 9/9 个 executed_passed row；九格全部具备完整五臂资源
证据，candidate/matched 参数与 inference trace 均按同一 K bundle 对齐，所有
candidate/matched wall multiplier 均低于 1.5×、peak multiplier 均低于 1.25×，每格
candidate-frozen 与 candidate-matched capability delta 均为 1.0，且 fixed-large
K-task-equivalent control 与 side-effect Gate 全部通过。最后一格 wall=1.2513×、
peak=1.0027×。

现在才允许进入 aggregate validator 阶段，但不能直接使用旧 detached-control
aggregate。下一步只实现 revised manifest 专用 validator：检查 9/9 完整性、控制臂
语义、worker 资源等价、因果/保留/rollback、阈值和内容寻址；失败则保持 blocked，
不删除或填补任何 cell。

## 16. Revised aggregate validator closure（2026-09-10）

revised manifest 专用 aggregate validator 已通过，报告为
reports/taiji_m4v2_r6_matched_control_aggregate_20260910.json。9/9 row 均为
executed_passed，所有 cell 的控制臂语义、worker-owned matched resource、参数/trace
等价、rollback/side-effect、candidate floor、K3 lesion、wall/peak cap 均通过；两组
paired capability delta 均 n=9、mean=1.0，单侧 95% lower bound=1.0，blocking_failures
为空。

下一步不是 promotion，而是独立 formal-admission gate：只审计该 aggregate 与其
manifest/registry 的 lineage、checkpoint、owner graph、side-effect 和阈值不变性。
审计通过前不启动 formal、不覆盖旧 aggregate、不接 default runtime/provider/MCP/
client/CUDA。
