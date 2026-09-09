# M4.V2.R6 matched-control revision 预注册（2026-09-10）

## 1. 触发原因

旧的 9-cell aggregate 已完成，但报告为 blocked_aggregate。候选路径和 K3
lesion 均有真实结果，阻断来自两个控制臂的定义错误：

- frozen-parent 没有执行 K admission，new capability 为 null；
- matched-fixed-capacity 也没有 attach K worker，参数数和 inference trace 都为 0，
  既不是 same-capacity reference，也无法生成 paired capability delta；
- candidate 相对 matched 的 wall multiplier 为 5.9264×–6.8290×，全部超过已冻结
  的 1.5×，不能通过更改阈值掩盖。

本修订只修正 control semantics，不修改 candidate、K3 lesion、资源上限或 promotion
门槛；旧 manifest、旧 execution ledger 和旧 aggregate 报告保持不可变，作为一次被
阻断的实验记录。

## 2. 新 manifest 与适用范围

新 manifest：

plans/manifests/taiji_m4v2_r6_matched_control_v2_20260910.json

由 scripts/training/build_taiji_m4v2_r6_matched_control_manifest.py 从旧输入
manifest 内容寻址派生，保留同一 parent/worker/fixed-large registry、model seed、
course seed、source digest 和 holdout 矩阵，只增加
control_revision=taiji-m4v2-r6-matched-control-v2 与控制资源归属声明。
新 manifest digest 不得覆盖旧 digest；所有新报告必须携带新 digest。

## 3. frozen-parent：可观测零能力基线

frozen-parent 仍不 attach K worker、参数数仍为 0；但它要对同一 formal K holdout
执行一次显式 admission attempt，记录：

- K phase status=rejected，reason 为没有 K worker；
- new_capability.task_success_rate=0.0、sample_count=1；
- inference_trace_count=1 表示一次 formal admission attempt，不表示执行了 K
  worker inference；
- k_task_admission=false、k_feedback_consumed=false；
- S/G retention、parent restore 和 side-effect Gate 仍必须通过。

这个 0 是被测的 admission rejection，不是把旧的 null 后处理成 0，也不把
frozen-parent 当作资源等价 reference。

## 4. matched-fixed-capacity：同容量无反馈对照

matched-fixed-capacity 必须：

1. attach 与 candidate 完全相同的 content-addressed K1/K2/K3 worker bundle；
2. 使用相同 parent restore、holdout observation、输入路径、inference trace 和
   float32 参数字节；
3. 执行真实 Workbench read-only 路径，允许 K1/K2/K3 产生内部 projection；
4. 在 output admission 前关闭 feedback_admitted，不 bind dependency projection、
   不写 candidate stage、不创建 exchange、不更新参数；
5. 记录 task_success_rate=0.0，使 candidate-minus-matched 是真实 paired delta；
6. 记录相同 worker 参数、checkpoint path/bytes、inference trace、RSS、wall-clock、
   training_update_steps=0 和完整 rollback/side-effect 字段。

matched 的 0 表示同一容量路径主动拒绝新反馈输出，不表示 worker 没有被执行。其
资源 owner 改为 worker resource manifest；candidate/matched wall 和 peak 预算继续
使用原来的 1.5×/1.25× 上限。

## 5. 新执行顺序与停止线

先只运行 model17/course0 的 revised five-arm cell，检查：

- 新 manifest digest、parent/worker/fixed-large lineage 全部匹配；
- frozen-parent 与 matched 的能力值分别为 0.0/0.0，candidate 为 1.0；
- candidate-minus-frozen-parent 与 candidate-minus-matched 均为 1.0；
- matched worker parameter count/bytes、inference trace 与 candidate 一致；
- candidate/matched wall/peak 预算通过或如实阻断；
- no-feedback matched 不生成 exchange/candidate stage，旧能力和 parent namespace
  不变。

该 cell 通过后，按固定顺序扩展到其余 8 个 course；任一 cell 失败即停止，不删除
失败报告、不用均值填补。新 revision 仍保持
can_start_r6_formal=false、can_promote=false，不接 default runtime、provider、
MCP、client 或 CUDA。

## 6. Revised runner course1 closure（2026-09-10）

独立 revised formal runner 已创建 monotonic execution ledger，使用新 manifest digest
和独立 preflight 报告；model17/course0 与 course1 均已通过，当前累计 2/9。
prior-ledger 校验有效，旧 R6 ledger 不被覆盖。两个 cell 均满足 frozen=0、matched=0、
candidate=1、lesion=0 和两个 delta=1.0。

下一步只执行 model17/course2；通过后再进入 model23，期间不运行 aggregate、不接入
default runtime/provider/MCP/client/CUDA。

## 7. Revised model17 slice closure（2026-09-10）

revised runner 已完成 model17/course0、course1、course2，累计 3/9 且无 failure；
三个 cell 都满足 frozen=0、matched=0、candidate=1、lesion=0 和两个 delta=1.0，
same-capacity 参数/trace、no-feedback admission 和 parent/side-effect Gate 均通过。

下一步只执行 model23/course0；在 9/9 完成前不运行 revised aggregate，不覆盖旧
aggregate，也不接入 default runtime/provider/MCP/client/CUDA。

## 8. Revised model23/course0 closure（2026-09-10）

model23/course0 已由 revised runner 追加并通过，monotonic ledger 累计 4/9 且无
failure；该格 frozen=0、matched no-feedback=0、candidate=1、lesion=0，两个
capability delta=1.0，same-capacity 参数/trace、no-feedback admission、parent
namespace 和 side-effect Gate 均通过，fixed-large K-task-equivalent control 也通过。

下一步只执行 model23/course1；9/9 完成前不运行 revised aggregate、不覆盖旧
aggregate，也不接入 default runtime/provider/MCP/client/CUDA。

## 9. Revised model23/course1 closure（2026-09-10）

model23/course1 已由 revised runner 追加并通过，monotonic ledger 累计 5/9 且无
failure；该格 frozen=0、matched no-feedback=0、candidate=1、lesion=0，两个
capability delta=1.0，same-capacity 参数/trace、no-feedback admission、parent
namespace 和 side-effect Gate 均通过，fixed-large K-task-equivalent control 也通过。

下一步只执行 model23/course2；9/9 完成前不运行 revised aggregate、不覆盖旧
aggregate，也不接入 default runtime/provider/MCP/client/CUDA。
