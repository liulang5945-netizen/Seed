# M4.V2.R6 fixed-capacity parent admission addendum

> 版本：v1，冻结日期：2026-09-09。本文是
> [M4V2_R6_A8_PROMOTION_PREREGISTRATION_20260909.md](M4V2_R6_A8_PROMOTION_PREREGISTRATION_20260909.md)
> 的入口补充合同，不是 R6 formal 结果。它把当前审计结论落实成一个唯一
> 可执行边界：R6 研究同一 fixed-capacity parent 的连续能力保持与新增能力，
> 暂不研究结构增长晋级或 learned routing 晋级。

## 1. 决策与依据

本 addendum 选择 **fixed-capacity parent continuation**，不选择“继续调 R4
结构增长”或“先解冻 R5 router”。依据是已提交的真实报告：

| 证据 | 事实 | 入口含义 |
|---|---|---|
| R4 formal | G 对 fixed-large 的 non-worse 只有 `4/9`，`can_promote=false` | R4 structural growth 不进入 R6 candidate |
| R5 formal | hypothesis rejected；G non-worse `4/7`，S non-worse `6/7`，`can_promote=false` | learned router 保持 shadow |
| K scorecard v2 | K1/K2/K3 evidence closed，但 parent retention、default owner、same-parent course、resource/rollback/old-capability Gate 均缺失 | K evidence 只能作为 frozen course/benchmark 输入 |
| R6 adapter preflight | 13/13 checkpoint/lineage/rollback checks 通过 | 只能证明安全边界，不等于可训练或可晋级 |

上述报告在 admission audit 中以 content digest 固定：

- R4：`3b6e3c42183d4e476af697abaa997127fcf4ca23687ec589152ddab23ef55fa3`；
- R5：`33871b9ea5b3d94dc9ac63da0832b61bb5836531b158614ea322a263c5380212`；
- K scorecard v2：`3ebf33ca20eba92dc839341c8220fe160556adfabd57088b2d6aa194fcb44f38`；
- R6 adapter preflight：`5f92a57e8f136f60b4645f47f2f55b171a16ebffedc694b660050d1cd40384ae`。

## 2. owner 和结构边界

### 2.1 parent 是唯一生产候选的起点

- parent 使用已经存在的 R3/pressure fixed-capacity substrate；parent checkpoint
  必须是 immutable、fresh-restorable、content-addressed，并带 owner graph、
  source manifest、resource manifest 和 rollback capability。
- R4 residual growth candidate、R5 conditional route learner、random-growth
  和 fixed-large 只能作为隔离对照或 shadow artifact，不能写入 parent，也不能
  成为 default runtime owner。
- K1/K2/K3 的 learner/report 不能直接挂到 runtime；它们只提供冻结的
  successor task、输入输出 schema、outcome/dependency 课程和对照量尺。

### 2.2 R6 不使用 learned router

本课程冻结为 **no learned router**：

- R6 不消费 R5 的 conditional route gate，不重新选择 R5 的输入白名单，也不
  用 task ID、语言 ID 或 phase label 作为隐式路由键；
- K continuation 由固定的 Taiji-owned adapter boundary 进入同一 parent owner
  graph，路由责任由 typed observation/world/outcome schema 和 parent 的固定
  owner 拓扑承担；
- 如果后续证据证明固定拓扑无法表达 K 课程，必须另立 R7 router 预注册，不能
  在 R6 中偷偷引入 learned routing。

这样做不是声称 router 已经解决，而是把 R6 的因变量收窄为“连续学习和旧能力
保持”，避免把 R5 已经否决的结构假设混入新的 promotion 结论。

### 2.3 adapter 当前仍是 shadow boundary

`taiji/continual_k_adapter.py` 的职责冻结为：绑定一个 parent、绑定一个已接受
的 K3 dependency projection、保存 candidate trial、fresh restore 和 explicit
rollback。它当前不得：

- 执行 learner update 或产生新权重；
- 接管 default runtime dispatch；
- 绕过 parent digest、dependency scope、owner/source/resource manifest；
- 把 rollback 后的候选输出留在 parent namespace。

只有本 addendum 的 baseline 和 all-arm preflight 全部通过，才允许写 controlled
runtime smoke；只有 controlled smoke 和 R6 canary 都通过，才允许实现 formal
runner。

## 3. parent baseline 与 epsilon 校准

### 3.1 固定矩阵

沿用 R6 主合同的 `3 model seeds × 3 course seeds = 9 cells`：

- model seeds：`17, 23, 31`；
- course seeds/orders：`0, 1, 2`；
- 阶段顺序固定为 `S → G → K`；
- baseline、candidate、对照使用同一 source/holdout digest 和同一资源预算。

### 3.2 baseline repeat

在任何 candidate training 之前，对 immutable parent 做独立 baseline repeat：

- 每个 cell 做 3 个 frozen-parent repeats，repeat seeds 固定为 `401, 503, 607`；
- 每个 repeat 都做 parent checkpoint save、fresh restore、source digest 和
  metric digest 校验；
- 测量全部既有 S/G/K holdout，并保留 domain-level score、平均分、最差域、
  运行时间、峰值 working set 和输出 digest；
- baseline repeat 自身任一 checkpoint、digest 或环境边界失败，R6 停止，不能
  由 candidate 结果抵销。

### 3.3 epsilon 规则

不看 candidate 结果，先从 baseline repeat 的 paired absolute deltas 计算每个
既有 metric/domain 的波动：

```text
epsilon_raw = q95(abs(parent_repeat_i - parent_repeat_j))
epsilon = min(0.05, max(0.01, epsilon_raw))
```

若任何 metric 的 `epsilon_raw > 0.05`，说明 parent 本身不稳定，R6 不得进入
candidate training；必须回到 baseline/harness 修复。`epsilon` 写入冻结
baseline report 后不得增大。candidate retention 必须同时满足平均旧能力
`delta >= -epsilon`、关键 domain `delta >= -0.05`，并单独报告 exact-zero
retention、average forgetting、worst-domain forgetting、backward/forward
transfer。

## 4. arms 与课程顺序

每个 9-cell 固定以下 arms，所有 arm 从同一 parent digest 分叉：

1. `frozen-parent`：不写入新能力，用于 retention/noise baseline；
2. `matched-fixed-capacity`：同一资源和参数预算，只允许 parent owner 适应；
3. `candidate-continuation`：只使用冻结的 Taiji-owned K adapter boundary，
   不使用 R4 growth、不使用 R5 router；
4. `fixed-large`：参数/资源匹配的最强容量对照；
5. `lesion`：candidate/adapter/dependency feedback lesion，验证新增收益
   是否来自声明的 K boundary。

课程严格按 `S → G → K`：

- S：已有语义/技能基础，记录 old-capability baseline；
- G：已有 world/transition 课程，记录 old-capability retention；
- K：K1/K2/K3 successor 课程，真实 outcome/dependency 只从允许的 typed
  projection 进入，不把动态 digest 当可学习事实；
- 每阶段结束都重新测全部旧 holdout，任何阶段触发 catastrophic forgetting
  都立即停止后续阶段并恢复 parent。

## 5. admission Gate 顺序

下列 Gate 必须按顺序通过，后一 Gate 不得掩盖前一 Gate：

1. **source/parent Gate**：parent checkpoint、owner graph、source/resource
   manifest、fresh restore、rollback 和 digest 全通过；
2. **baseline Gate**：9 cells × 3 repeats 完成，epsilon 合法且 parent 稳定；
3. **adapter Gate**：K projection scope/lineage、same-parent candidate、
   candidate namespace isolation、rollback 后 parent restore 全通过；
4. **canary Gate**：单 cell 的 A/B/C/lesion 课程通过后，才允许写 formal runner；
5. **formal Gate**：9/9 cells 满足新能力、retention、lesion、resource、
   checkpoint/rollback 和副作用条件，且不允许通过 aggregate mean 抵销单 cell；
6. **runtime rollout Gate**：formal 通过后另行审查，仍不自动接 default runtime。

R6 主能力阈值沿用 K formal 的冻结量尺：candidate K holdout `>=0.75`，相对
`matched-fixed-capacity` 和 frozen-parent 的 paired delta 各 `>=0.25`；除此之外
还必须通过本 addendum 的 retention epsilon、fixed-large 资源归一化和 lesion
Gate。任何一个失败都保持 `can_promote=false`。

## 6. 资源与 checkpoint 合同

- peak working set 不得超过 matched parent 的 `1.25×`；
- wall-clock 不得超过 matched parent 的 `1.5×`；
- 训练更新步数、写入字节、候选参数字节和 inference trace 数量在运行前固定；
- 每个 arm 都必须在训练前保存 checkpoint、fresh restore 并校验 digest；
- candidate 失败时必须先写 failed artifact/rollback record，再恢复 parent；
- rollback 后旧能力输出必须与 rollback 前的 parent checkpoint 在合同容差内
  一致，候选 artifact 不能污染 parent namespace。

本阶段不要求 CUDA；CPU 是唯一受支持的执行设备。CUDA、provider、MCP、客户端
写入和联网都不是 R6 隐含变量。

## 7. 停止线与下一步

- R4/R5 shadow 被误接入 parent/default runtime：立即停止并恢复 parent；
- baseline epsilon 不可校准、parent 不稳定或 checkpoint 不能 fresh restore：
  停止，不训练；
- adapter smoke 不能证明同一 parent 或 rollback：停止，不写 formal runner；
- candidate 不优于 matched fixed-capacity/fixed-large，或 retention 超过
  epsilon：停止，不称为自进化；
- lesion 不影响新增收益：判定 boundary 没有被真实消费；
- 资源超限、CI/native 账本有相关真实失败：停止并回到 harness/合同修复。

因此 addendum 冻结后的唯一开发动作是：实现 **parent baseline repeat + all-arm
checkpoint preflight**，只做冻结 parent 的重复测量和 checkpoint/rollback 验证，
不训练 candidate、不接 default runtime、不引入新外围变量。
