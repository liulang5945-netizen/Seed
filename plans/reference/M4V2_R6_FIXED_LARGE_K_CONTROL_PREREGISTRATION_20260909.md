# M4.V2.R6 K fixed-large control 预注册（2026-09-09）

> 本文件修正 R6 single-cell 暴露的 control mismatch。它只冻结 K-task-equivalent
> fixed-large 的输入、结构和停止线，不把旧 R4 structural shadow 转译成 K 结果，
> 也不授权 full formal 或 promotion。

## 1. 为什么必须修正

当前 `fixed-large` 资产来自 R4 structural shadow：它验证的是 residual/topology
capacity 的 checkpoint、shadow 和 rollback，不消费 K1 semantic、K2 transition、
真实 Workbench outcome 或 K3 dependency。R6 candidate 验证的却是：

```text
K1 semantic → K2 transition → read-only Workbench → real outcome → K3 dependency
```

因此把 R4 fixed-large 的分数放进 R6 K causal comparison 会产生错误归因：它既不能
证明 K worker 的能力，也不能成为同输入/同任务/同 resource 的容量对照。2026-09-09
single-cell runner 已按 `input_contract / fixed_large_k_control_required` 阻断，
没有放宽判据。

## 2. 冻结的 native fixed-large 定义

R6 的 fixed-large 必须是 Taiji native K control，而不是 Transformer/provider、
R4 structural shadow 或 learned router：

- **同一 parent**：使用与 candidate 完全相同的
  `parent_checkpoint_digest`、owner/source/resource lineage 和 candidate namespace
  隔离；
- **同一任务输入**：使用相同 course registry、workspace root 隔离、语言 registry、
  holdout observation/input digest、S→G→K 顺序和真实 read-only Workbench；
- **固定大容量**：K1 semantic 与 K2 transition 各使用两个独立、可恢复的 native
  worker replica，`ensemble_width=2` 在运行前冻结；K3 仍为同一 deterministic typed
  projection，不增加反馈路由；
- **确定性 ensemble**：对两个 worker 的 typed score map 做算术平均，再按现有
  native result contract 解码为一个 `StructuredSemanticResult` /
  `StructuredSemanticTransitionResult`。不按任务 ID、结果、置信度或列表顺序选择
  worker，不引入 learned router；
- **独立 worker course**：两个 replica 必须来自预注册、与 formal holdout 不重叠的
  worker-training task slices（初版固定为 task seed `3` 和 `4`），并记录各自
  source/checkpoint/artifact/bundle digest。formal course seed `0/1/2` 的 holdout
  不得进入 fixed-large training；
- **同一执行身体**：candidate 与 fixed-large 都只能经 native read-only intent
  planner 和隔离 Workbench 执行，不能用 provider/MCP/client/network 补能力。

这个 control 的“large”只表示固定的 native K worker capacity，不表示父代 topology
增长；candidate 仍是 single K bundle continuation，二者都不接 default runtime。

## 3. Artifact 与 checkpoint 合同

每个 fixed-large cell 需要一个 content-addressed ensemble envelope：

```json
{
  "format": "taiji-k-fixed-large-ensemble-v1",
  "version": 1,
  "parent_checkpoint_digest": "sha256...",
  "candidate_namespace": "taiji:k:fixed-large:model-17",
  "ensemble_width": 2,
  "worker_training_task_seeds": [3, 4],
  "k1_replicas": [
    {"artifact_digest": "sha256...", "worker_checkpoint_digest": "sha256..."},
    {"artifact_digest": "sha256...", "worker_checkpoint_digest": "sha256..."}
  ],
  "k2_replicas": [
    {"artifact_digest": "sha256...", "worker_checkpoint_digest": "sha256..."},
    {"artifact_digest": "sha256...", "worker_checkpoint_digest": "sha256..."}
  ],
  "k3_projection": {"artifact_digest": "sha256..."},
  "input_contract_digests": {},
  "output_contract_digests": {},
  "owner_graph_digest": "sha256...",
  "ensemble_digest": "sha256..."
}
```

构建前必须分别保存/回读四个 K1/K2 worker checkpoint 和 K3 checkpoint；构建后必须
fresh restore ensemble envelope，并验证：

- 两个 replica 的 parent/source/resource/candidate namespace 合同正确；
- K1/K2 score-map key 集合、shape 和 typed output contract 完全相同；
- arithmetic mean 的输入/输出 digest 可重放；
- fixed-large 不携带 optimizer state，不修改 parent namespace；
- 训练 task seed `3/4` 与 formal holdout `0/1/2` 的 source/observation digest
  无交集。

## 4. Single-cell Gate

实现完成后，先只运行 `model_seed=17 / course_seed=0`：

1. fixed-large ensemble preflight、fresh restore 和输入 digest Gate；
2. 与 candidate 相同的 one-step real Workbench read；
3. K1/K2 typed result、K3 projection、dependency lineage、exchange/checkpoint；
4. fixed-large candidate namespace stage/rollback 和 parent retention；
5. wall-clock、peak working set、worker parameter bytes、checkpoint write bytes、
   inference trace count；
6. fixed-large lesion（至少切断一个 K replica 或 ensemble branch）必须使声明的
   large-capacity contribution 可观测，且不能改变 corpus/threshold。

single-cell 不通过时按 failure contract 停止，不跑 9 cells。通过也只说明 control
可比，不代表 candidate 晋级。

## 5. 与 candidate 的比较边界

candidate 与 fixed-large 必须逐 task paired 记录：

- K task absolute success；
- candidate vs matched-fixed-capacity、candidate vs frozen-parent、candidate vs
  fixed-large 的 delta；
- S/G/K old-capability retention 和 rollback 恢复；
- parameter/checkpoint/resource/inference trace；
- K3 full-feedback 与 lesion 的差值。

R6 原有 candidate 主阈值不因新增 control 放宽：candidate K holdout `>=0.75`，
candidate 相对 matched-fixed-capacity 与 frozen-parent 的 paired delta 各 `>=0.25`，
且 lesion 必须破坏声明的增益。fixed-large 先作为同任务容量/资源对照报告；是否把
candidate-vs-fixed-large 加入 promotion 必要条件，必须在看到结果前另行写入正式
Gate，不能事后选择性使用。

## 6. 停止线

- 不能证明 formal holdout 与 worker-training task slices 不重叠：停止；
- ensemble score map 无法保持 typed contract 或平均过程不可重放：停止；
- 需要 learned router、任务 ID、provider、MCP、client 或 Transformer 才能完成：
  停止并退回架构设计；
- fixed-large 资源超出预注册预算：保留失败 artifact，不改预算；
- fixed-large 仍只是 structural shadow：继续 `fixed_large_k_control_required`，不
  得进入 R6 formal aggregate。

当前唯一下一步：**实现 native K fixed-large ensemble artifact builder、checkpoint
preflight 和 model17/course0 single-cell comparator；完成前 R6 formal 仍关闭。**
