# PLAN-N1-116：协调式 reader/writer/local 更新保持

**状态：工程前置停止，训练未开始（2026-10-18；`can_promote=false`）。** 本计划沿用 N1-110 的训练面与充分性门，只改变 RWL 三组参数的步幅约束。zero-checkpoint preflight 在读取来源模型时被旧 update-policy identity 守卫拒绝；保留失败 attempt 与空目标目录，不在 N1-116 下重试。

## 依据与假设

- PLAN-N1-115 的 32 态交叉显示：reader、writer、local 任一候选组单独替换到 baseline 背景，original/rename 的 joint-four 都降为0/48；context仍48/48，relation为40/48与36/48。整体候选为5/48与4/48。
- 对12,288行的逐父项离线归因显示，reader/writer/local分别在16个匹配背景上净损害相同的48个父项；它们的两两 joint-four 二阶交互非负，且常为正。该事实支持检验耦合更新，但不证明组裁剪是退化原因。
- 现行 Adam proposal 对五组分别缩放。候选长度平衡运行的 N114 归因还显示多组裁剪频率和步幅发生变化。假设是：reader/writer/local各自独立裁剪可能破坏共享的补偿方向；本批测试一个共同信任半径能否保住协同适配。

## 冻结配方

- 复用 PLAN-N1-110 同一 192 项训练来源及长度平衡 schedule，不改内容、父项、顺序、paired-question 目标或学习率。仍训练 seed 11/12/13 × `position`/`terminal`/`cut` 九个 fresh endpoint，每个24,576次更新，总计221,184次；初始化、Adam、矩估计、训练评价方式及其余协议与 PLAN-N1-110 相同。
- 唯一配方改动：Adam 正常提出完整参数步后，将 reader/writer/local 作为一个耦合块。令三组拼接参数为 θ，未投影 Adam 位移为 Δ，则共享上限为 `0.001 × max(||θ||₂, 1)`，并给三组的位移统一乘 `min(1, 上限 / ||Δ||₂)`。这保留三组 proposal 的相对方向。context 与 relation 继续分别使用既有相对位移上限。所有 Adam 一阶/二阶矩、梯度和 RNG 原样保留；不得逐组二次投影 RWL。
- 先完成纯计算合同，验证五组完整互斥覆盖、RWL共享缩放对三组同乘、shared-block位移不超上限、context/relation旧上限不变、Adam moments不变、`relative_limit=0` 等无效输入被拒。
- 正式训练前，将九个 fresh zero endpoint 保存到新目录 `data/checkpoints/n1_coupled_rwl_length_balanced_20261018/`，并做九次独立进程完整 bundle 恢复检查（模型、Adam、RNG、history及协议）。恢复不等、来源守卫或 runner readiness 有一项失败，就不开始训练。
- 正式批次 CPU 上限18,000秒，zero-checkpoint与恢复前置上限300秒计入总预算。九端点一次完成，不接续或因结果失败重试；报告与checkpoint路径 write-once。

## 判据与隔离

- 原样调用 PLAN-N1-110 的 `training_sufficient` 判据：三个 position seed 的 original 与 rename `joint_four_exact` 各自至少44/48，且 rename transition free-exact至少40/48。不得只看合计或改宽门槛。失败则停止，不生成范围面或确认面。
- 本批只用既有 train schedule 与 N1-110 train-only评分。dev、holdout、confirmation查询均为0；不生成或查询新面，不改变能力或晋升状态。全部通过只表示这项训练配方过原训练门，不能宣称未见泛化。
- 新代码、plan、schedule、初始化来源和输入报告全部纳入 SHA-256 source guards；输出checkpoint不覆盖任一既有目录。

## N1-116 前置结案

- `--validate-only` readiness已通过：模拟24,576个schedule位置，CPU4.40625秒，0模型/checkpoint加载、0查询/参数更新；耦合投影4项合同测试通过。
- checkpoint-preflight attempt [`receipt`](../../reports/taiji_n1_coupled_rwl_length_balanced_training_20261018_checkpoint_preflight_attempt.json)，SHA-256 `5d3034853bed21cfcca6e962af2f9a5482d69419188a2f673c3544e7a056518c`。它在读取首个run83 zero来源时，以`strict fresh initialization marker, phase, rule and identity required`停止；新checkpoint保存数0、训练更新0、已知模型查询0，目标目录为空。
- 独立离线故障报告 [`failure audit`](../../reports/taiji_n1_coupled_rwl_length_balanced_training_20261018_checkpoint_preflight_failure.json)，SHA-256 `731cedcdfe8e0ccd7be97d22cc81eefab20be6bb3f5d7d1ac78cd54f2d20914e`。根因是加载旧run83 zero时，旧loader将其update-policy与已切换的coupled policy比较；不是模型/optimizer/RNG恢复不等。

## 唯一下一步

由[PLAN-N1-117](PLAN-N1-117_coupled_rwl_source_loader_revision_draft_20261018.md)使用独立report/checkpoint路径，针对legacy zero schema临时桥接旧policy完成来源校验；新的coupled checkpoint仍严格用新policy恢复。其静态schedule readiness与source binding已通过；下一步执行九个zero checkpoint保存及独立恢复前置，成功后运行固定221,184更新批次。