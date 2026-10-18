# PLAN-N1-117：协调式 RWL 训练器的来源载入修订

**状态：按 owner 持续推进授权进入零更新工程、保存恢复前置及同配方训练。** PLAN-N1-116 的候选更新器和静态 schedule 验证均已通过；其 checkpoint 前置在读取首个 run83 fresh-zero 来源时失败于旧协议的 update-policy identity 守卫。错误回执见 [N1-116 preflight failure](../../reports/taiji_n1_coupled_rwl_length_balanced_training_20261018_checkpoint_preflight_failure.json)，原 attempt 保留，N1-116 不重试。

## 唯一修订

- 保留 N1-116 全部训练参数：同一 N1-110 长度平衡 schedule、相同九个端点/初始化/Adam/LR/更新次数、RWL共享位移半径、其他两组独立上限、CPU预算与原训练门。
- 改正 checkpoint 来源交接：九个 run83 初始化 bundle 按其自身 frozen update-policy identity 校验和恢复；之后写出新的零点 bundle 时，使用 N1-116 coupled policy 与新计划身份。仅识别旧 zero schema 才在验证期间临时恢复旧 policy；N1-117 zero/final bundle 一律使用 coupled policy 恢复。两种路径都必须完整保留模型、Adam、RNG和history。
- 通过纯函数合同测试验证 source-policy bridge 只作用于 legacy zero schema，调用后恢复当前coupled policy；N1-117新schema不得走兼容路径。静态runner readiness、source binding、九个zero checkpoint保存及九次独立进程恢复全部通过后，才开始正式训练。

## 冻结训练协议

- seeds 11/12/13 × `position`/`terminal`/`cut`，每个端点24,576次更新，总221,184次；N1-110同一192项和raw/hard长度平衡schedule，训练目标及模型架构不变。唯一优化器变化仍是N1-116的RWL共享半径。
- 九个zero checkpoint放入新目录 `data/checkpoints/n1_coupled_rwl_length_balanced_r1_20261018/`，不读取或覆盖N1-116空目录和其他已封存文件。zero preflight CPU≤300秒，正式批次总CPU≤18,000秒，均含独立恢复成本。
- 逐字保留 N1-110 的原 `training_sufficient` 判据：每个position seed的original/rename joint-four各≥44/48，rename transition free-exact≥40/48。任一失败即关闭批次，不追加训练、不查dev/holdout/confirmation、不建确认面，不晋升模型。
- 报告及attempt均使用N1-117独立write-once路径。N1-116失败attempt、空目录和失败报告只用于追溯，不删除、不覆盖。

## 唯一下一步

实现N1-117独立source-policy bridge和runner，先通过合同测试与零查询schedule readiness；随后保存并独立恢复九个coupled-policy zero bundle。全部相等后执行一次固定训练批次，再按原门结案。
