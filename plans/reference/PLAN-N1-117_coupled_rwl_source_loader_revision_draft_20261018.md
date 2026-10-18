# PLAN-N1-117：协调式 RWL 训练器的来源载入修订

**状态：训练批次完成，原充分性门未通过（2026-10-18；`can_promote=false`）。** PLAN-N1-116 的首个来源加载尝试按原样封存；后续 N1-117 使用独立协议路径完成训练，不覆盖N1-116失败记录。

## 唯一修订

- 保留 N1-116 全部训练参数：同一 N1-110 长度平衡 schedule、相同九个端点/初始化/Adam/LR/更新次数、RWL共享位移半径、其他两组独立上限、CPU预算与原训练门。
- 改正 checkpoint 来源交接：九个 run83 初始化 bundle 按其自身 frozen update-policy identity 校验和恢复；之后写出新的零点 bundle 时，使用 N1-116 coupled policy 与新计划身份。仅识别旧 zero schema 才在验证期间临时恢复旧 policy；N1-117 zero/final bundle 一律使用 coupled policy 恢复。两种路径都必须完整保留模型、Adam、RNG和history。
- 通过纯函数合同测试验证 source-policy bridge 只作用于 legacy zero schema，调用后恢复当前coupled policy；N1-117新schema不得走兼容路径。静态runner readiness、source binding、九个zero checkpoint保存及九次独立进程恢复全部通过后，才开始正式训练。

## 冻结训练协议

- seeds 11/12/13 × `position`/`terminal`/`cut`，每个端点24,576次更新，总221,184次；N1-110同一192项和raw/hard长度平衡schedule，训练目标及模型架构不变。唯一优化器变化仍是N1-116的RWL共享半径。
- 九个zero checkpoint放入新目录 `data/checkpoints/n1_coupled_rwl_length_balanced_r1_20261018/`，不读取或覆盖N1-116空目录和其他已封存文件。zero preflight CPU≤300秒，正式批次总CPU≤18,000秒，均含独立恢复成本。
- 逐字保留 N1-110 的原 `training_sufficient` 判据：每个position seed的original/rename joint-four各≥44/48，rename transition free-exact≥40/48。任一失败即关闭批次，不追加训练、不查dev/holdout/confirmation、不建确认面，不晋升模型。
- 报告及attempt均使用N1-117独立write-once路径。N1-116失败attempt、空目录和失败报告只用于追溯，不删除、不覆盖。

## 执行结果（2026-10-18）

- 完整训练报告：[`report`](../../reports/taiji_n1_coupled_rwl_length_balanced_r1_20261018.json)，SHA-256 `1d4f43519790a1e36cb7dfbc580506d697815dccbdecb95fd576ec3a7bc2e391`；执行时冻结计划快照SHA-256 `07d32a1f9056198b81b2a6fb55ad3f4dda23f5512fde169997c17eaad429932c`。CPU9,396.21875秒/18,000秒；九模型221,184更新、497,664 QA sequences、43,200 train评分行。length ledger、schedule和LR审计全通过；18/18 zero/final完整bundle独立恢复相等，source guards与代码身份未变，CPU核算完整。
+- 三个position seed的原始/改名joint-four：seed11为48/48、48/48；seed12为48/48、48/48；seed13为24/48、23/48。seed13 rename transition为41/48，达到40门；joint-four两视图均未达到44门，所以批次`training_sufficient=false`。相较N1-110 seed13的5/48、4/48和transition38/48，coupled RWL策略在此固定训练面提高19、19和3项，但仍未过原门。
+- 九个端点均完成；dev、holdout、confirmation为0，未生成或查询新确认面，`can_promote=false`。此结果仅是既有train面上的训练充分性证据，不支持未见泛化或能力晋升。

## 只读轨迹归因（2026-10-18）

- 报告：[`trajectory attribution`](../../reports/taiji_n1_coupled_rwl_trajectory_attribution_20261018.json)，SHA-256 `a6de24cb407bab157991084f1c9914628db9267fffa7155db68cfc0e5ab784f6`；仅读N1-110/N1-117六个已封存position训练shard，CPU2.265625秒，0 checkpoint/model读取、查询、更新或保存。
- seed13/position原N1-110在cycle0/1/2的writer独立scale平均为0.5174/0.7383/0.9045，local为0.2287/0.2951/0.5493；对应writer/local clipped分别6138/6144、5786/6144、2326/6144与6144/6144、6144/6144、4787/6144。N1-117共同RWL scale则为0.9832/0.9941/0.9966，仅279/6144、129/6144、56/6144次触发；末周期几乎不裁剪。cycle QA loss从N1-110的1.9033/0.9103/0.2051/0.0214变为N1-117的2.0091/1.3087/0.8725/0.5287。
- seed13 final factorial rows逐父项对账：original joint-four从5升至24，N1-110的5项全部保留、另救回19项、无回退；rename从4升至23，保留3项、救回20项、回退1项（`train-078`），两种视图各有24个父项仍四条件全错。原始与改名逐项集合与训练报告汇总相符。
- 该描述性证据表明N1-117以reader的大参数范数设RWL共同绝对半径后，writer/local早期更新远少于旧独立cap；共享scale大多接近1，position QA loss下降更慢。它解释了局部改进但不充分的现象，不单独证明因果，也不支持未见泛化。

## 唯一下一步

[PLAN-N1-118](PLAN-N1-118_normalized_coupled_rwl_trust_region_draft_20261018.md) 将RWL三组各自的relative proposal按平方和归一化，并把共同半径设为√3，以避免reader参数大范数掩盖writer/local的独立relative cap。更新器合同、九端点静态schedule readiness和46项来源绑定已通过（4.5625 CPU秒，0模型/checkpoint加载、0查询/更新）。唯一下一步是保存并独立恢复九个zero bundle；全部相等后执行固定训练批次，原门与0留出查询保持不变。
