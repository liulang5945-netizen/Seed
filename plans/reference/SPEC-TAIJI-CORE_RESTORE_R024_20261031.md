# R024：09.2.4 完整主体恢复及下一更新合同

> 文档状态：[合同] — 冻结09.2.4在[R021参考图](SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)（含[R023门控出生](SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)）上的主体bundle保存/独立进程恢复范围。不授予任务训练、定价或能力结论。调度只见[MAINLINE](../MAINLINE.md)。

日期2026-10-31。`taiji/living_core.py`字节不变；bundle/恢复机构全部落在本合同声明的仪器内。09.2.1–09.2.4全部通过才进入定价/有界比较的合同决策。

## 1. 主体bundle schema（声明）

单文件`torch.save`（load端`weights_only=True`可读，仅含tensor/dict/list/标量）：

- 身份与谱系：`subject_id`、`revision`、`step`、`config`（LivingCoreConfig asdict）、`growth_parent_units`/`growth_unit_branches`（出生结构，未出生为None）、`growth_gates`（门控子每步[copy,generate]门值表，来自出生时父端点trace，声明为bundle携带的审计量）。
- 神经参数：`parameters`（全部named_parameters，含增长后的R/P）。
- 状态：`h`/`fast`/`nu_ema`/`coverage`/`prior`/`keys`（[source,position]对列表）。
- 优化器：`optimizer_state`（声明Adam(lr=1e-2, betas=(0.9,0.999), eps=1e-8)的state_dict）及其超参——**声明合成更新表达式的一部分，不是训练**。
- 资格迹替身：`eligibility`[n]，每消耗一步`e ← 0.99·e + mean_i‖x_s^i‖₂/√p`（纯记录量，不改前向；真实资格迹规则属09.4）。
- RNG：`rng_state`（torch全局CPU RNG状态）。
- 经历与游标：`career`（journal的声明step序列源数据）、`journal_cursor`（已消耗spec数）。
- 单调journal：追加式条目列表（seq严格递增），种类＝`outcome`（未结结果及其resolution）、`fee`、`probe_exposure`、`restore`；**恢复只追加，不改写、不截断**。
- 未结结果：`pending`（outcome_id、reward、resolved标志）。

## 2. 声明的生涯推进与更新

`career_step_specs(journal)`把生涯展开为显式step序列（prompt 8字节patch＋每target字节及EOS一轮，轮内事件由previous_byte原文经**当前编码器**重编码）；`resume_career(state, start)`从bundle游标续推，`run_career`等价于自规范初态start=0。**下一次更新**＝声明合成更新表达式：自保存状态以journal为教师强制重放数据（重放分支，不动生命游标）计算平均NLL，执行**一步**Adam；父端与worker用同一公式，逐位比较。

## 3. 保存边界（全部落账）

(a) 无新输入（生涯消耗完毕、无待处理事件）；(b) 首次学习后（声明一步Adam更新后）；(c) 出生后（R023门控子，游标居中、剩余事件待续）；(d) 编码迁移提交后（R023事务committed、revision+1）；(e) 未结结果挂起（pending未决议，journal中无其resolution条目）。

## 4. 独立进程恢复与逐位比较

保存→`scripts/training/r024_restore_worker.py`以`python -I`子进程加载bundle→按模式续推→落盘结果→父进程比较：

- **下一微步**：续推一个spec，h与probabilities与父端同续推**逐位相等**。
- **下一次更新**：worker执行声明更新后，参数digest、Adam矩、更新后再微步输出与父端**逐位相等**。
- RNG：bundle保存后父端与worker各自`torch.rand(3)`，逐位相等。
- 资格迹与费用：续推后eligibility、fee按声明公式延续，与父端逐位相等；**恢复不清零**。

## 5. 重放幂等与账本

- 未结结果resolution的唯一闸＝**当前journal中该outcome_id的resolution条目**（worker以`--current-journal`提交宿主当前账本）；无条目→决议一次、奖励入账一次、journal追加resolution；有条目→no-op，奖励与账本不变。旧快照重放不得绕过该闸。
- 恢复与续推**不重复领取奖励、不重复学习（更新只按声明次数执行）、不执行外部动作**；fee/probe_exposure只随声明新事件累加，任何路径不得清账。
- journal单调性：恢复后seq最大值只增不减；`restore`条目追加。

## 6. 出口与边界

- 出口：`passed_r024_restore_engineering_only`／`failed_bundle`／`failed_bitwise`／`failed_idempotence`／`budget_stop`／`engineering_failure`。
- 通过仅表示：五边形的bundle保存、独立进程恢复、下一微步/下一次更新逐位一致性、RNG/资格迹/费用延续与resolution幂等在该合成夹具上成立。**不签语义保持**（迁移恢复通过不替代旧能力/语义保持证明）、不签任务能力、不签主体自主性；更新表达式仍是合成仪器。
- 预算：全验证≤900 CPU秒/20墙钟分钟（含子进程）；超限`budget_stop`。
- 交付：本合同、state_migration扩展、worker、`tests/taiji_native/test_subject_restore_r024.py`、`scripts/training/verify_taiji_subject_restore_r024.py`、`reports/taiji_core_r024_subject_restore_20261031.json`（逐边界结果、digest、journal账、费用、源码SHA与R021/R022/R023绑定核对）。
