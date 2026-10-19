# PLAN-N1-140：提高两行基线暴露的匹配上下文续训校准

**状态：执行方案已批准并冻结。** 用户于2026-10-19批准本方案。执行进度与最终读数以当前执行页及独立结案报告为准；N1-139历史结论保持原样。

**批准记录：** 用户回复“批准并执行（推荐）”。获批草案原始SHA-256：435fb79c2da9758aa5ad84b77be527ce84d1b5d2c9af3677f8e152efababdffe。

## 依据

[N1-139结案](PLAN-N1-139_execution_closeout_20261019.md)完成55,296次更新及7,488条train-only计分。候选在增强上下文合计840/864，对照411/864；但原N1-132训练门只有5/6 endpoint通过，候选seed13 transition从48/48降至28/48。离线逐题配对显示，同一批错误在增强上下文下大多恢复，表明目前候选对添加上下文的适应强于对原两行输入的保持。

本方案只重新分配候选已有的transition训练slot：提高两行baseline比例，检验能否保留原门同时维持增强上下文表现。它是新的训练配方，不延续或补写N1-139失败endpoint，也不改N1-139历史结论。

## 拟议固定设计

- 沿用N1-132 seed11/12/13 position终点，完整复制模型、Adam、RNG与history；每seed建立context候选与两行控制，初始bundle逐位一致。父checkpoint哈希沿用PLAN-N1-139中的三个冻结哈希。
- 每分支固定9,216次更新，共六分支55,296次；目标函数、0.25·L_bind、优化器、学习率、任务类别及总更新slot不变。
- 每个候选seed的48个transition父项仍各出现48次，但slot改为每父项24次原始两行baseline，加六种增强条件各4次（8/16行×d=1/2/4）。raw与hard路径分别含12个baseline slot；每个增强条件在raw/hard各2次。每seed共有1,152个baseline slot及每条件192个增强slot。
- 控制臂全部保持原schedule两行输入。其他任务类别与slot在候选/控制间逐项一致。
- 增强事实保留目标事实与答案，目标事实仍位于索引0/1。hard路径按schedule中的实际改名目标构造精确Hamming距离干扰；使用v3/v5-v20排除集合，来源并集为9,354个已消费实体。仅用冻结train父项构造。
- 使用独立write-once目录data/checkpoints/n1_plan140_baseline_retention_balanced_matched_continuation_20261019/。训练前保存并独立恢复六份zero bundle；训练后保存并独立恢复六份final bundle。零点/终点均校验完整模型、Adam、RNG和history。
- 只对六个endpoint运行N1-132冻结960行原门及288条增强train-only视图，共7,488条计分；原门数字与合取规则不变，增强视图只作描述性读数。不访问dev、holdout或confirmation，不新建测面，不晋升。
- 本批次独立CPU上限7,200秒，覆盖来源核验、zero/final恢复、55,296次更新及全部train-only计分。N1-139同规模执行总计3,493.89秒，距3,600秒上限仅106.11秒；为避免新配方在同预算下因运行时波动而中途停止，本草案保留完整余量。来源、恢复、固定slot、更新数或CPU守卫失败即停止，不重试、不追加训练。

## 判读与授权边界

首要判读是六个终点是否满足完全不变的N1-132训练门，并按seed、分支和六种增强条件保留逐题字节结果。即使通过，也只表示train-only校准完成；未见泛化、确认面准入或晋升主张。

用户批准的授权限于本方案指定的baseline/增强slot配比、7,200秒CPU上限、55,296次更新及7,488条train-only计分。不得访问dev、holdout或confirmation，不得改变N1-132原门，不得自动晋升模型或追加训练。
