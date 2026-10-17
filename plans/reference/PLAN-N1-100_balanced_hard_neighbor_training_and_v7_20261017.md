# PLAN-N1-100 · 平衡近邻实体训练与全新 v7 确认 · 2026-10-17

**状态：由 owner 2026-10-17“最高权限按最合适下一步推进”的持续授权覆盖；已冻结，执行中。**目标是让模型在新实体名与近邻干扰下学会依据目标实体是否有证据作答，同时保留 lookup、revision 与 transition 的已有行为。此计划保留原训练充分门、v6/PLAN94 四门和动态范围门，不降阈值，不晋升默认模型。

## 0. 证据与问题

- [EXP-N1-99](EXP-N1-99_unknown_abstention_factorial_diagnostic_adjudication_20261017.md) 的已知面 2×2 探针显示：position 对 v6 的 d=1 unknown 项为 0/15，d=2 在 2/4 观测格都只为 1/15；观测数单独没有抬升 position。它不足以作能力或因果结论，但说明局部 d=2 放宽不是解法。
- run88 训练 source attempt `reports/taiji_n1_transition_factorial_paired_train_20261016_attempt.json`（SHA `6a3124a1293c33dcbdc45bc55fb34971200236cb4673e52c9c981335e0bc6234`）绑定的 native evidence 组件之一是 v3 curriculum（SHA `ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604`）。train-only 摘要见 [train exposure audit](../../reports/taiji_n1_hard_neighbor_train_exposure_audit_20261017.json)：192 个 train 项按四类各48；unknown 48 项全是 2 观测／3 实体，目标实体均不在观测里，最近干扰名 Hamming 距离 8–10，d≤2 训练实体对为0。该证据只描述这个被 run88 锁定的训练组件，不声称覆盖其余所有训练源。

## 1. 固定训练数据与变换

- 只从 v3 curriculum 的 `split=train` 取192项；不得读取 dev/holdout 内容。提取器只解码 `train` episode；对其他 episode 仅读 `id` 与 `split` 前缀以跳过对象，其余字段不解析、不使用、不汇总。原 v3 manifest、run88 source attempt、run83 preflight 提供的九枚 full-fresh zero checkpoints、paired-question 与 renaming 来源按 SHA 逐项绑定，原始文件不改。run88c 只有 position 三枚 zero，不能作为九模型的统一初始化来源。
- 训练每6144步保持现有偶／奇配对节奏：偶数步直接用原始 train 项；奇数步用同一 parent 的近邻实体 hard view。每24,576步共12,288次原始视图与12,288次 hard view；同一个确定性3072项 hard schedule 循环4次。
- 每个 parent 在每个6144步周期有16个 hard-view slot。lookup 与 unknown 的16个 slot 各4次覆盖完整2×2：观测数2/4 × Hamming d=1/2。revision 使用4观测且 d=1/2 各8次。transition 保留当前两观测 renaming 输入，因为其冻结 quartet 学习器要求该二事实结构。
- hard view 先使用现有 train-only renaming stream 提供新的 query entity；其余干扰实体按 PLAN-N1-93 同一逐字符替换方式生成，名字长度10、距离精确为指定 d，生成随机种子固定为10002。未知 query 的目标实体在任何观测中都不出现；lookup 保留一条正确目标观测，revision 保留 stale→current 两条目标观测且顺序不变；相应类别再加入足够干扰观测达到4条。hard views 的实体身份和 observation slots 必须与其 parent 一一绑定。
- 新增干扰材料值只能从 train split 已有 `uses` 值池采样，排除本题正确答案与已有值；transition hard-view 不变。lookup／unknown 的 hard-view 仍用当前 `second_question` 规则，从实际存在的干扰观测派生 paired query。target-row 放置和 revision 两行相对顺序机检平衡。
- 预生成 `plans/manifests/n1_hard_neighbor_training_schedule_v1_20261017.json`（仅3072个 train-derived hard schedule 条目）并计算 SHA。任何 split、parent、距离、答案、target-presence、数目、顺序或实体冲突不合约，均在首个 optimizer update 前拒绝。

## 2. 训练合同

- 九个 fresh checkpoints：seeds 11/12/13 × position/terminal/cut；从既有完整空 Adam/RNG 零态开始，不继承任何训练权重。架构、参数数、Adam、五组 relative-update 规则、paired question / observation-fidelity objective 均沿 PLAN85/88；学习率前18,432步0.001，末6,144步0.00025，Adam moments continuous。
- 每模型24,576 updates；目标QA/训练会话与已知评分视图沿用冻结 runner。因 hard views 的观测量增加，实际QA/会话/CPU计数必须由执行器现取；协议预期若与实测不符，停止并更新报告，绝不把旧总数硬写进去。
- 正式训练前，对九个 seed/arm 独立完成 zero checkpoint 的保存、跨进程恢复及完整 bundle 指纹核验；任一失败即不得训练。训练中每模型另做 zero/final 保存和独立恢复；任何参数/Adam/RNG/history 不中性、source/code 改变、恢复差异或 CPU 触线都保留 partial 并终止，不续训或覆盖。
- CPU 总上限18,000秒，含零态预检、全部训练/评分、独立保存恢复及收尾。所有计划源、hard schedule、脚本和检查点 identity 在启动前冻结并写入 attempt/report。
- 三个 position seeds 必须全部通过 PLAN85 原六门；每个终点都按原/配对/改名/改名配对及 joint 视图判读。数值固定为：原题每seed≥173/192、原四类各≥40/48、pair≥87/96；改名每seed≥173/192、改名四类各≥40/48、改名pair≥87/96；原/改名joint各≥44/48。无补训、无择优checkpoint、无改门。

## 3. 全新 v7 面与一次性确认

- 训练结束、九个最终端点冻结之后，才生成 v7；40项，四类各10，sacrifice/confirm 各20，4 observations，干扰实体 Hamming d=1。builder 使用固定独立 seed 107；query 实体与 material-answer 词（`uses` 值）必须分别与 v3 train、v5、v6 的实体/材料词集合交集为0。固定方向词和`unknown`是任务标签，不作为新材料词计入隔离集合。面构造只在内存/新唯一 manifest 发生，不向模型查询。
- 在 sacrifice 子集上对9个新 hard-neighbor 终端跑范围检查，沿 PLAN-N1-93 冻门 `usable_endpoints≥6/9`。不达线则不查 confirm。
- 达线后，在同一 v7 confirm 子集一次性同时评估9个新终端和旧九端点（run88c position + run89a terminal + run89b cut），共360行。旧家族仅作同面描述性基线；原 v6/PLAN94 四门只应用于9个新 hard-neighbor 终端，绝对值和分母一个不改。CPU范围检验≤180秒、确认查询≤600秒。
- 新家族沿用 PLAN-N1-94 原四门：每个position seed `free_exact≥10/20`；每个(seed,kind) `free_exact≥2/5`；三seed合计position相对terminal优势≥12/60；相对cut优势≥12/60。不得据range或控制臂表现改写四条线。
- v7 attempt 写入即消费；中断、判红或边界故障均不重试、不重抽、不改 seed、不改门。原 C3/C4/保持集/三器官证据边界保持；`TRAIN_SHA=None`、`can_promote=false`。

## 4. 执行顺序

〔2026-10-17 实施与zero前置〕hard schedule、九模型控制器、v7 builder、sacrifice范围与confirm执行器已实现；四组 PLAN-N1-100 合同测试15项通过，Ruff、Black、py_compile通过。训练输入 schedule SHA-256 `dc3a96ad17d43bd898e403ecd119c12c781317a71ab06ba39cdab6db24c40d50`，3072项、每parent16槽、四类各768；lookup/unknown四格各192、revision两距离格各384、transition保留两事实改名输入；revision目标对行起点汇总为240/288/240。train提取器对非train对象仅读`id`/`split`以跳过，其余字段不解码；合同测试用不可解析的dev/holdout后缀验证该隔离。九枚run83 full-fresh zero的保存/独立恢复预检已完整通过：9/9模型加载、9/9零checkpoint重读恢复相等、86项来源守卫未变、CPU50.390625/300秒，training updates=0，v3 dev/holdout与确认查询均为0。预检件为 [checkpoint preflight](../../reports/taiji_n1_hard_neighbor_training_20261017_checkpoint_preflight.json)，SHA-256 `5dec355d765d1b5b3de98c680f8392990d6363dde1e035c56fb3db71db27d492`。未生成v7面或查询模型。实现审计纠正了初稿的两个路径误差：九枚full-fresh zero来自run83 preflight，run88c只有position三枚；材料答案隔离按`uses`词池定义，方向和弃权标签保留为任务常量。

唯一下一步：九枚zero checkpoint前置已通过，现按固定配方开始九模型训练；全程总CPU上限18,000秒，已耗50.390625秒。三个position seeds全部过原六训练门并冻结后才生成v7；再先执行sacrifice范围门，达≥6/9才运行一次confirm。任一训练工程守卫失败，保留partial并停止，不覆盖、不自动重跑、不查询其他面。本计划不推送远端、不删除数据。
