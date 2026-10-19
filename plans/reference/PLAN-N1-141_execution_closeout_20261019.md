# PLAN-N1-141 执行结案：交错candidate上下文slot

**状态：完整执行，N1-132原训练门6/6通过；`can_promote=false`。**本批只训练和计分冻结v3 train输入；没有访问dev、holdout或confirmation。

## 执行与完整性

- 六个endpoint各完成9,216次续训，共55,296次更新；候选每父项24/48为两行baseline、六类增强各4/48，candidate slot按冻结轮转序列交错，control维持两行输入。
- 完成7,488条train-only计分：每endpoint 960条冻结原门及288条增强上下文。六枚zero和六枚final bundle均独立恢复成功；final history均61,440。
- runner累计CPU 2,581.750/7,200秒：readiness 4.359秒、zero bundle预检58.578秒、训练与计分2,518.813秒。此前单独静态构造核验另耗4.500秒，不计入runner累计。
- attempt、122 MB本地progress、全部来源哈希及12枚zero/final checkpoint哈希已独立复核；报告完整、无运行错误。
- 训练门全部通过，但这只是冻结train门；不晋升、不形成未见泛化结论。

## 冻结训练门

| seed | 分支 | 原始自由回答 | 重命名自由回答 | transition原题 | 增强上下文 |
|---|---|---:|---:|---:|---:|
| 11 | candidate | 192/192 | 192/192 | 48/48 | 260/288 |
| 11 | control | 192/192 | 192/192 | 48/48 | 156/288 |
| 12 | candidate | 192/192 | 192/192 | 48/48 | 286/288 |
| 12 | control | 192/192 | 192/192 | 48/48 | 210/288 |
| 13 | candidate | 192/192 | 192/192 | 48/48 | 288/288 |
| 13 | control | 192/192 | 192/192 | 48/48 | 45/288 |

candidate增强上下文合计834/864，control为411/864。按条件汇总：

| 条件 | candidate | control |
|---|---:|---:|
| 8行 / d=1 | 135/144 | 58/144 |
| 8行 / d=2 | 144/144 | 79/144 |
| 8行 / d=4 | 144/144 | 93/144 |
| 16行 / d=1 | 123/144 | 46/144 |
| 16行 / d=2 | 144/144 | 61/144 |
| 16行 / d=4 | 144/144 | 74/144 |

## N1-140配对归因

[离线逐题归因](../../reports/taiji_n1_plan141_itemwise_retention_context_attribution_20261019.json)只读取N1-132、N1-140和N1-141已封存结果；CPU 0.094秒，0模型/checkpoint加载、0模型查询、0参数更新。seed13 candidate transition从N1-140的19/48升至48/48：原29个错误全部修复，未新增错误。candidate增强上下文从662/864升至834/864（+172），control仍为411/864。seed11/12候选原门保持全过。

这支持“交错slot顺序有助于本批seed13候选保持原始transition”的训练集结论；它不是新题泛化或能力晋升证据。完整逐题输出保存在[N1-141结果报告](../../reports/taiji_n1_plan141_interleaved_context_schedule_matched_continuation_result_20261019.json)。

## 唯一下一步

在用户授予的执行计划批准权下，已批准并冻结[N1-142 v21新题确认计划](PLAN-N1-142_v21_fresh_confirmation_after_n141_draft_20261019.md)。该计划为N1-141候选、N1-140块状候选、N1-141两行控制与N1-132父端点构造一次新v21面；只有牺牲子集达到冻结范围门才查询确认子集。测面与能力门均为新计划范围，不回看已消费的v15或更早题面。
