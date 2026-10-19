# PLAN-N1-140 执行结案：提高两行基线暴露的匹配续训

**状态：完整执行，原训练门 5/6 通过，`can_promote=false`。**获批配方、CPU上限及N1-132判据保持不变；没有追加训练，也没有访问dev、holdout或confirmation。

## 执行范围与完整性

- 六个endpoint各完成9,216次续训，共55,296次参数更新；全部六枚final bundle均独立恢复成功，恢复指纹相同，history为61,440。
- 完成7,488条train-only计分：每endpoint 960条冻结原门输入及288条增强上下文输入。全部逐题结果保存在[N1-140完整结果](../../reports/taiji_n1_plan140_baseline_retention_balanced_matched_continuation_result_20261019.json)。
- Runner记录累计CPU 2,553.406/7,200秒，其中readiness 3.500秒、zero bundle预检50.641秒、训练与计分2,499.266秒。此前单独静态构造/契约核验另耗3.875秒，不计入runner累计。
- readiness和训练期间来源哈希均未变化；attempt、122 MB本地progress与12枚zero/final checkpoint哈希已与最终报告逐项复核。zero和final bundle各6/6通过独立恢复。
- readiness/preflight/训练均未查询dev、holdout或confirmation；训练和计分完成后仍不晋升模型。

## 冻结门结果

| seed | 分支 | 原始自由回答 | 重命名自由回答 | transition原题 | 增强上下文 |
|---|---|---:|---:|---:|---:|
| 11 | candidate | 192/192 | 191/192 | 48/48 | 279/288 |
| 11 | control | 192/192 | 192/192 | 48/48 | 156/288 |
| 12 | candidate | 192/192 | 192/192 | 48/48 | 287/288 |
| 12 | control | 192/192 | 192/192 | 48/48 | 210/288 |
| 13 | candidate | 163/192 | 158/192 | 19/48 | 96/288 |
| 13 | control | 192/192 | 192/192 | 48/48 | 45/288 |

seed13 candidate 是唯一失门endpoint：lookup、revision、unknown仍各48/48，原题transition仅19/48，重命名transition为14/48。五个其余endpoint通过N1-132原门。candidate增强上下文合计662/864，高于control的411/864；这项train-only改善没有抵消原门失败，也不代表未见泛化。

| 增强条件 | candidate | control |
|---|---:|---:|
| 8行 / d=1 | 111/144 | 58/144 |
| 8行 / d=2 | 111/144 | 79/144 |
| 8行 / d=4 | 111/144 | 93/144 |
| 16行 / d=1 | 107/144 | 46/144 |
| 16行 / d=2 | 111/144 | 61/144 |
| 16行 / d=4 | 111/144 | 74/144 |

## 逐题归因与限制

[离线逐题报告](../../reports/taiji_n1_plan140_itemwise_retention_context_attribution_20261019.json)只读取N1-132、N1-139及N1-140的已封存结果，没有加载模型/checkpoint、更新参数或发出模型查询。seed13 candidate的原题transition由N1-139的28/48降到N1-140的19/48；两批有17个共同错误，N1-140新增12个错误、修复3个旧错误。总题目净下降9项。

N1-140的静态slot映射把每个父项/route的24个transition位置排成“前12个两行baseline，随后六类增强各2个”。这使baseline与增强条件在局部更新顺序上成块。该次序可能造成近期增强梯度对两行transition保持的干扰，但仅凭本批结果不能认定因果；增强上下文整体分数高于控制，且失门集中于seed13 candidate。

## 唯一建议的下一步

为隔离slot时间顺序，草拟[PLAN-N1-141交错slot匹配续训](PLAN-N1-141_interleaved_context_schedule_matched_continuation_draft_20261019.md)：从相同N1-132父checkpoint重新建立候选/控制配对，只把N1-140候选的slot顺序改成交错；每父项24/48 baseline、六种增强各4/48，更新数、计分、冻结门和无留出边界全部不变。该新训练配方尚待单独批准，批准前不做checkpoint预检或训练。
