# PLAN-N1-146：v23输出来源逐题离线归因

**状态：依据用户授予的执行计划批准权已批准并冻结。**只解析已消费v23确认面保存的输出字节，定位错误字符串是否对应本题某个实体的材料事实；不做模型推理，不消费任何新测面。

## 依据与目标

N1-145显示N1-141 candidate在32行面上lookup仅2/60，而revision为60/60、transition为45/60；lookup各目标起始位置均低，未呈现稳定位置递减。相对其他endpoint，candidate在transition的较多命中被lookup较少命中抵消。下一步需要辨别lookup错误更像是取到了题内其他实体的材料、revision旧值、无关词，还是非词汇/截断输出。

此项属于同一已消费确认面的post-hoc输出来源分析。所有读数均为描述性；不作因果或未见能力主张，不据此回头修改模型并重用v23确认。

## 冻结输入

- N1-144计划SHA-256：`b5527b5f8a56aa12604a7dbc40dad1fa74c3f4ea8e953a6b2539f1fa3c5fcf42`。
- v23题面SHA-256：`e0ddf062e3c5f081dd010db3a8951d9aa426d9717de39872527a4eeabb76f6ae`。
- N1-144完整结果SHA-256：`67038a9848b3c7ac088fea1983d9ba40ea153e91c8a4c436db830e8894e80388`。
- N1-145逐题归因SHA-256：`07e4a18ffae7b2deb94f1bf6e49985839658f77188ecf940eae80e98da3883f1`。
- 仅使用完整结果中960条confirm行及面内观察/答案。先校验结果/面SHA、960行、12端点每端点80题、题目ID对齐、`confirmation_failed`、来源哈希未变及已有`free_exact`汇总一致；任一不符即停止且不写报告。

## 冻结分析

1. 只对`output_hex`做严格UTF-8解码并记录解码失败；不重新评分，不以解码文本改写`free_exact`。
2. 对lookup/revision错答，将解码字符串与该题全部`uses`材料值精确比对，分类为正确答案、同一目标实体的旧值、其他实体材料值（列出实体及与目标的Hamming距离）、多重匹配歧义或未匹配材料值。revision中目标实体旧值单独标为陈旧事实读取。
3. 对transition错误，分类为初始方向、其他合法方向或非方向输出；unknown行只报告是否输出`unknown`或其他内容。
4. 按12个endpoint、类别和N1-141 candidate seed汇总类别计数。所有分层同时输出分子/分母；保留每条输出的题目ID和分类，便于追溯。不得作显著性、因果或能力声明。
5. 复核所有输出字符串都来自原报告的`output_hex`，已记录的`free_exact`与原结果不变；报告绑定结果/面/计划SHA并声明`can_promote=false`。

## 资源、边界与产物

- CPU≤30秒；仅解析JSON、十六进制输出和题面，不加载模型/checkpoint/optimizer/RNG，不读训练、dev或其他留出原始题库。
- 0模型查询、0参数更新；不改变N1-144门、不生成新题、不重新确认、不自动晋升。
- 分析脚本：`scripts/training/analyze_taiji_n1_v23_output_sources_after_n145.py`。
- 契约测试：`tests/taiji_native/test_n1_plan146_v23_output_source_attribution_contract.py`。
- write-once报告：`reports/taiji_n1_v23_output_source_attribution_20261019.json`。
