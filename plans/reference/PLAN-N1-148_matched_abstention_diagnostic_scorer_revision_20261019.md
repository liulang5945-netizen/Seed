# PLAN-N1-148：诊断split专用scorer的匹配弃答复测

**状态：依据用户授予的执行计划批准权已批准并冻结。**N1-147题面已因runner attempt消费，故本计划生成全新seed125题面；v24仅加入source exclusion，不重用题目或输出。

## 依据与目标

N1-147在N1-144通用确认scorer的`holdout` split守卫处停止，首个endpoint完成0行；没有任何行为读数。根因是测量入口与diagnostic数据split不匹配。N1-148保持40对known/unknown匹配设计及960条总查询，仅改用明确接受`diagnostic` split的独立scorer，确保split校验先于model forward和query计数。

## 题面与来源排除

- 新seed125，40对、80题，known/unknown问题共用相同target、question和32行上下文；仅目标槽位从target材料事实替换为近邻实体事实。目标起始位置`[0,8,16,24,31]`各8对，答案6/7字符各20对。
- source exclusion继承N1-147的36,802个实体名/13,968个词值，并加入整张已消费v24 seed124面新增的1,320名/1,320值；预期并集38,122名/15,288值，生成前核验与零交集并绑定来源SHA。
- seed125一次生成，write-once；不复用v24任何题目。新split固定为`diagnostic`，runner不得改标为holdout。

## scorer与查询

- 在`torch.no_grad()`下先验证全部80题均为diagnostic split、frozen position arm、唯一ID及40组完整配对，再开始forward。每项使用paired projection/observation/evidence、teacher logits和trace generation；不调用optimizer step/backward。
- 每endpoint query前后核对完整bundle fingerprint；session须新建且清理；progress逐行落盘。report保留known exact/abstain、unknown abstain/false-answer及同对组合。
- 12端点各80题，共960条诊断查询；没有通过线，不作显著性或因果主张。0训练/参数更新、0 dev/旧v3 holdout，`can_promote=false`。
- 任何preflight/scorer错误或中断均停止，不重试、不补抽；v25 face一旦query attempt启动即消费。

## 预算与产物

- 本机CPU≤3,600秒，含source union、面构造、12端点独立bundle恢复、查询和SHA复核。
- 先实现、测试并提交builder/runner，再生成seed125面；face合同通过后先跑零查询readiness并提交回执，再单次执行960条查询。
- builder：`scripts/training/build_taiji_n1_v25_abstention_after_n147.py`
- diagnostic runner/scorer：`scripts/training/eval_taiji_n1_v25_abstention_after_n147.py`
- 合同测试：`tests/taiji_native/test_n1_plan148_v25_abstention_scorer_contract.py`
- write-once face：`plans/manifests/n1_v25_abstention_seed125_20261019.json`
- write-once result：`reports/taiji_n1_v25_abstention_diagnostic_20261019.json`
