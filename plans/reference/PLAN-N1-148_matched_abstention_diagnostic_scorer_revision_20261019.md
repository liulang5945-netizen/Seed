# PLAN-N1-148：诊断split专用scorer的匹配弃答复测

**状态：依据用户授予的执行计划批准权已批准并冻结。**N1-147题面已因query attempt消费，故本计划生成全新seed125题面；v24仅加入source exclusion，不重用题目或输出。

## 依据与目标

N1-147的12端点readiness已通过，但首个query attempt在N1-144确认scorer的`holdout` split守卫处停止，completed rows为0。根因是测量入口与diagnostic数据split不匹配。N1-148保持40对known/unknown题面及960条总查询，改用明确接受`diagnostic` split的只读scorer，确保split校验先于forward与query计数。

## 新面与source exclusion

- 固定seed125，40对、80题；目标实体、问题和31条共享观察完全相同，known条件在目标槽位含`target uses answer`，unknown条件将同一槽替换成d=1近邻实体材料事实。32行、位置`[0,8,16,24,31]`各8对，答案6/7字符各20对。
- 继承N1-147 source exclusion并加入整张已消费v24 face新增的1,320个实体名和1,320个材料值；预期并集38,122个实体名/15,288个词值。生成前核对N1-147停止回执、readiness、attempt/progress/result、来源SHA和零交集。
- seed125一次构造，write-once。失败不重抽；v24/v25诊断题均不得进入dev/正式确认集合。

## diagnostic scorer与查询

- 保持`split="diagnostic"`。开跑前验证80题、40对、condition配对、target slot仅一行差异、frozen position arm、ID唯一。
- 使用诊断专用路径：`torch.no_grad()`；验证diagnostic split和position arm后，再执行paired projection/observation/evidence、teacher logits和trace generation。每endpoint前后校验完整bundle fingerprint，清理diagnostic session，不调用backward/optimizer step。
- 12冻结endpoint各查询80题，共960条。query输出逐行落progress；错误/中断即停止、不重试。
- 报告known精确答案/弃答率、unknown字面弃答/误答率及逐对输出组合。无通过门，无显著性或能力主张，`can_promote=false`。

## 预算与产物

- CPU≤3,600秒，含来源并集、face生成、12端点独立恢复、960条查询和哈希复核。
- 先实现/测试/提交build与diagnostic scorer，再一次生成v25 face；face检查后先做零查询readiness并提交，最后单次query。
- Builder：`scripts/training/build_taiji_n1_v25_abstention_after_n147.py`
- Scorer/runner：`scripts/training/eval_taiji_n1_v25_abstention_after_n147.py`
- Contract tests：`tests/taiji_native/test_n1_plan148_v25_abstention_scorer_contract.py`
- Face：`plans/manifests/n1_v25_abstention_seed125_20261019.json`
- Result：`reports/taiji_n1_v25_abstention_diagnostic_20261019.json`
