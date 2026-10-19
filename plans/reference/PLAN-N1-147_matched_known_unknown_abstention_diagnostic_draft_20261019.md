# PLAN-N1-147：匹配已知/未知事实的长上下文弃答诊断

**状态：依据用户授予的执行计划批准权已批准并冻结。**此项是新的只读行为测面，单独生成新题并一次查询；它不是确认面、不改N1-144判据，也不支持能力晋升。

## 依据与目标

N1-146在N1-144已消费v23面上发现，N1-141 candidate的58/60条错误lookup输出均为字面`unknown`，没有输出题内其他实体的材料值；revision为60/60正确。下一步以严格配对题面检查，仅把目标材料事实存在/缺席作为区分因素时，冻结endpoint是否能对已知题回答、对未知题弃答。

## 新测面与配对合同

- 固定seed124，40对共80题：每对使用同一目标实体、同一问题、同一组观察和位置；known条件在目标位置含`target uses answer`，unknown条件仅把该一行替换为一个与目标Hamming距离d=1的干扰实体材料事实。其余31行逐字相同。
- 每题32行观察，实体长度10；每对仅目标事实/替代干扰事实不同。known目标行起始位置轮转`[0,8,16,24,31]`，每位置8对；6/7字符答案各20对，按目标位置平衡。unknown不出现目标实体事实，答案为`unknown`。
- 全部实体名和材料值须排除N1-144已有来源清单以及整张已消费v23面；生成前重算并绑定并集、来源SHA与零交集。新题seed124一次构造，输出write-once，失败不重抽。
- 新题是diagnostic split，不混入训练、dev或确认集合。v23 face/结果不作为题目内容重用，仅用于来源排除和动机记录。

## endpoint与固定读数

- 使用N1-144相同12枚冻结endpoint：N1-132 position parent三seed、N1-140 block candidate三seed、N1-141 candidate/control各三seed。先独立恢复完整bundle并核对checkpoint/Adam/RNG/history；任一失败即0查询停止。
- readiness通过后一次查询每endpoint全部80题，共960条。每条结果先落progress；任何中断即封存，不重跑、不补抽。
- 按endpoint和seed报告known精确答案率、known输出`unknown`率、unknown输出`unknown`率、unknown误答率，并按40对列出known/unknown的逐对输出组合。复核position与长度配平、输出UTF-8和来源哈希。
- 不设置通过/失败阈值，不据此调整模型或N1-144门；只作为下一版训练目标设计的诊断读数。`can_promote=false`。

## 预算与边界

- 总本机CPU≤3,600秒，含source union、题面构造、12端点readiness、960条查询和末端SHA复核。
- 0训练、0参数/optimizer更新、0 dev/旧v3 holdout查询；不自动晋升。
- 题面与查询runner须先实现、合同测试并提交，之后才生成题面。保存/恢复前置必须通过，再启动一次查询。

## 产物路径

- 题面构造器：`scripts/training/build_taiji_n1_v24_abstention_after_n146.py`
- 查询与readiness：`scripts/training/eval_taiji_n1_v24_abstention_after_n146.py`
- 合同测试：`tests/taiji_native/test_n1_plan147_v24_abstention_contract.py`
- write-once题面：`plans/manifests/n1_v24_abstention_seed124_20261019.json`
- write-once逐题结果：`reports/taiji_n1_v24_abstention_diagnostic_20261019.json`
