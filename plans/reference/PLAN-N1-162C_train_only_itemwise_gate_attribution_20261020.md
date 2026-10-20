# PLAN-N1-162C：问句增强训练门逐题配对归因

**状态：已依用户授予的执行计划批准权批准并冻结**  
**日期：2026-10-20**  
**执行结案：** [PLAN-N1-162B](PLAN-N1-162B_execution_closeout_20261020.md)  
**冻结训练方案：** [PLAN-N1-162](PLAN-N1-162_question_surface_matched_continuation_20261020.md)  
**执行修订：** [PLAN-N1-162B](PLAN-N1-162B_首步解析器修正与隔离续执行_20261020.md)

## 动机与问题

N1-162B六个endpoint完成55,296次更新与7,488条train-only评分，但原N1-132门仅2/6通过；seed11和seed12的surface_augmented在transition/rename-factorial读数明显下降，seed12 control另有unknown类低于40/48。seed13 surface_augmented通过，但不足以支撑三seed稳定结论。按N1-162原规则未生成seed132新面。

对封存的5,760条原训练门行和1,728条长度行做一次离线逐题配对归因：确认哪些seed、evaluation set、类别、factorial条件和父任务形成candidate/control差异；汇总unknown长度差；将异常父任务映射到schedule中的增强模板曝光。分析只解释训练集充分性门的差异，不代表新能力泛化或表面模板的单独因果效应。

## 冻结输入与字段恢复

- 结果：`reports/taiji_n1_plan162b_question_surface_result_20261020.json`，SHA-256 `19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31`。
- progress：`reports/taiji_n1_plan162b_question_surface_progress_20261020.jsonl`，SHA-256 `48212bc24ff9267390f7568142a0afd99dfeb39ab787e36c420d6d5d28ed4614`。
- schedule：`plans/manifests/n1_plan162_v34_question_surface_schedule_20261020.json`；与结果中730项冻结source/checkpoint哈希并集一并核验。
- progress每个`original_train_gate_row`的上游`arm=position`覆盖了外层实验臂字段。修复分析器按冻结执行顺序将连续960条gate row依次绑定到六个`seed/arm` endpoint，并逐chunk核对seed、evaluation_set分母、ID唯一性及结果模型门汇总。每个endpoint的288条length row单独校验seed/arm和键唯一性；不依靠被覆盖的gate arm字段推断实验臂。
- 为每个transition/unknown错分父任务关联schedule里的kind、长度和`augmented_template_id`频次；只做分层共现描述，不声称单个模板导致某条错误。

## 分析和停止合同

- 必须复核全部62,784条progress行中的事件分母：55,296 training updates、5,760 original gate rows、1,728 length rows；无坏行，progress/result/source哈希全程不变。
- gate pair按`seed/evaluation_set/kind/id/condition/case_id`匹配surface_control与surface_augmented，列出exact四格转移、输出字节变化和transition错误父项；按同键分层，不重算或改变N1-132门。
- length row按`seed/context_length/kind/id/split`配对，给出每类及unknown 8/16/32行exact数和control→augmented转移。
- 将N1-132充分性规则显式复算为诊断字段：原题/rename各free exact≥173/192、每类≥40/48、pair≥87/96，且原始/rename factorial joint各≥44/48；逐项复算必须与封存六个布尔门完全一致，否则停止并只报告不一致。
- CPU≤120秒；0模型/checkpoint加载、0 inference/query、0参数更新、0 dev/holdout/confirmation访问；不造新面，不晋级。
- 任一来源SHA、进度分母、配对键或复算门不一致，写停止报告并保留原始文件，不重试、不修改训练历史。

## 产物

- 分析器：`scripts/training/analyze_taiji_n1_plan162b_train_only_itemwise_attribution.py`
- 契约测试：`tests/taiji_native/test_n1_plan162b_train_only_itemwise_attribution_contract.py`
- 报告：`reports/taiji_n1_plan162b_train_only_itemwise_attribution_20261020.json`

归因报告完成后，结合该证据重新选择唯一N1队首；不得自动补训、修改训练门或消费seed132面。
