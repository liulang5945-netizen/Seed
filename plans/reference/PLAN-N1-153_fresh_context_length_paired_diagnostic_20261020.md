# PLAN-N1-153：fresh内容的上下文长度配对诊断

**状态：已依据用户授予的执行计划批准权批准并冻结。**N1-150训练集逐题归因显示：长度混合candidate在32行已知类命中为215/216，8行control为190/216；unknown正确弃答在8行分别42/72与72/72。本计划用一个全新且来源排除的合成面，检查该train-only差异是否仍出现在新实体名和材料值上。只作描述性诊断，不晋级模型。

## 冻结测面

- 一次生成seed127面：40个lookup、40个revision、40个transition、40个unknown，共160个base任务；每个任务展开成8/16/32行三个嵌套上下文，target事实保持最前，长度之间只在尾部增加干扰事实。
- 六个冻结endpoint：N1-150 final `length_mixed`与`eight_row_control`，seed11/12/13。每endpoint 160×3=480条查询，总计2,880条。
- 新目标实体、近邻实体和材料值均须与全部已消费数据、N1-150 train来源及N1-150生成的94,848个实体/94,848个材料值不重叠。N1-150训练上下文通过冻结builder与schedule从原v3 train源确定性重建并加入排除集合；每个测试实体及其近邻需按face builder合同验证。
- 每题8/16/32视图必须逐项核对目标事实、答案、实体、材料和嵌套前缀；候选/control使用完全相同face、ID、顺序和context length。

## 资源、前置与边界

- 训练/更新为0；六个final full bundle在测量前独立完整恢复6/6，测量前后模型/Adam/RNG/history指纹不变。
- CPU≤3,600秒，包含来源排除重建、face/readiness、六个独立恢复、2,880条评分、逐题报告和终端SHA复核。
- 0 dev/holdout/confirmation查询；本诊断自身不设能力晋级门，不更改N1-132训练门、N1-133历史门或N1-150结论，`can_promote=false`。
- 任一source overlap、face配平、嵌套、保存恢复、计数、CPU或SHA守卫失败即停止；不重抽、不重试、不扩展样本。

## 实施与产物

1. 实现来源/face builder、独立final restore scorer及合同测试；Ruff、py_compile和测试通过后提交代码。
2. 生成一次性face和source audit，核验全量排除并提交；随后运行零查询readiness并提交。
3. readiness通过后先独立恢复六个final bundle，再运行固定2,880条诊断并保存全量逐题输出。

- Face builder：`scripts/training/build_taiji_n1_plan153_v28_fresh_context_length_face.py`
- Scorer：`scripts/training/run_taiji_n1_plan153_v28_fresh_context_length_confirmation.py`
- 合同测试：`tests/taiji_native/test_n1_plan153_v28_fresh_context_length_confirmation_contract.py`
- Face manifest：`plans/manifests/n1_plan153_v28_context_length_seed127_20261020.json`
- 完整逐题结果：`reports/taiji_n1_plan153_v28_fresh_context_length_confirmation_result_20261020.json`
