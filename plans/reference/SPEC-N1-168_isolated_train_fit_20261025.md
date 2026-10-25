# SPEC-N1-168：固定8轮隔离小集拟合预注册

冻结于任何真实更新前。只使用[PLAN-N1-168](PLAN-N1-168_isolated_train_fit_20261025.md)指定parent与N1-167的4个既有train样本。

1. 原学习方法`_target_pass(learn=True)`、配置、response-plan encoder/目标和所有允许更新组不变；显式维护原trainer的episode/global_step/history，等价于原train循环的子序列调度。禁止把子集伪装成新的corpus digest或重建目标。
2. 额外32 episode、9320目标字节更新；8轮固定终点。0/1/2/4/8轮共20项评估，每项原始/掩码TF、完整汉字TF和128字节自由生成；不按表现提前选checkpoint或扩轮数。
3. zero与最终/partial bundle均必须新进程恢复，完整trainer checkpoint digest、模型digest、计数及固定首步分布签名一致；先zero通过才训练。保存路径与parent分离，parent文件SHA不变。
4. TF与自由匹配≥0.80只支持4题已见可学性；TF≥0.80且自由匹配<0.20提示暴露/滚动问题；其他情况连续报告、不改门。不同题要分别报告，不用均值遮蔽失败。
5. 参数组摘要记录哪些已登记owner改变，状态变化不误称参数学习。主结论必须同时看未注入答案的自由生成；训练期target信息只作为合法监督，不在评估时注入oracle。
6. 本包≤1600 CPU秒/2200 wall秒（含新进程恢复）；它与N1-167合计仍受本轮1800/2400总额。触顶保存partial、实际更新和缺项后结束，无额外预算/旧消耗重置；0 dev/holdout/confirmation访问，`can_promote=false`。
