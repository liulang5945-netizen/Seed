# SPEC-N1-13 · 原生候选独立恢复工程审计（2026-10-15）

承接SPEC12，属于可逆工程诊断；一次独立CPU总上限60秒，父/两个独立worker均Torch单线程。恢复夹具使用新工程字节，不读取冻结16项、旧能力面或母档，不开展能力训练、不测净增益。

候选保存两个会话（来源名可相同）、其中一个会话两个外部观察；零更新bundle以独占创建写入 `.seed_test_tmp/n1_native_evidence_restore_20261015/zero.pt`。新进程weights-only恢复后，核对全部参数、两个会话证据来源/时序、每会话teacher logits及RNG；通过后才运行父进程1步恢复夹具，保存优化器和bundle。再由父/恢复worker各执行同一次续步，比较所有会话状态/输出、参数和RNG。总父2步、子1步，不报告loss收益，不计作训练能力。

新进程worker只加载候选模块与自足bundle，无SeedRuntime、Legacy或外部模型。代码/报告拒覆写；失败保留报告并停后继更新。报告路径 `reports/taiji_n1_native_evidence_restore_20261015.json`，结果不存在前本件不是通过证据。必须记录候选、复用算子及审计代码hash，独立CPU开销和全部等价守卫。暂不宣称NativeCheckpoint.components挂载、默认切换或完整持续认知。

本门通过后的唯一下一步：冻结未见组合、事实替换与不能直接复制的迁移任务课程、同信息三臂（逐位置/每来源末态/信用截断）的配对协议，并先做数据隔离与单episode成本检查；正式训练仍独立准入，不改旧门。
