# N1-69-CI · 干净检出的机械合同与真实档集成证据

## 已复现问题

根CI的`pytest tests/taiji_native`及全套会收集新增N1合同。部分机械合同直接调用source_payload/readiness或读取本机checkpoint；`.gitignore`忽略全部`.pt`，`git ls-files --error-unmatch data/checkpoints/n1_contextual_question_train_20261015/final.pt`失败。只读屏蔽本机`.pt`读取后，source_payload可复现FileNotFoundError；没有移动或删除真实数据。不能用本机10 passed声称干净检出可运行。

## 唯一修复

为机械单元合同提供显式、固定种子的真实模型合成状态：参数使用正常原生类，旧Adam moment与step用明确测试值，合成Observation及zero-probe，会话/RNG完整。它不假装是训练结果，也不作为实验起点。用该fixture继续验证原来全部机械不变量：零残差、信用、分组边界、moment保持、同臂逐位中性、规则marker和恢复。

范围为新增source-bound合同：contextual_question、local_relation、relation_confirmation、relation_paired_training、relation_update_replay、group_relative_adam、group_relative_adam_training、group_relative_paired。仅更改测试及专用fixture；不修改冻结训练器、模型、报告、checkpoint或实验准入/门。原冻结档真实性的整链验证保留为独立local-only集成合同，遵守根tests/conftest.py已有LOCAL_ONLY_ARTIFACTS政策；缺资产时明确skip，不把skip叫通过。artifact-free核心机械合同始终执行，不能整体skip来隐藏问题。

新fixture放`tests/taiji_native/_native_relation_fixtures.py`，真实档集成合同放`tests/taiji_native/test_native_local_checkpoint_integration.py`。若现有contextual迁移需要更早父类，提供相应明确测试状态，不绕过严格格式/缺键拒绝。v4无模型查询；现有纯数据/传输seal合同继续运行。

## 验证与后续

针对相关合同，先正常合跑，再以只读拦截data/checkpoints读取的方式模拟无本机资产并运行artifact-free合同；临时pytest checkpoint不受拦截。检查核心测试实际执行/通过与集成跳过数量，不能只看退出码。两套仓库Ruff、改动文件Black及项目导航合同必跑；历史全Black/CI债独立保留，不声称全CI绿。

完成后提交修复与证据，唯一下一步70已冻结正式配对训练。69工程完成状态保持，不重跑任何研究批；全项目出口与新确认面边界不变。

## 完成证据

正常31 passed；屏蔽data/checkpoints访问并模拟资产缺失：29 passed、2 skipped，ignored_checkpoint_read_attempts=0。独立真实档集成在本机实际执行通过。第一次合成fixture漏zero-probe由恢复合同直接检出，补齐测试会话后上述结果通过；冻结源代码未改。Ruff/Black/导航检查见本批提交。
