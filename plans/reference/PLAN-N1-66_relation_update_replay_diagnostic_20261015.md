# PLAN-N1-66 · seed13原配方精确重放与梯度/Adam位移诊断

## 唯一问题与固定来源

65证明seed13 position在最后约400步teacher/free共同退化，但原日志无梯度与参数位移，尚不能区分异常梯度、Adam尺度和模块漂移。仅为定位，对63同一seed13 position零档作一次完整精确重放；不产生新的能力小试，不选择中点或追加63。

来源63报告SHA `d982909f6722d37040fb21f7d869aed6817f3a42a3c4b08300b6746ef14e2922`，65报告SHA `6e36f68a9327140586f5a61d23be9f9fa502f9240316dd664e3c9158a7439ef8`。使用63已验证的s13_position_zero.pt，最终与s13_position_final.pt的完整参数/Adam/RNG/history及signature比较。原档与原报告全程字节守卫。

## 固定执行与预算

新独立本机CPU≤300秒，线程1；包含前置、诊断自身zero/final保存及两个独立新进程恢复、所有评分和失败partial保存。先完成自身zero独立恢复，再做固定6144更新；原Adam完整续接、lr0.001、43顺序、40双问题、QA均值+0.25保真、所有梯度与更新规则一字不改，不加入clip/decay/冻结。

沿63的0/3072/6144全四视图，共1728组已知train/probe teacher+free评分；所有评分逐项复现63对应seed13 position读数。每步loss与63固定同一步比较，容许浮点绝对误差≤1e-7且相对≤1e-6；不一致立即标记“未精确复现”，保存partial并停止，不拿漂移重放推断原运行。最终完整bundle指纹与63终点相同才认重放有效。

新增记录只取detach后的张量，不消耗RNG、不改前向/反向：每步reader/local/context/relation四组及writer子组的梯度L2/最大绝对值、更新前参数L2、实际Adam参数位移L2、位移/参数比；记录最大梯度与最大实际位移的参数名。保存全部6144步，不只摘峰值。重点比较首次loss抬升前后的更新，不把峰值后果冒充原因。

checkpoint位于`data/checkpoints/n1_relation_update_replay_20261015/`；脚本`scripts/training/audit_taiji_n1_relation_update_replay.py`，合同`tests/taiji_native/test_native_relation_update_replay_contract.py`，报告`reports/taiji_n1_relation_update_replay_20261015.json`。exclusive不覆盖63；预算/工程/复现失败即停，不自动重启。

## 判读与边界

只有复现与恢复/守卫均过才使用仪器读数；若无清晰定位，继续冻结训练面机制诊断，不盲改配方。0dev/holdout/v4，确认面仍零查询，旧四门不变。此重放6144更新只用于原配方可复现性与更新链取证，不算新seed、一般化或净收益；不支持默认推广。C4、旧能力保持、同bundle三器官、N系列其他线与历史CI债均保留。
