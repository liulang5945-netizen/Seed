# PLAN-N1-68 · 同seed13分组实际位移约束固定校准

## 固定来源与唯一变量

67工程完整，报告SHA `88313caa6a27dc9a7dfcb2340b15f1e009526fd0afa82db1e60a9bf6362d07ff`；源回到63的s13_position_zero.pt，由63报告SHA `d982909f6722d37040fb21f7d869aed6817f3a42a3c4b08300b6746ef14e2922`绑定。保留其全部62056参数/35旧Adam状态/RNG/history，不从67合成两步档或63失败终点续接。

唯一变量是67固定五组实际Adam位移规则：relative0.001、floor1.0，组reader排除writer；moment/lr0.001、结构、完整观察信用、43原流、40双问题、QA均值+0.25保真均不变。不裁梯度、不冻结参数、不搜索阈值、不选择中点、不换名字或删样本。

## 固定执行

独立CPU≤300秒，线程1，固定6144正式更新/9216正式QA序列；0/3072/6144各完整四视图，共1728组已知训练/probe teacher+free。zero评分须逐项复现63该零档全部576行。每步记录原loss和五组提议/实际位移、边界、是否缩放；保留所有步以检验后程扰动是否缓解。不中点选择或循环加训。

先保存自身zero并独立新进程恢复完整规则/状态与下一步行为，再正式更新；终点保存自身final并独立恢复后才认完整。沿67下一步签名机制，主进程与worker各档临时副本一步，共额外4次合成签名更新及8条合成QA序列，独立列账，不混入6144正式更新/9216序列；全部包括在同300秒CPU中。

最终按原四门：原题≥173/192且四类各≥40/48；pair≥87/96；已知改名≥173/192且各类≥40/48；改名pair≥87/96。只判固定终点；任一不足即失败并停，不追加或改变门。

来源63/65/66/67报告、零档/旧档、课程manifest及v4封存字节、源实现和新更新器的身份前后守卫；0v3dev/holdout、0v4查询。所有41项Adam moment与逐参数旧步数+6144一致，完整history27648→33792。更新规则marker缺失/错误拒绝恢复，不能恢复成普通Adam。

## 产物与停止

实现`scripts/training/run_taiji_n1_group_relative_adam.py`，合同`tests/taiji_native/test_native_group_relative_adam_training_contract.py`，exclusive报告`reports/taiji_n1_group_relative_adam_train_20261015.json`，checkpoint目录`data/checkpoints/n1_group_relative_adam_train_20261015/`。预算/工程失败保存partial并停止，不自动重启、不覆盖任何旧档。67工程中含合成session的checkpoint不作为起点。

## 判读与完整目标

失败seed13的定向已见校准只能检验候选优化器，不能当作三seed稳健性、切断信用优势或未见确认。通过后仍须另冻三seed三臂正式配对和新确认准入，v4在此保持未消费；未通过则按取证修订，不追加本配方。

持续保留C4净收益、旧能力保持、同bundle三器官、端到端独立seed和N系列其他线及CI历史债。没有开放域/AGI、默认推广或项目收官主张，不推送或删除数据。

## 执行器提交前验证

原63 seed13零档SHA `0f1cbc2d606d5a26a155216dec17061972ce18ad87d5c80768264a857ef78293`直接绑定，协议将52 checkpoint保留为祖先字段而非直接起点。零档完整bundle/下一步规则恢复通过；67工程协议、partial阶段和缺失/错误marker均拒绝。一项专项合同与两套仓库Ruff通过，提交后固定执行，不追加。

## 完整裁决

6144更新/9216 QA、1728项评分完整，CPU220.828125秒，两独立恢复与全部守卫通过。终点原192/192、pair96/96、改名190/192、改名pair96/96，四门通过；不作未见或三seed结论。见[68裁决](EXP-N1-68_group_relative_adam_adjudication_20261015.md)。唯一下一步[69配对更新器工程](PLAN-N1-69_group_relative_paired_engineering_20261015.md)，保持v4零查询。
