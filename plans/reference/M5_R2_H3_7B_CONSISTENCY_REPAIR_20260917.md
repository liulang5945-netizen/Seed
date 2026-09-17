# R2-H3.7B 主线一致性修复与有界验证

2026-09-17：用户明确继续主线，VISION/H3.8不是依赖或待批准事项。此包修复现有原生局部学习，不替换核心，不启用自动微分训练，不推进Mini。

## 范围与证据隔离

1. 计划目标更新后同步刷新模型缓存prior，不推进phase。原始训练首byte的257项缓存概率与实际读出全部不一致，已用失败测试复现。
2. 当前槽/全局槽混合的反传按前向0.75/0.25分配；phase0只回传槽0。用四phase数值差分验证混合映射，不能由此宣称整个局部学习等于端到端梯度。
3. 新目标格式使用精确16-byte窗口，与运行phase对齐；窗口允许包含UTF-8字节片段，仅用于byte sketch，运行输出合法性机制不变。新格式用固定单位scale，诚实记录它是确定性监督坐标，不是学得语义。旧encoder及payload保持旧语义，不自动迁移。

修复prior和信用后，从历史checkpoint继续训练的轨迹将改变；旧训练证据必须绑定旧Git提交，不宣称修复代码能复现旧训练。旧checkpoint的参数、前向及只读输出不因本修复改写。结束标记仍由原有byte损失监督，不强行把它当作回答内容target；尾槽饱和仍是现有设计限制。

## 冻结验证包

目的：证明修复链路可训练、可保存恢复，而非再次挑战H3.7三seed晋级门。独立输出目录，禁止覆盖旧checkpoint；不追加旧run的epoch。

- fixture：既有v3，SHA256 `0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691`；仅使用train，final不评分，dev也不据此调参。
- seed20260917，CPU单线程，300000参数上限，48维/4槽/16byte，原有学习率不调。
- A：修正prior与混合信用＋旧字符安全target；B：相同修复＋新byte对齐target。相同初始化、同12train episodes、各1epoch；不是与旧历史结果的公平架构比较。
- 每臂正式训练前执行零步/child原子保存恢复、parent不变、factorized preflight及新进程恢复后一步续训一致性。preflight更新只在检查用child，单独记账。
- 每臂正式训练后保存checkpoint并新进程复核digest，再执行一步续训一致性检查（仅内存检查，不覆盖正式child）；不生成final输出。
- 每臂限时15分钟，磁盘空闲至少512MiB。新进程检查限时180秒。失败停止、保留失败状态，不自动加预算。

通过标准：无非有限值、每臂完成12episodes/537byte更新、参数预算通过、恢复及续训digest一致、正式checkpoint保存成功。训练surprise只诊断，不是内容能力；不得据此晋级L2或宣布修复已带来能力收益。

## 唯一后续

工程验证通过后，先检查train可学习性及残余信用一致性，再冻结一个有明确问题和停止线的dev验证包；不自动把此一epoch探针扩成旧三seed补训。若发现确定性实现缺陷继续修复，不把它升级成VISION前置讨论。只有确实需要改变核心合同或资源范围才暂停请求决策。

## Dev 验证包冻结（2026-09-17；§唯一后续的第二半）

train 可学习性与残余信用一致性双绿后（`reports/r2_h3_7b_validation/learnability_analysis.json`，commit `768cca14`），冻结 dev 验证包：

1. **问题**：修复链路（fixed prior + 0.75/0.25 混合信用）下，byte 对齐 16-byte 窗口 target（B）相对 legacy char-safe target（A）在 fixture **dev split（8 episodes）**上是否改善 teacher-forced surprise/accuracy；两臂各自零步→训练后的 dev 变化作为 train→dev 迁移诊断。**性质与边界**：修复效果与几何比较的工程验证——fixture dev 经既往开发，结果仅诊断、不作为泛化或能力证据；final 不读；不与旧 H3.7 dev 数字比较。
2. **冻结预算**：seed 20260917；两臂各 **2 epochs**（12 train episodes × 2 = 24 episodes / 1074 byte 更新）；CPU 单线程；300k 参数预算；preflight 机制与工程验证臂完全一致；dev 评估只读且均自保存 checkpoint 的探针副本执行。wall cap 30 分钟。
3. **停止线（任一触发即停止，不扩预算）**：① 非有限值/参数预算超限/更新预算不匹配；② 两臂训练后 dev surprise 均不低于各自零步基线（train→dev 迁移为零）；③ B 臂训练后 dev surprise ≥ A 臂训练后 dev surprise（byte 对齐几何无 dev 优势，几何问题回答为负）；④ wall 超限。触发即如实落账，不自动加预算或 epoch。
4. **产物**：`reports/r2_h3_7b_validation/dev_<arm>.json` × 2 + 汇总；roadmap 更新；dev 结果不得据此晋级 L2 或宣称能力收益。
