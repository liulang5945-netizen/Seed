# SPEC-N1-12 · 原生证据候选接口与学习归属（2026-10-15）

状态：隔离研究候选工程合同，非默认采用、非能力通过、非训练批准。承接11参考工程结论，不改变既有冻结评测或母档。用户最新目标要求一致推进、按实际情况修订计划；本件属于可逆实现与检查。

## 身份与复用选择

以 `TAIJI_CORE_REQUIREMENTS` CR-1/5/8/9、`TAIJI_NATIVE_ARCHITECTURE_V1` §0.2及 `ARCHITECTURE_DIRECTION` §2/3为准。具体GRU、embedding、内容寻址和optimizer不被身份合同禁止；Taiji拥有参数/状态/决策、Seed没有隐藏认知、真实和生成分源、checkpoint自足才是不变边界。不能把旧TSK-v8单tick无autograd实现当完整Taiji的永久定义，也不能把一个新GRU叫作完整Taiji。

唯一实施选择：新增 `taiji/native_evidence_candidate.py`，复用11的逐位置写入、回答重读及参考字节解码算子，增加版本化Observation、会话持续状态、只读问题和自足bundle边界。不是复制第二份算法或修改默认adapter；候选可在移除Seed/Legacy后独立运行。与当前局部TSK路径的能力/保持比较仍欠，绝不向旧路径补发梯度证据。

## 输入、持续性与学习责任

- 显式会话ID由宿主可信调用参数提供，不从文本内容推断；只接受 `Observation(modality='text', value=非空str/bytes, provenance='external')`。来源和时间戳作为审计身份，不喂给编码器，严格单调timestamp避免迟到覆盖。重复source不静默覆盖。
- 每次observe仅将合法外部bytes加入当前会话证据；生成/query/imagined拒收。不同会话可使用相同来源名，按会话隔离。question接口只读，不追加证据；无证据时拒绝生成。
- 每次构图从该会话原始观察顺序重算逐位置状态，保证更新后不消费旧参数产生的缓存；每source内部有写入递推，跨source寻址，回答期每步重读。跨source因果时序暂未建模，不能声称完整持续工作空间或世界模型。
- 可学习参数全属候选：embedding、writer、question、key/value/query、decoder、readout。teacher-forced目标只进损失接口；训练器负责loss和optimizer，推理接口无target。末端loss到最早write的完整信用及全write-detach负对照复用11，不将optimizer更新等同能力。
- 同信息对照使用相同源码、参数维度和全部观察；末态对照保留每个source末态而非仅最后source，不能因为某source恰好排最后而剥夺基线其他材料。正式比较需同数据/更新/seed/预算，未来训练合同单独冻结。

## 保存、迁移与出口

候选bundle独立schema，保存配置/参数、全部会话的Observation payload、严格时间戳顺序、RNG和可选optimizer状态。拒绝未知schema、重复会话/来源、来源或时间不合法的恢复；不静默吞缺失参数。兼容 `NativeCheckpoint.components` 的候选payload插槽由后续集成实现；本件不宣称默认adapter已挂载或正式envelope集成通过。

工程测试覆盖：真实Observation接入、会话间相同source隔离、问题/生成不写入、来源与时间拒收、跨source末态对照保留各source、信用负对照、weights-only存档/恢复后多会话输出一致。正式任何训练前必须再做候选自身零更新与优化器新进程恢复；11参考恢复不能替候选补发。

下一步唯一为候选自身独立恢复与资源标定，之后才设计冻结未见组合课程与配对能力训练。当前不采样冻结16项做拟合、不改三器官/净增益出口、不覆盖母档、不推送或删数据。
