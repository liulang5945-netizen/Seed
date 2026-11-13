# U2请求条件证据读取候选

> 22.5.P3.2，2026-11-12。依据P1/P2负结果及链路核查提出的架构干预。现行U2与旧实验身份保留，本稿定义新候选，能力结论待正常主体实验。

## 1. 目标与边界

同一主体用当前请求直接查询已经获得的公开感知证据，补足当前request→H汇聚→对象查询的间接路径。原对象槽、动力学、H/F、连续goal及原生字节生成共同保留。候选不读取物理引擎的实体坐标、答案、颜色名对照表、gold或任务oracle。

本次只改读取组织。自驱、继承生长和反馈保持仍沿U2主体合同，须后继任务验证。当前研究父没有已核验的A能力，参数继承与功能保持分别登记。

## 2. 证据表示与两次组合读取

对已合法接收事件j的模态m、token n，用当前编码器获得e_jmn∈R^d。c_jmn为原公共8维坐标/时间/缺失元数据，u_jmn=[e_jmn;c_jmn]∈R^(d+8)。模态身份来自现有编码器，时间与来源字段来自原包。token细节、空间坐标与来源保持关联，不预先对整个场景求均值。

设每支路H读取h_b∈R^d，当前request编码q∈R^d，原branch_context为c_b。证据读取使用共享参数，单头宽度D=d/heads，首版d=8、heads=2、组合步数L=2。

\[
r_b^0=W_{in}[q;h_b],\quad K_n=W_Ku_n,\quad V_n=W_Vu_n.
\]

每步每头a先计算Q_ba^l为W_Q r_b^l的分头结果，随后

\[
s_{ban}^{l}=\langle Q_{ba}^{l},K_{na}\rangle/\sqrt D,\quad
\alpha_{ban}^{l}=\operatorname{softmax}_{n\in\mathcal E}(s_{ban}^{l}),
\]
\[
v_b^l=\operatorname{concat}_a\sum_{n\in\mathcal E}\alpha_{ban}^{l}V_{na},\quad
r_b^{l+1}=r_b^l+B(W_Ov_b^l),\quad
c_b'=c_b+0.1\,W_{out}(r_b^L-r_b^0).
\]

B(x)=x/sqrt(1+mean(x²))采用现有平滑有界映射。Q/K/V及W_in/W_O/W_out均可学习。0.1为固定初始残差尺度，候选不宣称函数中性。W_out采用非零常规初始化，避免零输出投影令首步Q/K/V梯度被结构性屏蔽。所有支路和两次读取共享参数。后一次查询依赖第一次读出的证据，能表达组合读取，实际是否学会由对照判定。

有效token由来源与权限验证得到布尔mask，mask仅排除缺失/撤销/未授权证据。它不按答案或实体类别挑选。空证据时显式返回c_b、全零注意力及available=false，避免全−∞softmax。该状态交给现有主体不确定性处理，不能据此宣称模型会正确输出未知。

## 3. 参数与状态合同

局部算子输入为request[d]、read_h[branches,d]、原context[branches,d]、encoded_tokens[N,d]、metadata[N,8]及valid[N]，输出为context'[branches,d]、attention[2,branches,heads,N]及available。参数以同一dtype/device注册，float32/64均检查。d必须可被heads整除，N与数组形状不符拒绝。合法证据共同置换应保持输出不变，注意力同步置换。

主体集成时新增的证据目录登记canonical原始事件id/摘要、模态/token范围、原始时间、授权/撤销状态和编码版本。首版最多4个完整事件、128个token，按时间与稳定事件id排序。容量不足必须具名停止或按冻结规则转unknown，不能静默截掉本任务需要的事件。current/previous由同一个可学习读取器使用原时间元数据解释。当前/前次目录是来源管理，不预选正确目标对象。首版超窗能力另记范围。

持久规范仍为已接收的原始公开包。每次慢参数更新和恢复后按当前编码器从canonical包重建可微e，派生K/V缓存绑定编码版本，版本变化失效。禁止从旧detach编码推断任务信用到达感知器；禁止将任意历史、计划或oracle直接注册为感知证据。四事件上限含重编码CPU/内存，不能把历史重建成本藏在训练之外。

## 4. 信用与恢复

answer_loss→原字节表达→c'→Q/K/V及当前编码器形成任务信用。仍检查原H/F/对象路径的使用。继承P2参数须有显式迁移清单，新增参数单列；源文件变化后旧checkpoint的source pin按旧快照复核，通过迁移工具载入，不能关闭旧身份校验。

主体schema、参数清单、state重建与runtime工厂需一起覆盖。当前局部算子实现通过只能签模块检查，正常begin_observation/assimilate/set_request/begin_utterance/output/accept_byte及独立进程恢复全部接入后才能签P3.3集成交付。

## 5. 验证与比较顺序

1. 局部检查包括有限性、空证据、缺失token严格屏蔽、token置换等变、请求×内容交互、各声明参数与感知输入的梯度、两步读取的值依赖、state_dict恢复及float32/64。
2. 正常主体检查包括完整原生字节输出、目标/请求独立、事件来源/权限、撤销、旧参数迁移、重编码、独立进程恢复和原路径回归。
3. 新合同比较旧读取、直接读取、参数配平控制及匹配Transformer。保留相同自由输出与任务分母，以同训练CPU主比较，另报同更新诊断及完整成本。来源/时间打乱、request置换、证据切断和关闭新读取为必要干预。
4. 训练数据、fit新分区、验收门、种子、成本与停止在实价后冻结。若直接读取无输入依赖收益，保留负结果并全局重审感知/表达优化。不得再追加同目标的无边界探针。

P3.3已按[主体集成回执](VALIDATION-TAIJI_U2_EVIDENCE_SUBJECT_20261112.md)交付。P3.4.1研究父及P3.4.2实价/分区均已交付，P3.4四臂已[完整结案](CLOSEOUT-TAIJI_U2_EVIDENCE_LEARNING_20261112.md)，当前唯一后继为[P3.6.4正交学习实价与准入](PLAN-TAIJI_U2_STATE_CONDITIONING_20261113.md)。

## 6. 集成语义补充

实际候选使用原evidence保留公开包，新增evidence_directory保存id/规范内容摘要，revoked_evidence保存直接读取权限撤销。目录摘要排除编码器subject_version，原包其余字段参与摘要，以支持在保持原始事实的同时迁移编码器。模型bundle采用taiji-u2-query-evidence-subject-v1，typed runtime工厂、参数迁移及raw replay共同恢复该候选。

坐标第6位是坐标类型标识，RGB合法token由已接收的模态集合确定；proprio缺失按原有效位处理。当前读取元数据将原相对时间a_3/a_4转换为asinh(sinh(a)+t_end−world_time)，保持原时间窗宽度及相对年龄。排序依据t_end与稳定packet id，id不作为语义向量输入。实际包含16 RGB、7 audio、1 proprio，共24token。

目录及被撤销包均保留规范来源核验，缺包/摘要不一致拒绝恢复。撤销合同覆盖后续直接证据读取权限，H/F的历史影响与知识纠正效果按后继B合同验证。原始历史和当前编码器共同构建正常任务图，派生K/V每次重新计算。原U2及该候选分别保有显式身份。
