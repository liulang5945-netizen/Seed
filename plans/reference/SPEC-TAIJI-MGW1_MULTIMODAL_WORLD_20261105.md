# 态极MGW1：多模态世界信念、动作预测与纠正算法

> 状态 [完整算法候选规格·非能力声明]，2026-11-05起草、2026-11-06补编。owner要求直接补齐算法。本件将上一份[职责设计](DESIGN-MULTIMODAL_PHYSICAL_WORLD_20261105.md)的未定义接口替换成确定形状、运算、更新顺序、目标与失败出口。MGW1是新候选，不是当前CF1的代数整理，不更换现有运行/训练，不将旧任务成绩迁移给它。参考核验范围单列，完整工程图和任务能力仍须按此规格实现后取证。**2026-11-06补编闭合两处起草时缺口**：§3.1对象槽生命周期（原§3只声明“槽可合并/重复/闲置”未给操作符）与§5.1分支塌缩诊断（核验明列“不塌缩”未签）；补编算子经[独立预审](../../reports/taiji_mgw1_lifecycle_supplement_20261106.json)代数核验，并固定§8发射循环的FSM屏蔽归一口径。§1–§12原文不改。

## 1. 唯一首版选择

采用**专用局部感知＋K套持续对象假设＋动作条件先验＋观测创新纠正＋受许可误差校正F＋共同H读出＋有限信息规划**。不先平均互斥世界状态；预测分布在最后混合。首物理profile是固定相机的二维可推动物体域，不声称3D、真实机器人或任意物理规律已覆盖。

首场景含1–3个不可破坏但可遮挡的物体，实际操作在[-1,1]^2；运动响应可跨episode变化，真值物体ID/质量/摩擦不输入。目标、公开期限及实际完成/失败由环境合同给出。M/K/d是首profile资源范围，非永久规模上限；扩域或增加支持须新版本和保持验收。

| 配置 | 首版 |
|---|---|
| 公共宽度d、神经群n、群内单元m | 128、4、256；群内R/P可按§12出生 |
| 世界假设K、对象track数M | 3、4；是内部假设索引，不是环境ID |
| 每分支快矩阵 | F∈R^(32×32)，不同对象/群用不同query读同一F |
| 观测内纠正轮R | 2；物理时间只在prior推进一次，似然只结算一次 |
| 图像 | RGB float[0,1]，64×64×3，8×8 patch，64×192 |
| 音频 | 单声道[-1,1]，16kHz×160ms＝2560采样；Hann窗256/hop128/center=false，19×129的log1p幅度 |
| 文本 | ≤256 UTF-8原始字节，embedding64，3层因果卷积kernel3/dilation1,2,4，通道d，保留逐字节位置 |
| 本体感受 | 8个已标定[-1,1]通道和有效mask，不是物体位置/质量真值 |
| 时间/预测预算 | 运动子步≤0.05s，单次≤20子步；规划深度2、每节点候选≤6、观测样本2、完整树最多157节点，硬上限256 |
| 存储/回执 | 原始E及journal各16MiB上限、最多512包；溢出明确停止/请求维护，不静默截断 |

小维数学参考使用相同参数化运算缩小d/n/M/K/key，不能把其结果称为首版全尺度或真实任务已通过。

所有新头默认MLP_(a→b)(x)=W2 tanh(W1x+b1)+b2，W1形状(2d,a)、W2形状(b,2d)。Linear按输出×输入排列。B(x)=x/sqrt(1+||x||²)，N(h)=h/sqrt(1+mean(h²))，sigmoid/softmax/LSE用稳定形式。状态解释参数属于Θ；策略/价值专属MLP属于Psi_pi/Psi_V，更新不改变状态坐标。权重默认std=1/sqrt(fan_in)、bias=0；h0=tanh(不同小幅种子)、p0=tanh(内部网格种子)，均非真值。nu为8个标准化实测量＋8个missing位，D_nu形状d×16。参考FD可用另列非退化压力参数点，不用它签默认初始化学习有效。

## 2. 输入、因果时间与原始证据

优化清单排除softmax公共平移规范项：goal_score.bias、actor末层bias、relation_geometry末层bias，以及Read_H/Read_O/relation/decoder的共享key bias。它们不算可学信用贡献。首profile的proprio decoder坐标恒0，其query投影weight也为profile-inactive，只保留query bias可学；若将来加坐标再另声明。这些冗余张量可在参考中保留作零对照，不能据其非None梯度宣称有效学习。

每包p=(packet_id,source,kind,t_start,t_end,t_arrival,frame_id,calibration_revision,valid,payload)。仅t_end≤当前真实时钟、t_arrival≤当前时钟且来源合法的包可消费。source/split/rule/object真值ID不神经编码；packet_id只去重和审计。

一次推断消费的是fusion event：同一公开step/result_bundle_id的预期传感/Outcome集合，t_end相同，audio窗口可向过去延伸。全部到达或首包到达后.1s wall封束为missing，之后只做一次联合likelihood、R轮纠正及FP1。t_end比较环境物理时钟，t_arrival/封束超时比较单调wall时钟，不直接比较两种时钟数值。bundle ID只分组/审计，不神经编码。

fusion event按非递减t_end消费，同一bundle迟到成员或更早已封束事件记late_unassimilated，不补一次当前Corr；若合同要求该信息，返回late_requires_replay/求新观测。首profile仅处理公开action-step快照，完整平滑重放另分版本。不能改写历史已执行动作/预测。参考correct()的多模态字典就是一个已封束event，不是任意逐包/合包实现任选。

文本command/goal是外生控制条件，不纳入物理传感似然；已核验结果可提供reward∈{-1,0,1}与done布尔，未到达时各自missing。自生成答案、想象和未来监督不作为新真实观测。首profile是步进模拟器，物理dt来自环境，CPU/wall计费另账；不假装现实在计算期间必然暂停。

首profile的模态有效性由动作/设备计划决定，给定这些控制时不额外依赖隐藏分支。若缺失本身与隐藏物理量相关，需为mask建立likelihood并另分版本；不能删掉该证据却声称原Bayes解释仍完整。图像内遮挡不等于整个模态missing。

### 2.1 四模态编码的确定运算

RGB按行优先切patch；audio按上表STFT。局部raw-codec向量经Linear_(D_in,m→d)，RGB输入192、audio129、proprio输入16＝8个置零值＋8个valid位；proprio emission仍8维，只对valid坐标结算似然，全invalid视为missing。文本embedding先Linear_(64→d)再3个tanh残差卷积。每token加学得模态向量及Linear_(8→d)xi后做B。

Hann用periodic=true，FFT normalized=false、onesided=true，win_length=n_fft，无center padding；log1p(abs(STFT))。RGB patch按行列网格顺序，patch内按行列通道展开。文本卷积左padding=2×dilation、右0，逐层x←x+tanh(Conv(x))。

xi=(x,y,u_seq,asinh(lag_start/T0),asinh(lag_end/T0),spatial_valid,sequence_valid,duration_valid)，T0=1s。图像x/y是patch中心[-1,1]，音频u_seq为帧位置，文本为字节位置，其他不适用坐标为0并置对应valid=0。时间lag相对包end，仅用实际已见区间。缺失模态不产生零token，观测到的静音/黑色仍是有效数据。

首profile文本只作command/goal和显式语言任务，不当物理sensor likelihood。非空goal用Linear_(d→1)在文本tokens上softmax，加权Linear_(d→d)值；空goal沿用Γ，无goal初值0，不另添null参数。

reward/done是第五种非空间反馈token：6维Outcome→Linear_(6→d)＋独立模态向量后做B，任一valid结果存在才生成；它有自己的binding K/V、null和可靠性头，不是第四种sensor decoder。传感与Outcome都missing才是全missing；goal-only不触发物理证据/F事实写入。原始媒体存E、缓存带revision，坐标参数改变按§12迁移。

## 3. 持续状态及假设语义

每个分支b持有S_b=(H_b,F_b,{o_bi}_{i=1..M})；o_i=(h_i,p_i,v_i,s_i,l_i)，维度为d+2+2+4+1=d+9。h∈[-1,1]^d，p/v为二维归一图像坐标/响应速度，s为4个正尺度量（非已证真实协方差），rho_i=sigmoid(l_i)。另存归一log权重ell_b、E/Γ/ν、RNG、pending动作/反馈、参数/编码/结构revision及journal watermark。

初态H/F=0，p为内部网格先验、v=0、s=.1、l=0；K分支的h种子不同。参数在分支间共享。分支索引是持久离散解释模型，状态是历史的确定函数，不称粒子滤波；没有proposal/importance修正的含义。槽可合并/重复/闲置（操作符由§3.1补编定义），归一化不证明发现了真实物体，必须用任务与干预另验。

### 3.1 对象槽生命周期（2026-11-06补编）

原§3声明“槽可合并/重复/闲置”但未定义操作符，本节闭合该缺口。三个假设管理算子只用模型内部量——上一事件的绑定质量、null质量、token坐标与编码、槽自身状态——不读对象真值/分割/ID。算子是离散事件：先写单调journal（event_id、slot_id、触发统计、变更前后参数摘要）再改状态，历史推断不回改；执行位置固定在§11步骤1a（事件校验后、prior推进前），输入是上一事件的统计；生涯首事件无上一统计则跳过。

**出生（birth-from-null）**。记上一事件空间模态（首profile仅RGB）token n的null绑定质量a_n与坐标xy_n。arena划为4×4网格bin，Q_bin=Σ_{n∈bin}a_n，A=Σ_n a_n。若A≥θ_birth=0.15且某bin满足Q_bin≥κ_birth·A（κ_birth=0.6），该bin为出生簇；从满足g_i<θ_idle=0.02的槽（含退役槽）中取rho_i最低者（并列取稳定槽序号最小），执行

\[
p_i\leftarrow\frac{\sum_{n\in bin}a_n\,xy_n}{\sum_{n\in bin}a_n},\quad
v_i\leftarrow0,\quad s_i\leftarrow0.1,\quad l_i\leftarrow0,\quad
h_i\leftarrow\tanh\!\Big(Linear_{d\to d}\big(\tfrac{\sum_{n\in bin}a_n\,B(V\,token_n)}{\sum_{n\in bin}a_n}\big)\Big).
\]

出生只改该槽的o_i并追加journal；其他槽、H、F、E、Γ、branch权重与全部历史量逐位不变。质心与h种子是模型内部统计，不是真值位置/外观。单事件至多出生1槽，无合格bin时不动作。

**退役（retire）**。槽i连续T_retire=8个fusion event满足g_i<θ_idle且期间总null质量A<θ_birth（无出生需求），则退役：l_i←−8（rho≈3.3e-4），h/p/v/s保留为归档值。场景关闭回执（settled_terminal且环境声明场景结束）触发全部槽逐槽退役并journal记录；跨场景对象身份连续性不属于首profile，不冒称保留。退役是假设管理事件，不是证据删除：E与journal原样保留，“未看到≠不存在”由可追溯journal承担；证据重新出现时按出生算子复活退役槽（退役槽rho最低，天然优先被选中）。退役槽在后续绑定竞争分数中被log(eps+rho_i)项相对rho=0.5压低约−7.3 nats，数值上退出竞争（预审核验）。

**合并（merge）**。两槽i,j满足‖p_i−p_j‖<δ_merge=0.05、‖v_i−v_j‖<δ_v=0.5，且上一事件它们在各模态token上的绑定质量向量（跨token拼接）余弦>κ_merge=0.8时，合并到i：

\[
h_i\leftarrow\tanh\!\big(\tfrac{g_i h_i+g_j h_j}{g_i+g_j+\epsilon}\big),\qquad
(p_i,v_i,s_i)\leftarrow\tfrac{g_i(\cdot)_i+g_j(\cdot)_j}{g_i+g_j+\epsilon},\qquad
l_i\leftarrow\mathrm{logit}\big(\tfrac{\rho_i+\rho_j}{2}\big),
\]

j槽退役并记merged_into=i。合并每事件≤1次，对第三槽中性，journal记录两侧快照。

**失败出口与对照**。连续E_storm=5个事件触发出生→报hypothesis_churn_stop并停该主体推断（仪器停止，不静默继续）；任何算子异常按engineering_failure出口结案。生命周期由布尔开关lifecycle_enabled控制（默认true），关闭时§3原语义不变、逐位回到无生命周期图。正式比较必须含无生命周期对照臂（同形同预算）；机制贡献未过预注册门不得宣称必要。θ/κ/T/E_storm为首profile研究准入值，改值须新版本。

**不宣称**。出生/合并/退役是假设空间管理，不是对象发现或跟踪能力；null簇可能来自遮挡边缘、噪声或真新物体，区分它们是任务与干预的职责。槽归一化与质量加权不证明对应真实物体，原§3边界全部保持。

## 4. 动作条件prior

动作描述a∈R^15：8种mode one-hot＋接触坐标2＋力2＋持续时间1＋已知资源成本2。坐标来自模型各假设的预测，不将slot号交给环境当物体ID。A_emb=B(Linear_(15→d)(a))。

令z_i=B(Linear_(d+9→d)(o_i))。关系消息

\[
w_{ij}=softmax_{j\ne i}\{(Qz_i)^TKz_j/\sqrt d+MLP_{4\to1}([p_j-p_i;v_j-v_i])+\log(eps+rho_j)\},
\quad m_i=\sum_{j\ne i}w_{ij}B(Linear_{d+4\to d}([h_j;p_j-p_i;v_j-v_i])).
\]

M=1时m=0。全算子参数共享，不使用track序号特有偏置。

F读取定义q_i=B(Linear_(4d→k)([z_i;Read_H(H,g);g;A_emb]))，r_i=Linear_(v→d)(F^Tq_i)。Read_H的query=B(Linear_(d→d)(g))，对群的K/V Linear_(d→d)做单头softmax读取；对象读取Read_O同理，用z_i加null，槽权重含log rho_i。群条件F-query的4d输入为[N(H_g);Read_O;g;A_emb]，复用Q_F；所有权与形状固定，不用h.mean()。u_i=[h_i;m_i;r_i;g;A_emb;p_i;v_i;dt]∈R^(5d+5)。MLP_(5d+5→d+6)输出h驱动、加速度2、过程尺度4。

\[
h_i^-=(1-ell_o)h_i+ell_o\tanh(u_i^{h}),\quad ell_o=1-e^{-1/4};\qquad
v_i^-=\Pi_{\|v\|\le4}(v_i+dt\,2\tanh(u_i^a)),\quad p_i^-=\Pi_{\|p\|\le4}(p_i+dt\,v_i^-),
\]
\[
s_i^-=clip(s_i+dt[10^{-4}+.1\,sigmoid(u_i^s)],10^{-4},4),\qquad l_i^-=l_i.
\]

这是有界运动响应归纳偏置，不是已证真实牛顿方程；p不自动是米制坐标。物理dt与神经ell时钟分开。大dt切≤.05s子步，超过20步预算返回extrapolation_budget_stop，不静默少走时间。

H使用原生n群方程：x_g=[N(H_g);e_event;m_g;r_g^F;r_g^O;g;D_nu nu]∈R^(7d)，a_g=W_eff x_g+P tanh(Rx_g)，H_g'=(1-ell_g)H_g+ell_g tanh(a_g)，tau初值1/4/16/64微步、ell=1-exp(-1/tau)。e_event在prior为0，r^O为对z_i的群条件注意力，m_g为群间Q/K/V消息。运动每子步一次，观测内R轮只做推断，不再推进p/v的物理dt。

MGW1的W_eff是直接Theta矩阵，不启用H的在线低秩adapter。m_g按softmax_j((Q_g H_g+Q_goal g)^T K_j H_j/sqrt d)混合V_j H_j；Q_g/K_j/V_j各为n×d×d，Q_goal为d→d。Read_H的投影与这组群间投影分名，不能在参考中暗中共享成另一图。

## 5. 看当前观测前的规范化预测分布

对每个模态m的已知局部坐标xi_n，query=B(Linear_(8→d)(xi_n)+e_m)。以objects z_i为K/V，另有始终合法背景向量，注意力含log rho_i；context为加权和。proprio的context还读取Read_H。每模态MLP_(d+8→2D_m)输出mu/sigma；proprio将Read_H投影相加到context，不改变输入宽度。

采用固定codec与有界截断正态TNormal(mu,sigma;L,U)：RGB的D=192、区间[0,1]；audio的D=129、区间[0,6]；proprio D=8、区间[-1,1]。mu=L+(U−L)sigmoid(head_mu)，sigma=(U−L)(.02+.18sigmoid(head_s))。归一密度为

\[
p(y)=\frac{\phi((y-mu)/sigma)}{sigma[\Phi((U-mu)/sigma)-\Phi((L-mu)/sigma)]},\quad L\le y\le U.
\]

音频幅度范围来自[-1,1]输入及固定窗；超profile输入明确拒绝/换profile，不悄悄截断。RGB首profile是连续float模拟观测，真实uint8接入须声明量化bin概率或dequantization，不能将密度直接叫8bit质量。

分支sensor likelihood是所有实际observed且valid的codec标量密度乘积；missing因子=1。reward用三类softmax，done用Bernoulli，头为MLP_(d+15→4)([Read_H;a])。其真实字段到达时才乘相应likelihood。学习型模态可靠性只用于后文融合，不任意乘log-likelihood而继续称原模型后验。

\[
log\,omega_b^+=log\,omega_b^-+log\,p(y_t\mid S_b^-,a_{t-1})-LSE_c(\cdots).
\]

所有观测为空时直接复制原log权重，避免无意义数值重归一。此式是本规范定义的有限分支自回归模型之离散分支后验；不是环境真实物理状态的精确后验。后续神经校正后的状态仍是有限近似。不得把吸收当前观测后的重建误差拿来计算上述predictive evidence。

### 5.1 分支塌缩诊断与对照臂（2026-11-06补编）

K套分支权重omega没有内生机制阻止质量塌缩到单分支（混合模型的经典失败模式；[核验§3](VALIDATION-MGW1_REFERENCE_20261106.md)明列“不塌缩”未签）。本节把它变成可观测、可对照的责任，不新增默认组件。

**报告义务**。每个fusion event计算有效分支数ESS=1/Σ_b omega_b²∈[1,K]与分支熵，随journal落账；训练/评价回执必须报告ESS时间序列摘要（min/中位/末值），缺该字段记仪器不完整。若ESS<θ_ess=1.2连续T_ess=10个event，报branch_collapse_warning：只报告不强制干预，随后事件照常推断，是否处置由后续合同裁定。

**预注册对照臂（不默认启用）**。若正式比较出现塌缩告警，允许立两个单一变量研究臂：①温度臂——§5权重更新改用log p(y_t|S_b^-,a_{t-1})/T，T=2冻结；②下限臂——分支权重omega_b←max(omega_b,0.05)后归一，0.05冻结。两臂与默认臂同形同预算，须先过活性门（触发占比与梯度范数，沿用既有纪律）；未过门不得写“温度/下限解决塌缩”。

**不宣称**。ESS高不证明假设覆盖真相（K个错误假设同样ESS=K）；ESS低不自动判失败（真单态环境下单分支可能是正确后验）。塌缩判读必须结合任务与干预读数，本诊断只提供可观测性。

## 6. 观测创新、遮挡与跨模态绑定

各模态对token n计算M个object与一个null的列竞争softmax：s_ni=(Q_bind z_i^-)^T K_m token_n/sqrt d＋图像时的−||xy_n−p_i^-||²/(.01+s_i,p均值)＋log(eps+rho_i)。null分数是独立Linear_(d→1)(token)，不使用真实分割或对象ID。

每模态按N_m归一：mass_i^m=mean_n alpha_ni，c_i^m=mean_n alpha_ni B(V_m token_n)。binding域为rgb/audio/proprio/Outcome；仅在有token的域上softmax可靠性rho_m（Linear_(d→1)(mean token)）。mass_i=sum rho_m mass_i^m，c_i=sum rho_m c_i^m，obs_i=c_i/(mass_i+.01)。mass_pos=sum_{m有空间坐标}rho_m mass_i^m，g_pos=mass_pos/(mass_pos+.01)，首profile仅RGB为空间观测。由||B(Vtoken)||<1得||obs_i||≤1；均匀重复同模态token不因分辨率改变聚合量。

S^(0)=S^-，r=0,1时以S^(r)重算z/位置核/绑定/质量、F读取和H；F在两轮中保持旧值。MLP_(3d+9→d+9)([h_i^(r);obs_i^(r);r_i^F;p_i^(r);v_i^(r);log s_i^(r);l_i^(r)])输出delta_h/delta_p/delta_v/s_target/l_target。g_i=mass/(mass+.01)，h_i^(r+1)=(1−g_i)h_i^(r)+g_i tanh(delta_h)。p/v分别加g_pos×tanh(delta_p)、g_pos×2tanh(delta_v)，再投影半径4；s在g_pos下向[1e-4,4]的sigmoid目标插值，l在g_i下向8 tanh(l_target)插值。最终S^+=S^(2)。H辅助drive是各实际模态token均值的均值，局部tokens仍参与逐对象绑定，不是唯一信息通路。

无空间观测时p/v/s的观测纠正为0，先验仍可推进。无任何观测时整段Corr直接返回prior，包括H、objects、F、weights；有图像但物体被遮挡不等于全missing，正确null匹配须学习和取证，不声称软注意力自动解决遮挡。

R=2轮中重算绑定/创新，H取同模态均值驱动及对象条件读取按§4原生方程更新；均值只是辅助drive，局部token和对象路径仍在。likelihood与E/F事实提交不重复R次。

## 7. FP1：受许可的批量误差校正快权重

每分支以prior保存的u_pred,i=[z_i^-;Read_H^-;g_receipt;A_exec]∈R^(4d)形成key_i=B(Linear_(4d→k)(u_pred,i))，以实际obs_i、合法Outcome和source配对形成value_i=B(MLP_(2d→v)([obs_i;outcome_embed]))。outcome_embed为Linear_(6→d)([reward三类onehot;done;valid_reward;valid_done])，未到达字段置0且valid=0。Outcome缺失用显式valid，不能把原完整query再次塞进value恢复错绑定。运行写value可学；不是EMA目标stopgrad。

global chi=p_allowed·1[有实际观测]·sigmoid(MLP_(2d→1)([mean obs;g]))，alpha=sigmoid(MLP_(2d→k)(同输入))，beta_i=chi·g_i·16sigmoid(MLP_(2d→1)([z_i;obs_i]))。p_allowed由真实/合法反馈/获准重放决定；想象、未来或自生成内容禁止真实F写。

\[
F_0=D(1-chi+chi\,alpha)F,\quad
F^+=\arg\min_{\|X\|_F\le C_F}
\tfrac12\|X-F_0\|_F^2+\tfrac12\sum_i\beta_i\|X^Tk_i-v_i\|^2,\qquad C_F=4.
\]

禁止写入或chi=0时直接返回F，不借投影修改。FP1改变旧chi整体差量插值，单独分版，不叫v1.1等价整理。批量求解对object行重排不变，不按track顺序串写产生顺序伪效应。

令A=I+sum beta_i k_i k_i^T，B_F=F0+sum beta_i k_i v_i^T。无约束X=A^-1 B_F；若超球，用X_mu=(A+mu I)^-1 B_F、共同mu≥0使||X_mu||=C。mu_hi=max(1,||B_F||/C)，64步二分取可行侧；C=0直接0。禁止naive detach mu再宣称精确active梯度。

对active分支，Z=(A+mu I)^-1 X，固定C下dmu=<Z,dB_F−dA X>/<X,Z>，dX=(A+mu I)^-1(dB_F−dA X−dmu X)。反向令Y=(A+mu I)^-1 G、c=<G,Z>/<X,Z>，grad_B=Y−cZ，grad_A=−(Y−cZ)X^T；若C可微，grad_C=cC。inactive不减cZ。

近似证书：c0=.5||F0||²+.5sum beta||v||²，dual(mu)=c0−.5tr(B_F^T(A+mu I)^-1B_F)−.5mu C²。精确算术中，可行X_hat的J(X_hat)−dual(mu)上界目标差，距离≤sqrt(2gap)。机器证书还须检查可行性并给舍入上界；普通float gap仅是数值估计，不能把负gap夹0后报告零误差。边界切换不强判中心FD。v_i=0且固定门/key下非扩张，不推广全图收缩。

写后用同一Q_F再按各object/query读F^+，经MLP_(3d→d)([h_i^+;r_i^{F+};g])和g_i门控更新对象h，再做一次H读出。禁止用整个F快照的池化值代替query读取。实体坐标和物理dt此时不再推进。

## 8. 表达、候选动作与信息规划

每分支由goal/上一字节query读取H、objects和F，z_b=MLP_(3d→d)([Read_H;Read_O;Read_F])；字节logits=Linear_(d+64→257)([z_b;embed(prev_byte)])。语言仅由外生answer_request触发，首版不把它当第九种RL动作，也不给字节头套mode的REINFORCE。utterance scratch权重eta^(0)=omega，P_byte=sum eta_b softmax(logits_b)；接受字节后仅在scratch按该发射概率更新eta，世界omega不因自己的输出重加权。

输出上限64字节、初始prev=0；UTF-8前缀FSM只屏蔽不合法字节，EOS仅在完整字符边界合法。上限无EOS标truncated，不当成功。实际字节只推进一次H；F和物理p/v/s不变成“自己说的事实”。teacher-forced目标只在评分副本，不提交真实历史。首profile不启用原文复制，Task1旧资产另保留，联合迁移需独立保持合同。

物理候选：仅p_i在[-1,1]^2可执行域内才给接触点，枚举±x/±y单位力，另有OBSERVE/WAIT/REST。首profile固定正交相机，归一图像平面与arena相同，接触点即p_i，力轴同xy、单位由环境公开标定；视野/执行域外不外推接触。重复坐标可保留，不平均互斥位置。ASK无外部测量合同，首profile合法mask=0。ACT/OBSERVE/WAIT持续.05s，REST .2s。OBSERVE返回所请求真实传感，ACT返回proprio/reward/done，WAIT/REST无新传感；不能按正确动作改可用性。

mode顺序为ACT/OBSERVE/WAIT/REST/ASK/REPLAY/CONSOLIDATE/GROW。后3个只在维护窗及相应源/参数/预算权利成立时入候选，按§12定义实际操作；它们不是外部物理动作。

softmax前处理空集：已有pending返回PENDING且不执行新WAIT/ACT；真实terminal且无维护权限返回TERMINAL；硬资源耗尽返回BUDGET_STOP；其他无合法动作返回UNAVAILABLE并保留目标。WAIT只有真实合法时才保留，不对全mask做softmax。规划子节点已terminal取V=0；资源/操作不可达的非终态叶取保守下界−2，标为近似回退，不冒充已收到的真实reward。

先按一阶reward+critic预分数选top4，强制保留OBSERVE/WAIT，共≤6；tie稳定按(mode,branch,track,force序号)。深度2、每节点S=2的完整树≤157节点，256节点cap足够；剩余资源不足以完整评估某根动作就整项屏蔽，至少保留可执行WAIT，不能比较半棵树。定义d_credit(a)=max(1,dt_phys/.05)、g_a=gamma^d_credit。c∈[0,1]为归一CPU/wall/动作/保持风险（各clip[0,1]）的均值，原始费用另全记；采用效用u_a=(1−g_a)(r−c)。固定噪声RNG：

\[
Q_h(B,a)=(1-g_a)(\hat r(B,a)-c(a))+g_a\frac1S\sum_s(1-done_s)V_{h-1}(Assimilate_{sim}(Predict(B,a),\tilde o_s)),\quad V_h=\max_{legal\ a}Q_h,
\]

V0=sum omega_b 2tanh(value_PsiV(z_b,g))，gamma=.99。r−c∈[-2,1]与上述归一效用使折扣回报落在[-2,1]，critic[-2,2]可表达，原始预算不因归一缩小。终态样本后续V=0。采样先按omega抽branch，再按该branch的传感/结果分布抽样，未返回模态不采样。TNormal用FP64开区间Uniform，数值取u∈[2^-24,1−2^-24]再逆CDF时，每标量删尾质量≤2·2^-24，累积/条件化影响另计。

Assimilate_sim包含一次likelihood、R轮纠正、模拟FP1及写后读取；仅clone演化，真实E/F/journal不变，不结算真实reward。OBSERVE包含看完再选择，而非开环平均。

内部维护采用独立类型分支：MLP_(2d+8→3)([z;g;mode])预测新loss下降/旧loss增量/成本。r_dev_hat=tanh(Delta_Lnew−max(0,Delta_Lold)/.02)，Q_internal=(1−gamma)(r_dev_hat−c)+gamma V0。执行后P_R实际测r_dev=tanh((Lnew_before−Lnew_after)−max(0,Lold_after−Lold_before)/.02)，另记development reward，不进外部三类reward likelihood。无合法probe/依赖/预算则mask，不走物理transition，不用预测收益结算奖励。

u_Psi(B,a)=sum_b omega_b MLP_(2d+15→1)([z_b;g;a])；它混合的是标量效用，不是把物体位置先平均。行为策略pi_Psi=masked_softmax((sg Q_plan+u_Psi(B,a))/T)，T=1；实际按pi采样并保存行为概率/RNG/revision。评估greedy另报。不能对planner硬argmax冒用REINFORCE。真实done后不再提交外部ACT，进入settled_terminal/合法维护；已执行结果未知时保持pending，不重发。

## 9. 训练与信用

费用归一的CPU/wall单位各1秒、动作单位为1单位冲量、保持风险单位为.02旧loss变化；未知风险用missing标志并保守按1估算。真实原始费用仍保留。环境合同必须提供真实完成/失败/期限终态，不能让未完成任务零成本无限WAIT最优；若环境根本不给行动/观察价值，则该环境不足以签主动性。

\[
L_Theta=L_{next\ sensor}+.1L_{posterior\ recon}+.1L_{masked\ modality}+.5L_{reward}+.1L_{done}+L_{text\ task}.
\]

next_sensor精确定义为−LSE_b(log omega_b＋所有实际sensor有效标量loglik之和)/N_total，不先每模态平均或逐标量混合；每模态诊断另报。posterior recon同一归一规则单列，不拿它重算§5证据。reward/done各用归一mixture NLL，不用weighted branch CE。masked_modality遮掉实际已观察的模态，以剩余推断预测被遮部分的NLL/有效标量数，仅≥2模态时生效。task为含EOS的真实目标字节CE。目标/valid不为可学输入，运行写入编码保留梯度；EMA辅助若加入，只该目标stopgrad且不叫raw likelihood。

一个训练窗必须覆盖编码→绑定→F写→写后query读取→真实后继/任务loss；Θ在完整窗结束后更新。禁止teacher-forced目标提交到真实历史，跨进程只恢复值，不宣称无限BPTT。物体位置/质量/分割真值仅评分，不进首版训练输入或辅助target。

在线Ψ拆为Psi_pi/Psi_V，包含policy/value专属MLP，状态解释参数固定。保存行为packet（belief/goal/revision/π/RNG/动作），只允许一个pending，pending期间Ψ冻结。决策时q_j=clip_norm(条件偏导grad_Psi_pi log pi,1)，E_j=E_carry+q_j并存receipt。实际结果到达后u_j=(1−gamma^d_j)(r_j−c_j)，delta=u_j+gamma^d_j(1−terminal)sg V(next,旧goal)−V(old,旧goal)，只对该E_j结算，再置E_carry=(gamma lambda)^d_j E_j。新/取消goal终止旧价值问题并清该goal的trace carry，不能用新goal结算旧动作；未知真实结果不伪造reward，仍按pending/执行回执管理，不因取消重发动作。

critic走Psi_V的TD半梯度；actor只以delta E_j和.01熵项更新Psi_pi。Q_plan依赖critic但在actor中sg，不能称整个Ψ实际行为的联合score梯度；q的动作相关裁剪也明确是近似、非一般无偏REINFORCE。lambda=.95、d_j≥1、||q_j||≤1给固定维||E_j||≤1/[1−gamma lambda]。学习率1e-4、Ψ范数投影10。未知/晚结果不套当前trace，receipt只结算一次。Θ走真实观测/任务/结果目标，不谎称环境可微。

维护预测器的Theta目标另加.1L_dev：对新loss改善、旧损伤、归一成本的真实开发Outcome做逐项Huber(delta=1)回归，字段未测不造target。物理reward三类与开发reward分域，维护不能吃到终态后虚构物理奖励。

数值核验分开两个目标：Theta的真实观测/任务梯度，以及Psi在固定belief/行为数据下的条件梯度。FD扰动Theta时不能重算被sg冻结的policy/value输入后，再将差分与半梯度作比较；那是另一个目标。代理/条件梯度与联合真梯度的范围必须写进回执。

## 10. 可证明的性质与误差

H/object h有界、F球约束、状态p/v/s数值域、soft分配和字节/branch混合归一；全missing Corr精确返回prior。固定门/key且无value写入的FP1非扩张。上述都不是整个耦合Jacobian收缩、对象识别成功或物理正确性证明。

K/M有限、确定性分支、对角尺度、有限R、codec、运动子步、规划样本/节点和FP求根都是显式近似。近端目标差可给距离证书；完整误差传播还依赖各映射敏感性和决策margin。相同合法观测对应不同物理真值时不保证立刻辨识，有限假设也可能漏掉真相。模型应报告校准范围、失败/求证与预算停止，不能用多头方差自行认证真理。

## 11. 一次事件的确定顺序与恢复

1. 校验来源/时间/重复id/资源；读取真正已执行动作及当时goal revision的receipt。
1a. lifecycle_enabled时执行§3.1假设管理（出生/退役/合并）：输入为上一事件的绑定/null统计与当前槽状态，先记journal后改状态；生涯首事件无上一统计则跳过。
2. 仅从旧状态、旧已知goal和动作/dt生成prior，保存其decoder参数与key视图。刚到达的新command不回填过去运动条件。
3. 对真正到达且valid的观测计算predictive likelihood及branch权重。
4. 吸收已到达的新command更新g，编码局部tokens，R轮创新校正；不再重复物理时间或likelihood。Outcome仍绑定原动作goal，不被新goal重解释。
5. 合法真实反馈/观测触发一次批量FP1；写后按query再读并更新共同状态。
6. 输出文本/规划/采样动作；动作先记proposed/authorized/executing，真实回执后才算executed/settled。
7. 原始E、state、RNG、pending/journal watermark原子提交。重复包/Outcome不重复F写/奖励。

checkpoint包含Θ/Ψ、全部K分支/weights/H/F/objects、源媒体引用和编码revision、动作与反馈pending、optimizer/eligibility、RNG和资源账。恢复不得重新执行已发生的动作。感知/状态坐标参数升级采用shadow完整合法因果重放，预算不足不提交；仅改policy/value专属读出网络可按依赖清单免状态重建。

predictive packet不可变绑定prior、旧goal、a_exec、dt、对应decoder参数与key view。新goal到达只能影响随后的推断/选择，不能通过调用者传参覆盖旧预测条件。恢复同时保存packet；其中Tensor训练窗可保留图，持久化只存值。

Θ在packet预测到对应观测吸收之间固定；需要升级编码/世界参数时先结清pending并做状态迁移。参考的缓存control test限定同一Θ；不能用重算的新参数likelihood签成旧预测。语言每接受一个非EOS字节，以e_emit=B(text_input(embed(byte)))和零物理action描述做一次H微步；EOS不再查256类原字节embedding，世界omega/p/v/s/F不更新。UTF-8 FSM还屏蔽剩余字节预算不足的多字节起始符；至多64数据字节＋一次EOS预测，若下一模型选择仍非EOS则truncated，不伪装主动终止。

## 12. 重放、慢学习与神经生长的闭环

REPLAY从许可E用softmax(MLP_(3d→1)([历史证据表示;g;D_nu nu]))选项，每源≤4次；只用历史receipt配对的key/value做FP1关联复习，编码revision不兼容先在历史shadow重算，否则mask。它不走当前sensor likelihood/Corr，不改当前p/v/s/l/branch logweights，不重复物理dt/reward；F变化可通过未来query影响推断。重建当前belief必须完整历史shadow因果重放，不能单帧回填旧位置。

CONSOLIDATE用已释放经验求原主稿§10.2保持QP，信赖半径1e-3、梯度步标度1e-4；P_A独立验实际功能，失败回退参数/状态，不清journal/费用。

GROW沿用a(x)=Wx+P tanh(Rx)，R_new=0、P_new=B。最近32个合法样本的C=sum(delta x^T)做SVD，B=.1U_r，每群r≤4且r≤rank_num(C)、总m≤512。相同奇异子空间以其投影矩阵逐个投影标准基e_1..e_d，再Gram–Schmidt选首个范数>1e-10的方向，首个非零坐标取正；不把裸符号作准入。rank门为max(1e-8,1e-6 sigma1,tau_noise)，tau_noise取128次源分组残差置换的.99分位，属校准而非任意依赖数据上的检验定理。逐单元||B_j^TC||>1e-8并在独立后继窗检查投影对齐。

真实梯度/ST代理分名；成熟8次合法受约束更新后，P_A要求新loss改善≥.001、旧loss增≤.02、lesion移除至少10%新收益且扣实际成本后净收益为正。P_R/P_A来源独立、各自尝试≤4，P_C不选方向/阈值；缺校准/合法数据/预算则mask。这些是首profile研究准入值，不是性能保证，改值须新版本。

函数中性仅指出生瞬间既有主图（d、消息支持、归一维度不变），不指物理费用相同。父W自由时启动切向可由delta_W=B delta_R复现，启动不签新增能力；不无条件投影完整父切空间。对象track数/世界假设数扩容也不冒称这种神经出生的中性证明；超M/K范围先报容量/不确定性，另分版本扩容和保持。

## 13. 实现覆盖矩阵与验证出口

后继实现逐行绑定：raw codecs/encoders、时序门、prior/H/object、规范化decoder、branch filter、soft binding/corrector、FP1 active/inactive backward、写后query、byte head、belief planner/behavior策略、真实Outcome信用、生命周期/恢复/生长。每行列参数、状态、目标、零类和负对照，不能仅报总norm。

本轮[独立张量参考](../../scripts/training/verify_taiji_mgw1_math.py)与[20组回执](../../reports/taiji_mgw1_math_20261106.json)覆盖主推断/学习核及小候选规划，不代签生产候选、真实对象发现或物理能力。[核验范围与未分辨项](VALIDATION-MGW1_REFERENCE_20261106.md)逐项列明；未执行的64字节生成、生产event controller/回执、慢参数事务和生长准入不能因公式完整便冒充工程已交付。规格起草2026-11-05，跨日核验更新至2026-11-06。

2026-11-06补编的槽生命周期（§3.1）、ESS诊断（§5.1）与64字节发射循环经[补编预审](../../reports/taiji_mgw1_lifecycle_supplement_20261106.json)代数核验：出生对未选中槽/H/F/omega逐位中性、质心只依赖null质量与坐标、同event幂等拒绝、E_storm风暴门、退役槽竞争分数压低≈−7.3 nats、合并g加权与第三槽中性、每事件≤1次、ESS已知值、200次随机生成的UTF-8全程合法（含truncated在字符边界）、eta按分支发射概率更新且世界omega逐位不变、剩余预算不足的多字节起始符被FSM屏蔽。预审同时固定发射口径：**FSM屏蔽在各分支内重归一后再按eta混合；字节预算耗尽后的最后一次预测不强制EOS，模型原始选择非EOS则记truncated，不伪装主动终止**。该预审不代签生产event controller与真实发射工程。

原语采用已有注意力/对象表示、递归动力学、近端回归、有限belief规划及函数中性生长的组合，不主张首次发明。参照：[Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)、[DreamerV3](https://www.nature.com/articles/s41586-025-08744-2)、[V-JEPA 2](https://arxiv.org/abs/2506.09985)；其前提/规模/结果不迁移为MGW1能力。
