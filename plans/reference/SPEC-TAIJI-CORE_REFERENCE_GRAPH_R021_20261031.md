# R021：正式参考图的工程接口冻结

> 文档状态：[合同] — 只冻结09.2.1的计算接口与后继工程边界，不是训练/定价/能力批准。调度只见[MAINLINE](../MAINLINE.md)。

日期2026-10-31。底座为[算法主稿](TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)§4–8/13，生长候选为§24。先冻结本件，再实现`taiji/living_core.py`。R021只交正式前向、显式参数/状态和方程覆盖矩阵；09.2.2信用/实际更新、09.2.3出生/编码迁移、09.2.4完整主体恢复分别验收，不能提前补签。公式变更须新接口修订，不让两个实现者自行补含义。

## 1. 范围、形状与时序

默认n=4、d=128、m=256、低秩适配a=16、每群L=2个k=v=32快头、共同读出J=2个宽d的头。工程夹具用n=2/d=8/m=12/a=2/L=2/k=v=4/J=2，拓扑同构。d须为偶数；R021最多4群、512输入字节、512来源字节、128目标字节，不静默截断。这些是首版工程界，不是模型终身规模上限。

连续状态含subject_id/revision/step、H[n,d]、F[n,L,k,v]、nu_ema[8]、来源位置keys、累计coverage和位置先验。状态由调用者跨步传入/返回，不在step中新建主体。当前不含完整optimizer、事件journal或持久恢复，故不称完整生命主体。所有群先读旧H/F，再同步更新H，再预测/表达，最后按许可写F；不原地修改父状态。

## 2. 六项冻结接口

### I1：字节patch到事件e

raw字节id为0..255；257行embedding宽64，第256行为开始/结束协议符的输入embedding，不是语义词表。编码：embedding→Linear(64,d)→三层因果残差卷积，kernel=3、dilation=1/2/4，逐层u_next=u+tanh(Conv(left_pad(u)))。每层只左补零，不读未来位置。

逐字节a_p=u_p+sin/cos(p,d)。patch按连续8字节划分，patch_j=实际字节u的均值，最后不足8字节只除实际长度；e_j=Linear(d,d)(patch_j)。**每个patch对应一次微步，不把多patch默默平均成一次事件**。无新事件时e=0，不调用带bias投影伪造事件。goal用其全部实际字节u的均值经独立Linear(d,d)编码；空goal为0。每次来源读取用当前编码器重新编码原文，不消费序列化latent替身。

### I2：nu八读数、缺失与D_nu

顺序固定：资源余额比例、CPU代价、已到达预测误差、已校准不确定性、未结目标/结果计数、净学习进展、记忆干扰、梯度冲突。调用者只提供当前可见真实读数raw[8]和known[8]；缺读数不得填成已知0。

资源/未结数两项直接使用；其余known项按0.95旧EMA+0.05raw更新，unknown项保留EMA值但送入网络的对应值为0。固定工程标度s=[1,1,1,1,16,1,1,1]，CPU以秒计，其他量按上述原单位。v_j=known_j*tanh(ema_j/s_j)，missing_j=1-known_j。D_nu=Linear(16,d)([v_0..v_7;missing_0..missing_7])，保留8个读数加8个缺失标志，不能把缺失与真实0混同。标度可在后继版本重定价，当前不得按结果改变。

### I3：情景打分、null与value

记录只含source_id、原文字节、owner/scope、observed_at、version、real标志；不含gold实体/属性/time-intent/选中ID。source_id指不可复用的原文版本身份，换内容/版本须换id；R021调用者负责此输入条件，跨生命周期的存储身份验证归09.2.4。合法性仅为同owner/scope、observed_at≤当前事件钟、real；来源按输入顺序、字节按原位置展开。非法原文在编码前排除，不按目标语义筛选。

每位置T=[tanh(age/64),tanh(observed_at/64),tanh(version/16),real]，a_p为I1上下文/位置特征。score_i,p=v_r^T tanh(W_2 tanh(W_a a_p+W_h N(h_i)+W_g g+W_T T+b_1)+b_2)+b_3。两层隐藏宽d；标量score后softmax，包含一个始终合法的可学null。普通value=V_a a_p+V_T T+b_v；null有自己的可学key特征和值。r_i为权重value和。无记录时权重全在null；不以零向量替代该可学候选。

### I4：群间Q/K/V

每群Q_i/K_i/V_i均d×d、无bias；共享目标Q_g为d×d、无bias。q_i=Q_i h_i+Q_g g，k_j=K_j h_j，v_j=V_j h_j；alpha_ij=softmax_j(q_i^T k_j/sqrt(d))，m_i=sum_j alpha_ij v_j。完整旧群支持含自身；使用旧H而非更新中的H。这里不额外单位化，也不以N(H)点积替代正式矩阵。

快读取独立：u_i=[N(h_i);e;g;D_nu]宽4d，Q^F_i,l为Linear(4d,k)，q^F=B(Q^F u)，每头F_i,l^T q^F，按头0..L-1拼接经Linear(Lv,d)得到f_i。

### I5：查询共同读出与拼接

xi=Linear(2d+64+7,d)([g;embedding(previous_byte);one_hot(mode,7);D_nu])；没有当前待预测字节。每输出头a=0..J-1有Q_a[d,d]与各群K_a,i/V_a,i[d,d]，均无bias。对**更新后H**做缩放点积softmax；head_a=sum_i omega_a,i V_a,i h_i。严格按[head_0;...;head_(J-1)]拼接，经W_z[d,Jd]（无bias）得到z。

r_t按各输出头omega的算术均值对旧r_i汇聚。P_gen=softmax(Linear(d+64,257)([z;previous_embedding]))，第256类为EOS，仅生成分支可给EOS质量。g_mix=sigmoid(Linear(2d+64,1)([z;r_t;previous_embedding]))。无合法复制位置时强制g_mix=1。各头、编码、读取和表达参数均注册可学；正常零导数/禁写分支在09.2.2逐项检验，不以requires_grad代签有效信用。

### I6：逐步位置分布与coverage递推

复制基础score_p=v_s^T tanh(W_z z+W_a a_p+W_T T_p+W_P PE_p+w_c coverage_p+W_g g+b_s)，softmax覆盖全部合法原文位置；同字节不同位置独立计。连续门lambda=sigmoid(Linear(d+64,1)([z;previous_embedding]))。若已有的shift_prior有正质量，w=(1-lambda)*base_w+lambda*shift_prior；否则w=base_w，禁止除零或伪造位置。

先输出P(y)=g_mix P_gen+(1-g_mix) scatter_byte(w)，再消费实际输出字节b。令m_p=w_p*1[raw_p=b]；mass>0则posterior=m/mass，coverage_next=coverage+posterior，shift_prior_next由posterior仅向**同source的下一位置**平移后归一；若无后继质量，先验归零。mass=0或EOS则不增加coverage、先验归零。它是位置启发先验，不证明字节实际来自复制分支；不得跨记录尾首平移。

来源keys变化时只按稳定(source_id,position)重排既有coverage/先验，新位置补0；来源id不得重复。teacher forcing在输出P(y_j|y_<j)之后才消费y_j；推理消费自己的输出。两者调用同一个step/consume_output，不预先把目标字节写进coverage。R021检验分布与时序，不主张UTF-8自由生成或中文能力。

## 3. 其他前向算子的唯一补全

N(h)=h/sqrt(1+mean(h²))，B(x)=x/sqrt(1+sum(x²))。动力学x=[N(h);e;m;f;r;g;D_nu]，a=(W+UV^T)x+P tanh(Rx)，h_next=(1-ell)h+ell*tanh(a)，ell=1-exp(-1/tau)，tau=0.5+127.5 sigmoid(rho)。初值1/4/16/64按群索引取；U=0、V随机非零。固定d通信宽不变。

每群写门MLP为Linear(4d+9,d)→tanh→Linear(d,L*(3k+2v+2))，输入[N(h_next);e;g;D_nu;past_error;error_missing;mode_one_hot(7)]。按头拆为raw_key[k]、raw_value[v]、decay[k]、erase[k]、write[v]、erase_strength[1]、chi_logit[1]；key=B(raw_key)、value=B(tanh(raw_value))，其余sigmoid，eta=16 sigmoid(strength)。初始chi bias=-2、decay bias=4；这是工程初值，非已测最优。

采用主稿近端擦除与整体chi插值，Frobenius上限默认16；硬许可[n,L]为False直接保留旧F，包含不投影。预测/表达产生后才提交快写。三个世界头分别对[z;action_vector]做Linear(2d,d)→tanh→Linear(d,d+1)，返回后继表示预测和reward预测；无动作时action_vector=0。EMA目标、世界损失、活动策略、meta/consolidation、出生和恢复当前未实现，覆盖矩阵必须明列后继阶段，不能因前向头存在签闭环。

## 4. 软门与覆盖出生的后继工程冻结

诊断/覆盖分支采用g=softmax(logits/T)，T=1、K≤4；非负且每样本和为1。非法支路先mask，至少有一条合法支路；全mask为不可判/停止，不静默造均匀门。实际新支路和诊断必须使用同一个g，不能对任意门只在诊断时重新归一。硬分区另作对照；软门不套sum rank≤N的上界。

R_max=8，r_c≤rank_num(C_c)，候选枚举上界(8+1)^4=6561；最多检查6561个状态、1 CPU秒/2墙钟秒，超出任一值停止该次提案，保留父状态与费用，不自动提高预算。R021只冻结该合同，分配/出生实现属于09.2.3。它不是永久生长规模上限；扩展须新工程合同。

数值秩tau=max(1e-8,1e-5*sigma_1,tau_noise)。在固定父端点、合法E_fit中分离方向拟合/检查窗口，各至少32个非零贡献位置；128次固定seed的残差—输入对应关系置换，保持门/整体平均/维度一致，tau_noise取每支路最大奇异值分布0.99分位（higher次序统计）。这只是工程噪声参照，不声称统计显著性或可交换性保证；校准缺失/样本不足即未标定，不准出生。

绝对下限：||C_c||_F>1e-6，所选sigma_j>tau，逐单元||B_c,j^T C_c||>1e-8；检查窗中投影范数>1e-8且与拟合窗余弦≥0.1。均为当前有尺度约定的工程准入，非任务收益门；若全不满足则不出生并记负/不可判，不用相对rho=1选取极小能量。拟合/检查共用delta=整体按目标字节平均loss导数，禁止重复平均。c_c按父尺度另冻结前不可提交出生。

共同尺度C=sum C_c、kappa分母0标未定义；不得靠epsilon产生假读数。门未形成、噪声未标定、数值不分辨、实际总体信用为0、分配/迁移预算超限都保留父版本并记录，不以L_start或总norm放行。真实任务效应、保持、资源总账及P_R/P_A/P_C数值门仍须09.6冻结，不从这些工程阈值推导能力。

## 5. 当前准入与证据链

09.2.1只运行合成原文/张量的前向与边界检查，无backward/optimizer、无语料分区/训练、无产品默认替换。报告绑定本合同/源码/测试SHA，并给出方程→源码方法及行号→参数/状态→检查ID矩阵；全部注册参数与未实现责任分开登记。09.2.1通过不代表09.2.2信用有效、09.2.3生长有效或09.2.4恢复完成。

历史独立评估对象固定为64fdf8576的v1.1/E1；入库身份、原始字节与可读版本映射另记admission manifest。它既不审当前v1.2，也不替代R021验收。只有09.2.1–09.2.4全部完成，才可进入定价/有界比较的合同决策，不自动开训。
