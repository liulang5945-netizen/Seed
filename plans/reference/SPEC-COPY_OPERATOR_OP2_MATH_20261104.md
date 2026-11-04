# OP2复制算子的概率定义、推导与验证合同

> 文档状态 [数学候选·独立参考验证]。日期2026-11-04。OP2是一套record-first概率转移核提案。复推起始基线N17.5已按原冻结算子/语义身份负结案，当前N18.1也已负结案并进入归因；本文不替换这些候选。手工推导、独立有理数枚举及数值梯度检查共同限定本轮结论，任务收益、完整候选及部署采用另验。当前队首只由MAINLINE维护。

## 1. 本轮补正与范围

上一轮的两处结论需要补前提。

1. 将softmax(alpha*beta/T)改为log-beta温度化，只能去掉fresh pointer内部的alpha影响。全局cursor混合后再条件化记录，会重新引入旧alpha。
2. 复制责任乘CE梯度只描述冻结G/gamma等上下文的直接发射通路。Q还影响content、H_next、G与gamma，总梯度需要完整链式项。

现核全局continue/restart后条件化与OP2记录先选后continue/restart都能定义合法模型。OP2消除小记录质量除法及其fallback，符合本候选要求的记录/位置分工，但改变转移语义，不能称为原核的代数等价优化，也不保证行为更好。

### 1.1 算子与参考实现的版本差异

| 层 | 审查基线的既有语义 | OP2候选 |
|---|---|---|
| 源/转移 | softmax(alpha beta/T)及logit几何池 | 源logits温度化及行概率边缘化 |
| cursor与记录 | global续读/重启后按record质量归一 | record-first单位条件核，lambda_i m_i＋(1−lambda_i M_i)p_i |
| 低质量 | 小record质量触发beta回退 | 数学单位核，无旧alpha小质量除法；数值norm只校正单位质量舍入 |
| 后验 | helper外传gamma/P与部分legacy字段 | 同一不可变packet内的最终Q/G/gamma/byte映射 |
| 数值 | 概率域/epsilon目标按原实现 | log-domain reference及log cursor/补事件，按规定容差修复 |
| 调用与能力 | N17.5按原身份提供已结案行为证据 | 独立数学参考，完整主体与任务比较待新版本实施 |

## 2. 数学对象与合法域

合法非空记录i=1..N，长度L_i≥1，原文字节b_ij∈{0,..,255}，EOS=256。N可以为0，空/全mask记录在路由前排除，null始终合法。记录身份为(source_id,version)，版本内原文不可静默改变。下文可推广到任意有限发射字母表和一个不在源内的EOS。

| 对象 | 形状及语义 | 前提 |
|---|---|---|
| a=(a_1,..,a_N,a_0) | 本步记录控制概率，0表示null | 非负、总和1，soft或声明的hard数值 |
| u_i | 源锚点logits，L_i维 | 固定mask后至少一个合法源，未mask值有限，温度缩放后的gap需在数值范围内 |
| Z_i | 源位置到目标位置logits，L_i×L_i | 每个有正源权重的行至少一个合法目标，屏蔽值为−∞；零权重行不归一化 |
| B_i | 目标位置bias，L_i维 | 与实际正常读取同一份声明参数 |
| T_s,T_p,T_r | 源、目标、记录温度 | 有限且严格正，各自定义与持久化 |
| m_i | 上一复制责任的合法后继次概率，L_i维 | m≥0、sum m≤1；普通shift下每记录首位m_i0=0 |
| log_m | 零质量为−∞的对数cursor | 递推时保留小于浮点exp范围的责任；概率m仅为视图 |
| log_noncontinuation_i | log(1−M_i)的冗余补集表示 | 从后验补事件直接求和，和log_m一起保存/对齐，避免近1的减法丢掉罕见重启 |
| lambda_i | 续读概率 | [0,1]，可先共用一个lambda标量 |
| G | 257类生成分布 | 非负且总和1，源不可复制EOS |
| gamma | 本步实际生成门 | [0,1]，前向与责任更新必须使用同一个effective值 |

上述m条件是可达域的必要约束。正常接受上一字节y后还满足m_i,j>0⇒源字节b_i,j−1=y；本轮梯度夹具遵守这一支持条件，验证器也检查每次accept的输出支持。数学核可定义在更宽的次概率先验上，不把每个满足首位0的向量称为已证实历史可达。

正常状态域与诊断强制域分开。强制m在记录首位非零可以做oracle控制，不能拿它证明普通轨迹可达。数学上不将m归一化到1，它的小质量包含生成责任、末位无后继等置信信息。OP2接口明确lambda为有效概率；若调用者使用logit，应显式转换并另命名，不能把裸lambda_override同时解释为两种量。

## 3. 先审清旧核的语义

令p_i为与alpha无关的记录内fresh分布，M_i=sum_j m_ij，M=sum_i M_i，c=1−lambda M。旧未锁读取为

\[
Q^{old}_{ij}=\lambda m_{ij}+c\alpha_i p_{ij},\quad
q^{old}_0=c\alpha_0.
\]

合法域上总质量1。旧算子的记录级小质量fallback仍按原版本解释。锁定前再取条件分布得到

\[
\eta_i^{old}=\frac{\lambda m_i+c\alpha_ip_i}{D_i},\quad
D_i=\lambda M_i+c\alpha_i.
\]

D_i>0、M_i>0时，它是nu_i=m_i/M_i与p_i的混合，continuation权重为lambda M_i/D_i。保持其他变量不变，有

\[
\frac{\partial\eta_i^{old}}{\partial\alpha_i}
=\frac{c\lambda M_i}{D_i^2}(p_i-\nu_i).
\]

所以即使p_i已去alpha依赖，最终eta一般仍受alpha影响。可达反例取lambda=4/5，m_i=(0,1/2)，p_i=(99/100,1/100)。alpha=1/10时eta=(297/2300,2003/2300)，alpha=9/10时eta=(2673/4700,2027/4700)，位置argmax翻转。该m可由上一责任(1/2,0)正常shift产生。

旧核可以解释成先根据上一复制状态决定全局续读，否则按alpha重启，再按记录重新分配边缘质量。alpha影响续读赔率是该语义的结果。若当前设计想让新记录控制a独立负责记录选择，就必须明确选择另一种转移语义。

## 4. fresh分布的概率边缘化

OP2定义

\[
s_i=\operatorname{maskedSoftmax}(u_i/T_s),\qquad
A_i[r,j]=\operatorname{maskedSoftmax}_j((Z_i[r,j]+B_i[j])/T_p),
\]
\[
p_i[j]=\sum_r s_i[r]A_i[r,j].
\]

s是源不确定性，A是每源条件转移，所以p是其边缘分布，非负且总和1。源权重从logits构造，等价于softmax(log beta/T_s)，避免先将beta算到下溢再取log。记录质量alpha不进入这一条件分布。

当前softmax(sum_r s_r Z_r/T_p)是另一合法的几何池，其值正比于各行概率的加权乘积，不能与上述算术边缘化混称。取s=(.9,.1)、两行概率(.6,.4)与(.01,.99)，算术混合第一位为.541，几何池约.476369，argmax相反。若两行mask为(1,0)、(0,1)，算术混合可为(.5,.5)，几何池则无共同支持，归一化未定义。

OP2选概率边缘化作为规范reference，几何池保留为命名清楚的独立候选。行logits加常数、源行重排及目标坐标连同数据重排具有对应不变性。位置坐标重排是算子重标记，真实字节顺序变化不是同一个不变性主张。

## 5. record-first续读与输出分布

定义

\[
\boxed{\eta_{ij}=\lambda_i m_{ij}+(1-\lambda_iM_i)p_{ij}},\qquad
\boxed{Q_{ij}=a_i\eta_{ij},\quad q_0=a_0}.
\]

因为lambda_i M_i≤1，eta非负且sum_j eta_ij=lambda_i M_i+1−lambda_i M_i=1。因此Q及q_0非负，总质量sum_i a_i+a_0=1。M_i=0时eta_i=p_i，无小记录质量除法和1e−12分支。上例中OP2始终给eta=(297/500,203/500)。

该分工的条件不变性只在固定a、lambda_i、p_i、m_i等输入时成立。真实网络中加入其他记录可能改变H、查询、lambda或a，不能把这个局部命题扩大为整个模型不受上下文影响。

令C_b=sum_ij Q_ij 1[b_ij=b]，A_G=gamma+(1−gamma)q_0，则

\[
P_b=A_GG_b+(1-\gamma)C_b.
\]

总和A_G+(1−gamma)(1−q_0)=1。null或无记录时a_0=1，P=G。gamma=1同样P=G。gamma地板不构成greedy序列正确率的乘积上限。

### 5.1 记录控制与继承续读不能无条件兼得

要求最终记录质量恰为任意给定a_i，同时原样保留旧续读分量lambda_i m_i，并让其余fresh质量逐位置非负，即

\[
Q_{ij}=\lambda_i m_{ij}+R_{ij},\quad R_{ij}\ge0,\quad \sum_jQ_{ij}=a_i.
\]

对j求和得到sum_j R_ij=a_i−lambda_i M_i。因此在剩余质量有合法归一fresh分布p_i的前提下，可行的必要且充分条件为

\[
\boxed{a_i\ge\lambda_iM_i\quad\text{对每条记录成立}}.
\]

必要性来自R非负；充分性取R_i=(a_i−lambda_iM_i)p_i。空记录必须先mask掉。固定null质量时，sum_i lambda_iM_i≤1−a_0只保证存在某个可行a，不保证任意指定a可行。例如a_i=.1、lambda_i=.8、M_i=.5，原续读质量.4已经超过整条记录预算.1，任何非负重启项都不能补救。

OP2明确选择记录控制优先，保留的续读分量为a_i lambda_i m_i，声明的continue隐变量先验总质量为

\[
K_{\rm OP2}=\sum_i a_i\lambda_iM_i.
\]

共同lambda、M>0时，它相对未锁定global核lambda M的比值是sum_i (M_i/M)a_i。无null、N条记录等概率时比值为1/N；有null且记录均分时为(1−a_0)/N。这是固定算子输入下的语义代价，不能拿lambda M冒充已做记录条件化的legacy locked核的续读质量。硬a选择同记录时不再受均匀路由的因子，选错记录时则直接抑制该记录的继承续读。

K并非可观测唯一的续写归因。如果m_i=M_i p_i，则eta_i=p_i，与lambda_i完全无关，但K仍随lambda_i变化。例如p=(0,1)、m=(0,.5)、a_i=1时，lambda=.2与.8给出相同Q，却分别有K=.1与.4。必须将声明分量、输出分布、干预效果和任务收益分开。OP2不能凭单位归一化就宣称解决了跨步承载或任务归因。

## 6. 受控潜变量模型及精确责任

给定已经观察的历史，a、p、lambda、G、gamma和确定性H是已知控制量。先按gamma选择直接生成；否则按a选择null/记录。null生成G。若选记录i，上一步隐变量为同记录复制位置且有合法后继时，以lambda_i继续，否则在i内按p_i重启。生成发射G，复制确定性发射源字节。a的策略可依赖已观察历史，但在给定历史时不能再偷看上一未观测latent。

令Z_t取gen或一个源位置。上述转移使记录i的预测位置恰为第5节eta。对P_y>0的接受字节y，

\[
r_{ij}=\frac{(1-\gamma)Q_{ij}1[b_{ij}=y]}{P_y},\qquad
r_G=\frac{A_GG_y}{P_y},\qquad r_G+\sum r_{ij}=1.
\]

当前a若为观察历史的确定性hard决策，可条件于该控制。若用soft a，则按上述混合解释。G/H使用对历史确定的预测Q/content，不得暗依赖未观察的latent样本或未来字节。该明确模型下可作递归滤波，不能泛称任意Markov模型或任意神经更新都有同一后验。

后继cursor为同记录shift(r)。它非负、sum m_next≤sum r≤1，每记录首位为0、长度1记录m恒0，末位责任丢弃并在下一步重启。EOS无源复制责任，接受EOS时m_next=0。非法token拒绝，P_y=0时Bayes条件未定义，工程reset必须单独标记。accept接口仅接收同一份不可变forward packet和接受字节，effective gamma、最终Q/G、原文byte映射在packet内固定，避免重新传一份不匹配的门。

coverage_next=coverage+r是在线责任累计sum_s r_(s|s)，未来观测可能修订过去隐变量的平滑后验，因此它不等于一般的E[全历史复制次数|当前完整历史]。记录删除/撤权只保留合法后继质量，其他质量转入重启，不能把剩余m重新归一为1。

## 7. 梯度与数值目标

完整微分为

\[
dP_y=A_GdG_y+(1-\gamma)dC_y+(1-\gamma)G_y dq_0+
[(1-q_0)G_y-C_y]d\gamma.
\]

直接copy emission的责任缩放公式需要冻结其他路径。实际Q经content→H_next→G和gamma，必须加这些链式项。即使目标字节不在源内、直接复制责任为0，pointer仍可通过内容影响生成而获得非零总梯度。

对独立的算子输入、精确L=−log P_y，可写

\[
\partial_\gamma L=[C_y-(1-a_0)G_y]/P_y,
\]
\[
\partial_{\lambda_i}L=-\frac{(1-\gamma)a_i}{P_y}
\left[\sum_jm_{ij}1[b_{ij}=y]-M_i\sum_jp_{ij}1[b_{ij}=y]\right].
\]

其余解析梯度通过softmax Jacobian传递。令v_j=−(1−gamma)a_i(1−lambda_iM_i)1[b_ij=y]/P_y，g_r=sum_j A_rj v_j，则

\[
\partial_{u_r}L=s_r(g_r-\sum_l s_lg_l)/T_s,
\quad
\partial_{Z_{rj}}L=s_rA_{rj}(v_j-g_r)/T_p.
\]

bias梯度为Z各行对应列之和。记录与生成logits也通过各自softmax Jacobian。这里lambda/gamma/G等视为独立输入，神经参数共享时需要全部链式项，不能用这组局部偏导签完整主体。

当前实现L_epsilon=−log(P_y+1e−30)，其梯度相对精确NLL有P_y/(P_y+epsilon)缩放。状态责任仍用原P，二者不能混写。OP2 reference从log域计算似然与责任。责任采用各latent发射likelihood的max-shift log-softmax，避免直接相减两个巨大负log值的消减误差；即使exp(log P_y)下溢为0也保留有限log似然。真正log P_y=−∞时拒绝条件化。

### 7.1 数值闭包及梯度验证边界

严格概率域仍可能在浮点后验得到copy总量1.0000000000000002。只放宽输入检查会使lambda=1时restart系数变负。因此参考实现只在舍入容差内对过量copy质量作微小收缩，误差超128·eps·位置数立即拒绝；修正加入8·eps·位置数的余量并记账。合法的低copy质量从不归一化到1。参考API先验证合法域，普通递推由accept输出的log_m与其概率视图继续，视图不一致拒绝。

为保留极罕见重启，数值状态还存log_noncontinuation_i。令U_i为记录i中有合法后继的旧复制位置集合，则D_i=r_G＋其他记录全部r＋本记录末位r=1−M_i,next；用完整后验补事件的log-sum-exp计算D_i，而非用1−exp(log M_i)相减。重启权重按(1−lambda_i)＋lambda_i D_i组合。copy质量舍入到1时，log D_i仍可为−1000，后一步重启log似然也能保留−1000。它是同一状态的数值冗余，不是额外学习机制。

修复copy舍入超量后，当前reference的m与修复前补事件表示可有O(eps)偏差，按位置数预算检查，不能声称有限精度下逐项严格互补。给定同一G/gamma/a/p，对同步更新补集的严格OP2核，copy-state收缩L1量delta引起本步发射L1误差≤2delta。若数值修复保留原补事件、再校正记录条件总质量，delta<1时可用2delta/(1−delta)的局部条件界，其他舍入误差另计；完整神经反馈或极罕见观测可放大误差，不能用这个局部界保证长序列精确。活跃修复边界的梯度不在本轮FD声明内。

概率域对照张量图在float32 logits=[0,−110,0]、float64 [0,−800,0]时出现inf/NaN，负例保留。新增log-domain张量图计算log-sum-exp似然与log后验，不先exp再除P；下一个cursor也保留log责任。例中exp(log m)=0但log m=−110仍能支撑后一步似然和有限信用。数值修复与不可达mask边界不是全域光滑函数，内点导数和边界前向闭包分开验。

本轮单步/三步FD覆盖固定初始m及固定温度下33个声明条件参数分量，包含record/source/transition/bias/G/lambda/gamma；另以单步解析式和标量FD检查3个初始cursor内点方向、3个温度方向；不覆盖支持/修复边界导数。温度偏导为NLL条件目标，若熵系数也依赖温度，还需加相应链式项。三步递归会通过后验→shift传播这些参数的信用。float32/64正常幅度概率和三步梯度均有对照，极端测试只保证列明的profile。真实梯度可以因边界/条件数很大而超过dtype范围，log-domain避免下溢并不提供任意梯度有界保证。

### 7.2 门、责任与信用饥饿的条件命题

冻结其他输入，记D_y=C_y−(1−a_0)G_y。精确NLL给出partial_gamma L=D_y/P_y。若gamma=sigmoid(v)，则

\[
\partial_v L=\gamma(1-\gamma)D_y/P_y.
\]

对仅通过C_y影响P_y的指针参数theta，有

\[
\partial_\theta L=-\frac{1-\gamma}{P_y}\partial_\theta C_y.
\]

固定G_y>0、C_y，且指针偏导有界时，v趋向正无穷使这两条直接信用趋零。更可量化地，若G_y≥g_min>0、norm(partial_theta C_y)≤B，则norm(partial_theta L)≤(1−gamma)B/(gamma g_min)。这是给定条件的上界，实际共享content/H/G/gamma和跨步总梯度不由它界定。

G_y>0不能省略。若G_y=0、C_y>0，gamma<1时partial_v L=gamma、partial_theta L=−partial_theta C_y/C_y，均不因gamma趋1而必然消失；这是边界或masked生成的反例，非普通有限logit全支持softmax的状态。有限门值也不足以判断饥饿。gamma=.92、a_0=0、G_y=.001、C_y=.5时，P_y=.04092，复制后验责任.04/.04092≈.977517，指针直接梯度系数.08/.04092≈1.955034。不得由N17/N18的大gamma单独归因，应结合受控干预、责任和实际梯度/更新。

若gamma=ell+(u−ell)sigmoid(v)，令s=sigmoid(v)，则partial_v L=(u−ell)s(1−s)D_y/P_y。u=1且G_y>0时保留上述极限；u<1时门logit信用仍可饱和，但指针系数一般保留。硬clamp的活跃区单独计零梯度。实际epsilon目标的两个偏导分母改成P_y+epsilon，其极小概率极限与精确NLL不同。

D_y=0只意味着这个观测的直接门信用为0；若整个C=(1−a_0)G，则冻结其他路径的完整发射分布对gamma不可辨识。这不签共享参数/递归链也为零。在正常有限logit softmax且G_EOS>0、a_0<1时，copy无EOS，整分布相等的情形不成立，单字节不辨识仍可能发生。类似地，冻结m/lambda时partial_eta/partial_u=(1−lambda_iM_i)partial_p/partial_u，lambda_iM_i=1使fresh支路的这一条件信用为零；不能用它代签共享神经参数的总梯度。

§5.1与本节为后续补推的条件命题，以下精确算术见证可独立复现，未冒充原11组数值回执的新增覆盖。

```python
from fractions import Fraction as F

a, lam, mass = F(1, 10), F(4, 5), F(1, 2)
assert a - lam * mass == -F(3, 10)  # preservation is infeasible
assert sum(F(1, 2) * lam * x for x in [mass, F(0)]) == lam * mass / 2
p, m = [F(0), F(1)], [F(0), mass]
for value in [F(1, 5), F(4, 5)]:
    assert [value*x + (1-value*mass)*y for x, y in zip(m, p)] == p
gamma, gen, copy = F(23, 25), F(1, 1000), F(1, 2)
prob = gamma * gen + (1-gamma) * copy
assert prob == F(1023, 25000)
assert (1-gamma)*copy/prob == F(1000, 1023)
assert (1-gamma)/prob == F(2000, 1023)
print("semantic tradeoff and gate counterexamples passed")
```

## 8. hard、温度、ST及下降界

fresh pointer在唯一源/目标margin Δ_s,Δ_p>0时有界

\[
1-p_{j^*}\le (L_i-1)e^{-\Delta_s/T_s}+(L_i-1)e^{-\Delta_p/T_p}.
\]

并列时通常得到混合，不能无条件等于按数组首项的hard选择。旧softmax(alpha beta/T)对固定正alpha可低温集中，但alpha=T²时T趋0仍趋均匀，其收敛不一致。温度减小在唯一赢家附近可能使梯度衰减，在tie附近Jacobian却可按1/T增大。

hard fresh仍不能保证最终eta为delta。例如lambda=.8、m=(0,.5)、fresh=(1,0)，eta=(.6,.4)。若部署要求最终位置为一点，必须在eta之后另定义hard投影，并用这份实际Q计算content、G/gamma、后验和状态。这个投影改变模型，不自动随温度退火消失。

规定soft模式作为精确可微概率模型。hard控制需另定record/position tie策略；记录并列时选择null可维持记录重排对称，位置按合法稳定offset处理。hard/soft轨迹一致需要同前缀下最终字节概率margin大于2倍分布误差等条件，不能由pointer熵下降单独推出。

ST的J_soft^T v_hard只验证在硬值点取固定下游v的代理VJP。它通常不同于soft前向损失梯度，也不是hard argmax的真实导数。硬目标区域内可恒定、tie处可不可微。若用代理G套局部下降界，还需真实梯度对齐及光滑邻域条件，不能直接套用真实梯度的descent lemma。

熵项−cH鼓励分散，与CE锐化存在权衡。合法soft支持上梯度为(c/T)p_j(log p_j−sum p log p)，0log0按0、masked项不参与。不能对hard零概率直接求log后把NaN当ST信用。

## 9. 必须统一的派生状态与对照

一份最终Q生成record_mass、byte_mass、content、posterior和coverage；旧unlocked read字段明确标legacy，禁止混入final诊断。当前helper收到EOS会清m，但当前生成循环可能先break，不能把helper性质签为调用路径已提交EOS。新规范先accept/提交终止状态再结束，输出byte cursor与prediction/EOS计数分开。

OP2移除旧alpha在A/B条件位置核的角色后，旧null/record-only参数可能变为结构零信用。按真实消费者冻结、标inert或转交新record head，不能要求每个旧参数非零，也不能静默保留为宣称可学组。

若A/B采用OP2而C仍旧global核，A−C同时含转移语义差异。OP2拟比较A_OP2_ST（新head硬record＋ST）、B_OP2_STOP（同硬前向、停record支路）、C_OP2_SOFTSTOP（同head软record、停支路）；三者共享同一eta及位置模型。A−B测代理信用，B−C测该前向硬化，A−C测两者组合。旧global核及用旧alpha作record权重的版本另列legacy参照，不能冒用N17同名臂的结果。N17.5已按原冻结合同负结案，采用OP2需新身份和重新配平信息/容量/费用；先完成当前N18.1归因，再依据证据决定是否纳入后继合同。

## 10. 独立验证与结论边界

验证器为[scripts/training/verify_taiji_copy_operator_math_v2.py](../../scripts/training/verify_taiji_copy_operator_math_v2.py)，包括以下互相独立的检查。

- Python log域reference与PyTorch概率域张量图对照。
- Fraction隐藏状态路径枚举，比较21个合法观测前缀的序列概率、后验与shift。
- 手推每组算子偏导、逐元素中心FD三档、三步后验递归梯度。
- 240组固定随机域检查，源/记录/坐标重标记、logit常数、温度gap界、mask、空源、单位置、末位和下溢。
- 旧条件化alpha翻转、几何/算术池翻转、hard fresh仍多点、同gamma/错gamma、直接/总梯度和ST局部VJP反例。
- 从当前源码只提取两个helper进行交叉核对，不构造主体或加载checkpoint。

数值FD在float64小型算子图上检查，float32覆盖正常幅度前向及三步梯度对齐，另保留极小生成概率、log cursor跨步信用、近单位重启和罕见补事件的指定压力profile。闭包回归为64条轨迹各16步，合计1024步。−10000平衡后验检查限定Python reference，近单位/罕见重启边界主要验前向，不能推成这些边界的全部梯度已验。它们不代签真实神经参数组、全主体或任务能力。Fraction枚举限定两记录各两位置、固定控制和fresh、最多三步；目标坐标重排只测单步emission，未证明任意位置重排与固定shift可交换。本轮证明为手工概率与代数推导，数值结果作为一致性/反例核对，完整回执见[OP2数学验证](../../reports/taiji_copy_operator_op2_math_20261104.json)。

Log状态合法零为−inf，NaN/+inf拒绝，旧“所有状态必须有限”的检查要按字段分版。持久化须同时保存log_m、log_noncontinuation与源身份/原文版本；JSON若需快照应采用显式zero标记，不能依赖非标准Infinity字面量。

序列验证目标为sum token NLL，实际训练若用mean，应明确1/T缩放后再比较梯度/信用矩阵。代码的数值保护仅在容差内起作用，超范围scaled logits或不一致概率拒绝；概率/梯度/长程误差不具有无条件的全域机器保证。

源码helper交叉核对用于复现本次审阅版本的legacy行为。后续实现修正或迁移后，该诊断要按新旧版本分别归档，不能把保留旧耦合作为永久CI成功条件。数学性质与legacy现象分开判读。

复现命令不指定report时只输出摘要。若写回执必须选择reports下未存在的新文件，禁止覆写冻结回执。

```powershell
python scripts/training/verify_taiji_copy_operator_math_v2.py
```
