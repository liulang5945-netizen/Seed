# 态极统一主体代数 U1：核心算法与多模态的一体化推导

> 状态 [统一代数候选·推导交付·非能力声明]，2026-11-07。owner 2026-11-07 裁定：多模态部分的定位是**整个算法内部系统的一部分**，不是并列候选；本件据此把[MGW1规格](SPEC-TAIJI-MGW1_MULTIMODAL_WORLD_20261105.md)与既有核心（[算法主稿v1.1](TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)底座＋14–20内容线）闭合为**同一个代数系统**，补齐2026-11-07 seam审计列出的三个未推导seam：状态统一、记录绑定域、表达嫁接。MGW1自此文起重新定位为统一主体的**多模态profile**（公式不变），旧核心是其**text-record profile**。全部命题附推导；可机器核验部分见[预审回执](../../reports/taiji_unified_u1_math_20261107.json)。0训练、0权重加载、0能力声称；生产工程按MGW1覆盖矩阵＋本件§8扩展另行验收。

## 1. 统一状态空间（seam A：状态统一）

统一主体（一个系统，不是模型联邦）：

\[
\mathcal S=(\Theta,\ \Psi_{\pi},\ \Psi_{V},\ \{S_b\}_{b=1..K},\ \mathcal E,\ \Gamma,\ \nu,\ \mathcal C),
\qquad
S_b=(H_b,\ F_b,\ \{o_{bi}\}_{i=1..M},\ \omega_b,\ m_b,\ cov_b).
\]

- Θ/Ψ 全分支共享；每分支持有群状态 H_b、快记忆 F_b、M 个对象槽、分支权重 ω_b，**以及本分支的记录游标 m_b 与覆盖 cov_b**（15.1 的 m/coverage 挂到分支上，不是全局）。
- E 是**共享证据库**：媒体包（rgb/audio/proprio）、**文本记录（字节位置可寻址，含 source/version/时间/许可）**、Outcome，统一为一种包 schema（§7）。
- 两个旧系统都是本空间的限制：**旧核心 = K=1、M=0、E 仅文本记录**；**MGW1 = 记录域为空**。K=1 时分支后验恒等（ω≡1，seam审计S4，1000例偏差0.0），退化自洽。

## 2. 统一 H 方程（7d→8d）

\[
x_g=[N(H_g);\ e_{event};\ m_g;\ r_g^F;\ r_g^O;\ \boldsymbol{r_g^R};\ g;\ D_\nu\nu]\in\mathbb R^{8d},\qquad
H_g'=(1-\ell_g)H_g+\ell_g\tanh(W_gx_g+P_g\tanh(R_gx_g)).
\]

新增第 6 槽 r^R 为**记录读出**（§3），其余 7 槽与 MGW1 逐字相同；W_g 从 d×7d 变 d×8d（每群 +d×d）。

**命题 2.1（有界性承袭）** h∈[−1,1]、0<ℓ≤1 ⟹ h′∈[−1,1]，与输入块数无关。
*证明*：h′=(1−ℓ)h+ℓ·tanh(a) 是凸组合且 |tanh|≤1，对任意有限幅度的 x_g 成立；8d 只是加宽了 a 的定义域。∎（预审 U6：无界随机输入下 20 万步 max|h′|<1）

**命题 2.2（扩展中性：MGW1→U1 的函数保真路径）** 记 Read_R 的输出投影零初始化（末层权重/bias 全零 ⟹ 输出恒为 0 向量）时，8d 图与 MGW1 的 7d 图**逐位同函数**（W 的 r^R 列块乘零向量）；且 ∂L/∂(Read_R 末层) = W_{r^R}ᵀ·[ℓ(1−tanh²(a))⊙∂L/∂h′] 一般非零（启动信用存在，与 §11.2 出生中性同一模式）。∎（预审 U3：逐位相等＋启动梯度范数>0）

推论：U1 可以从**训练中的 MGW1 主体上函数中性地上线**记录域，不需要重训旧参数；旧核心（7d、逐群 F）→U1 **没有**函数保真路径（W 形状与 F 种群都变），只有 §8 的重放式迁移，不宣称权重等价。

## 3. Read_R：记录绑定域（seam B）

统一图上的第四个读者（与 Read_H/Read_F/Read_O 并列；MGW1 已为 Outcome token 建立"自有 binding K/V/null"先例，此处循同一模式接入文本记录）。**每分支**用自己的查询读共享 E：

\[
q_b^R=Q_{read}([s_b;\ \mathrm{goalatt}(s_b,\text{goal units});\ \mathrm{embed}(prev)]),
\quad
\alpha_b=\mathrm{softmax}([\langle K_{rec}\,pool_i,\ q_b^R\rangle/\sqrt{k};\ null_b]),
\]
\[
\beta_{b,i}=\mathrm{softmax}(\langle K_{pos}u_{i,p},\ Q_{val}[q_b^R;pool_i]\rangle/\sqrt{k}-\mathrm{softplus}(\eta)\,cov_{b,i,p}),
\quad
r_b^R=\textstyle\sum_i\alpha_{b,i}\beta_{b,i}V_{pos}(u_i)+\alpha_{b,null}\,v_{null}.
\]

- 查询源 s_b 是分支 H_b 的 QueryRead——记录域由此成为**分支条件**的读取，参数全域共享。
- β 的 coverage 抑制、可学 null、K_rec 无共享偏置（14.4 的死参数纪律）原样承袭。
- r^R 填入 §2 第 6 槽；是否再加 17 式有序选择头/锁定路由属任务级装配（17 线已结案），不进核心代数。
- **transport 默认关闭**：记录内分布直接取 anchor=α·β（19.2c 证明冻结 transport 把复合行压在架构上限；19.2r 的 ByteMatchRoute 是任务级结构原语，保持装配身份）。γ 地板同理：**U1 默认 γ_floor=0**（L5 精确性优先），0.2 备选作为预注册研究臂（2026-11-03 数学审查已给定量后果：地板 f 封顶复制质量 1−f，12 字节名字上限 (1−f)^12）。

## 4. 统一表达（seam C：表达嫁接＋M2 承袭）

**每分支**一个完整混合（15.1 形式、anchor-direct 复制）：

\[
P_b(y)=\underbrace{(\gamma_b+(1-\gamma_b)q^b_{null})}_{\gamma'_b}\ G_b(y)\;+\;(1-\gamma_b)\,C_b(y),
\qquad
C_b(y)=\sum_p w^b_p\,\mathbf 1[y_p=y],\quad \textstyle\sum_p w^b_p=1-q^b_{null},
\]

其中 w^b=λ_b m_b+(1−λ_b m_b^{mass})·anchor_b（λ 为续读门）。**统一发射**按 MGW1 的 scratch 机制跨分支混合：

\[
P(y)=\sum_b \eta_b\,P_b(y),\qquad \eta^{(0)}=\omega,\quad\text{接受字节后 } \eta_b\propto \eta_b P_b(y)\ (\text{世界 }\omega\text{ 不动，L-M2}).
\]

**命题 4.1（归一性）** Σ_y P(y)=Σ_b η_b[γ'_b+(1−γ_b)(1−q^b_null)]=Σ_b η_b=1。∎（预审 U1）

**命题 4.2（后验分解＝两级混合的精确贝叶斯）** 接受字节 y 时，
\[
\text{分支责任：}\ P(b\,|\,y)=\frac{\eta_bP_b(y)}{\sum_{b'}\eta_{b'}P_{b'}(y)}
\;\Longleftrightarrow\; \eta_b\leftarrow \eta_bP_b(y)\ \text{后归一};
\]
\[
\text{分支内位置责任：}\ r^b_p=P(p\,|\,b,y)=\frac{(1-\gamma_b)\,w^b_p\,\mathbf 1[y_p=y]}{P_b(y)}
\;\Longleftrightarrow\; m_b\text{ 游标按 }r^b\text{ 平移}.
\]
*证明*：联合 P(b,p,y)=η_b(1−γ_b)w^b_p1[y_p=y]，边缘 P(b,y)=η_bP_b(y)，条件=联合/分支边缘。∎ **两个旧算子同时恢复**：K=1 时 η≡1、r^b 化为 15.1 的 advance_copy；记录域为空（w^b≡0）时 P_b=G_b、η 更新化为 MGW1 的 scratch 规则。（预审 U2：2M 样本 MC，两式偏差均在采样噪声内）

**命题 4.3（M1/M2/M3 承袭）** M1（∂η/∂α≠0，内容保真与记录选择不可结构性分离）对 anchor-direct 形式逐字成立：copy_mass 对 α_i 的偏导 = Σ_p β_{i,p}1[y_{i,p}=y]，一般非零（预审 U4）。M2 裁定①（逐位继承优先，记录切换限定同一控制范围）统辖 m_b 游标语义：切换记录时 m_b 清零重起，同记录内逐位递推。M3（γ 不是信用）不变。OP2 定理族因此**原样覆盖统一表达**，无需重证。

**注记 4.4（“两个更新”＝一个后验：链式法则，融合的等价与边界）** 命题4.2的两个更新不是两个机制：前向发射已是单一混合，接受字节后的联合后验 P(b,源|y) 是**一个**更新，η 与 m_b 只是它的链式法则因子 P(b|y)·P(p|b,y)。融合形式——对源表 {(b,gen)}∪{(b,p)} 做一次乘法重归一——与分解形式在精确算术下给出**同一联合分布**（[融合注记预审](../../reports/taiji_unified_u1_fusion_note_20261107.json) F1：5字节话语两种发射模式逐步最大差 1.11e-16）；两种形式的持久状态同为 (η,{m_b})，flat 表只是逐步物化。实现可任选其一，逐位恢复须冻结规范形式（两形式浮点结合序不同，ulp 级差异）。分解表述的价值只在可验证性：每个因子恰为既有已验算子（15.1 游标、MGW1 scratch），使统一成为定理而非重设计。**不可融合的是寿命，不是记账**：世界信念ω（仅真实观测更新）、话语信念η（scratch）、游标m_b 各有更新律；把η并回ω，模型自发的5个字节即可把ω推偏54–83%，期间无任何新观测到达（预审 F2/F3：scratch 分离下ω逐位不变）——世界信念跟着自己的话走，正是 L-M2 的算子层含义；脑的同源纪律是 efference copy／感知-运动副本分离。把对象/记录/生成并入单一键空间的 flat 参数化不带来数学优势（两种形式的闭式后验同样精确），登记为研究臂、不默认启用。

## 5. 统一记忆算子（FP1 全域化）

所有 F 写入（对象域、记录/规则域）统一为批量近端算子：

\[
F^+=\arg\min_{\|X\|_F\le C_F}\tfrac12\|X-F_0\|_F^2+\tfrac12\sum_j\beta_j\|X^\top k_j-v_j\|_F^2,
\]

域只决定 (k_j,v_j,β_j) 的构造：对象域用 u_pred 键/[obs;outcome] 值（MGW1 §7）；记录/规则域用问句字节表示键/反馈行值（20.2t 选项2语义；20.3a 的三闸失败是**任务级负结果**，在案不改代数）。

**定理 5.1（同族定理，seam审计S3）** 旧 edit_fast 在 e=k、w=η/(1+η‖k‖²) 时与单对 FP1 无约束解**逐项相等**（500 例最大差 1.87e-14）；β→∞ 时读出响应沿 k 精确收敛 v。两代 F 是同一近端岭族。
**命题 5.2（域序不变性）** 联合批量解对 (k_j,v_j,β_j) 的任意排列（含跨域重排）不变且唯一；顺序分域更新（先对象后记录 vs 反序）一般给出不同结果（预审 U5）——统一代数取联合解，顺序伪效应在算子层消除。

## 6. 统一信用与生命期

- Θ 训练窗扩展为：编码→绑定（对象＋记录）→FP1→写后读回→损失（L_next_sensor＋recon＋masked＋reward＋done＋text task＋**应答窗记录复制 NLL**）；短窗内全图可微，跨窗只存值。
- Ψ 的半马尔可夫迹代数在两个父系统里逐字相同（E_j/δ/γ^d/λ），U1 不改。
- 生长（§12）承袭：出生中性对 8d W 同样成立（新列随 R_new=0 出生零贡献）；槽生命周期（MGW1 §3.1）作用于对象域，记录域无槽概念、由 E 的版本语义管理。

## 7. 统一事件 schema（输入层的"一体"胶水）

一种包：p=(packet_id, source, kind, t_start, t_end, t_arrival, valid, payload)，kind∈{rgb, audio, proprio, **text_command, text_record**, outcome}。旧核心的整个事件流 = text_command＋text_record 的 K=1/M=0 投影；MGW1 的 = 媒体＋outcome 的记录空投影。

**命题 7.1（记录域因果门）** 文本记录与媒体同样受 t_end≤now 门控：t_end>T 的记录在 T 时刻不可读、不进任何预测的输入（预审 U7）；predictive packet 不可变绑定 prior 量。统一系统因此继承 MGW1 的因果非循环闭合，记录域不引入未来泄漏。∎

## 8. 迁移、保持与装配

| 路径 | 语义 | 状态 |
|---|---|---|
| MGW1 → U1 | Read_R 零初始化＝函数中性扩展，旧参数不重训 | 命题 2.2，已推导＋预审 |
| 旧核心(7d/逐群F) → U1 | 无函数保真；shadow 因果重放（§10.3 模式）＋新参数重训 | 机器存在、未物化；不宣称权重等价 |
| Task1 结构装配（ByteMatchRoute 等） | U1 复制算子之上的任务级装配，不是核心代数 | 19.2 判定在案（D16-26） |
| 20 线保持合同 | U1 不改当前执行 20 的身份/预算/冻结门 | 队首不变 |

## 9. 新增不变量（随账本 §20 登记）

- **L-U1 单一主体多 profile**：text-record 与多模态是同一状态空间的限制，禁止把任一 profile 当独立模型发布能力。
- **L-U2 记录域因果门**：记录是 t_end 门控证据；未来记录不得进入过去预测（predictive packet 不可变）。
- **L-U3 表达继承 M2①**：分支内 m_b 逐位继承；记录切换限同一控制范围，切换即清游标。
- **L-U4 全域记忆联合解**：跨域 F 写入用联合批量近端解，禁顺序串写产生域序伪效应。

## 10. 边界与不宣称

统一的是**代数**（状态空间、算子、信用、闭合性质），不是已训练能力：0 次训练、0 权重迁移实测；生产 event controller/发射/慢学习/生长准入按 MGW1 覆盖矩阵＋记录域扩展另行工程验收；20.3a 内容寻址的任务级失败不因算子统一而翻案；ByteMatchRoute 的结构成绩不得记入 U1 代数收益。后续工程包从参考张量核验开始（扩展 MGW1 的 20 组回执到 8d H＋Read_R＋统一表达），先定价后实施。
