# U2 22.4.2 连续物理世界、感知与执行合同

> 设计合同，2026-11-12。数值、字段与接口由[机读合同](../manifests/taiji_u2_22_4_2_world_contract_20261112.json)绑定。本卡22.4.2.1签规格、静态一致性及局部算子核算，完整世界、执行器、数据和任务分别由22.4.2.2–6交付。

## 1. 父身份与实现责任

神经候选使用22.3接受的small初始化配方，通过22.4.1的`metadata_candidate()`构造194参数张量的记录元数据图。模型输入为原始UTF-8目标/请求、获许可记录、传感token及其标定坐标。实际能力资格仍取[22.4.1结案](VALIDATION-TAIJI-U2_22_4_1_CLOSEOUT_20261112.md)。

首世界采用固定相机、1–3个可移动圆片和连续二维平面。每个圆片具有位置、速度、半径、质量、线性阻尼率、颜色和声学共振频率。根目标中的跨模态因果、持续学习、成长和主动选择，继续按同主体任务及消融验收。

| 落地卡 | 交付 |
|---|---|
| 22.4.2.2 | `taiji/u2_task_world.py`的纯世界推进及snapshot；`taiji/u2_task_sensors.py`的真实渲染、声学与确定codec |
| 22.4.2.3 | 物理执行器、恢复分派、外生观测意图、分段因果档案及完整主体接续 |
| 22.4.2.4 | A/B/C/D/J实例、合法历史oracle及全分母评分 |
| 22.4.2.5 | 动作、历史、模态缺失/冲突的固定干预矩阵 |
| 22.4.2.6 | 完整schema、三臂信息权利及资源样例合拢 |

## 2. 时间与动作

世界处于步进模拟时基。一次`act`或`inspect`推进固定`Δ=0.05`世界秒，`ask`推进0秒。处理时钟采用runtime的单调host epoch。包的`t_arrival`取实际生成完成时的处理时钟读数，恢复时按已存epoch及逻辑处理年龄映射。

action共15维，零基0–8为操作one-hot，9/10为目标坐标`q∈[-1,1]²`，11/12为`||u||≤1`的力方向，13为强度`I∈[0,1]`，14为请求时长`τ∈[0,0.05]`。one-hot容差1e-6，其余坐标采用声明闭域并要求有限。runtime issue的`dt`为世界步Δ，`τ`为该步内部的执行脉冲或曝光窗口长度。

- `act`在`[t,t+τ)`内施力，固定相机在步末进行全景采样。
- `inspect`采用零外力，固定相机对q附近区域做精细采样。其裁剪与曝光均由命令决定。
- `ask`由合法记录引用服务回答，物理状态与采样时钟均为零步。帮助量和字节成本另记。
- `express/replay/consolidate/grow/rest/wait`由主体接口处理。模拟器的外生`elapsed_tick`可单独推进Δ并交付被设备计划允许的观测。

每个物理step的传感包覆盖`[t,t+Δ]`，RGB和audio token的实际采样区间保存在坐标中。`requested_duration`、世界推进量、处理wall和CPU费用分别入账。

## 3. 世界状态与动力学

平面边界为`[-1,1]²`，圆片中心满足每轴`[-1+r_i,1-r_i]`。首版半径取0.14/0.18，质量取0.5/1/2，阻尼率γ取0.5/1/2秒⁻¹，回弹系数e=0.2。初始速度各轴在[-0.2,0.2]内，初始圆片之间至少留0.08间隙。采样参数、颜色、声学材质分别使用独立RNG子流。

geometry流先均匀抽n∈{1,2,3}及n个半径，再按body序逐个从`[-1+r+0.04,1-r−0.04]²`拒绝采样中心，满足间隙后抽两项初速，每片最多1024次候选。material流按body序依次均匀抽质量、γ及共振频率。appearance流对机读合同的六种RGB颜色做一次randperm，取前n个。默认相机遮罩为空，具体遮罩由版本化recipe显式给出。世界ID由合同hash、seed和canonical recipe共同散列，body ID仅作内部引用。

外力总尺度`F0=80`。接触工具半径为0.10，对每个圆片定义

\[
w_i=\max\left(0,1-\frac{\|x_i-q\|}{r_i+0.10}\right),\quad
a_i=\frac{w_i}{\max(1,\sum_j w_j)},\quad
F_i=80 I a_i u.
\]

多个接触共享同一力预算。该接触核是首模拟器的柔性工具模型，参数会进入实际代价及任务范围记录。

子步长度满足

\[
h\le\min(10^{-3},\Delta_{remaining}),\quad
h\max_i\|v_i\|+\tfrac12 A_{max}h^2\le r_{min}/8,
\qquad A_{max}=80/m_{min}.
\]

选择上式的最大合法h。微子步最多8192次，余时小于1e-12视为舍入尾量并记录。脉冲占空比为`p=clip((τ−elapsed)/h,0,1)`，每步按当前位置重算接触权重，执行

实现用稳定根`h_move=2δ/(v_max+sqrt(v_max²+2A_maxδ))`，其中`δ=r_min/8`，再与1ms及剩余量取最小值。

\[
v_i^*=\frac{v_i+h p F_i/m_i}{1+\gamma_i h},\qquad
x_i^*=x_i+h v_i^*.
\]

随后进行边界与圆片接触。圆片法向`n=(x_j−x_i)/d`，重叠量`a=max(0,r_i+r_j−d)`。按逆质量分配位置修正

\[
x_i\leftarrow x_i-\frac{a/m_i}{1/m_i+1/m_j}n,\quad
x_j\leftarrow x_j+\frac{a/m_j}{1/m_i+1/m_j}n.
\]

相对法向速度`u_n=(v_j−v_i)·n<0`时施加冲量

\[
J=-\frac{(1+e)u_n}{1/m_i+1/m_j},\qquad
v_i\leftarrow v_i-Jn/m_i,\quad v_j\leftarrow v_j+Jn/m_j.
\]

边界将中心修正到合法区间，仅对向外分量反向乘e。每个子步按起始几何位置(x,y,r)的字典序固定接触顺序，最多16轮；各轮包含全部边界和成对接触。最大穿透≤1e-9且剩余闭合接触的向内法向速度≤1e-8才接受该子步。d≤1e-12、接触不收敛、NaN、微子步耗尽均使整个step失败，世界和RNG退回step前。

无外力阻尼子步的动能乘`(1+γh)⁻²`。孤立碰撞的动能变化为

\[
\Delta K=-\frac{(1-e^2)u_n^2}{2(1/m_i+1/m_j)}\le0.
\]

本式核算覆盖局部离散算子，接触组、整段轨迹、可达性和实际误差由22.4.2.2验收。

## 4. 真实感知与codec

### 4.1 图像与inspect

图像为16×16×3的uint8，经除255后得到模型raw RGB。像素先做每轴2点空间采样，再对4个曝光时间中点做平均。圆片以本身颜色覆盖背景，相机遮罩最后覆盖。遮罩属于成像装置，碰撞几何由第3节给定。

空间子像素偏移为0.25/0.75，图像行从上到下对应世界y从+1到−1，列从左到右对应x从−1到+1。绘制圆片按采样时刻几何位置字典序，矩形遮罩按recipe声明顺序覆盖。像素背景和遮罩颜色在机读合同中给定。

`act/elapsed_tick`使用全景中心c=0、半幅L=1、曝光4ms。`inspect`使用

\[
L=1-0.75I,\qquad c=(1-L)q,
\]

以及步末长度τ的曝光。每个空间采样点先映射为`c+L·pixel_coordinate`，相机外参固定。曝光期间的位置按已经积分得到的子步轨迹做线性插值。τ=0的inspect交付RGB缺失。

每像素加入独立读出噪声，标准差为`min(0.1,(2/255)sqrt(0.016/max(exposure,1/16000)))`，clip到[0,1]后按round-half-to-even量化为uint8。噪声取自image子流，每个物理tick固定抽取16×16×3个标准正态值，缺失时仍推进该流。

RGB按4×4 patch逐行展开为16×48。patch内顺序为行、列、通道。patch坐标为裁剪后映射到世界平面的中心，范围[-1,1]。

### 4.2 声学

接触冲量驱动400/700/1000Hz三组阻尼共振，圆片材质分别选其中一组。共振时间常数0.012秒，冲量幅度`A=min(0.8,|J|/0.5)`。墙碰撞激发该圆片材质；圆片对碰撞按各1/2幅度激发两片材质。工具接触每个子步按实际施加的`||p F_i||h`激发该片材质。

各组复状态遵循`z(t+δ)=z(t)exp((-1/0.012+i2πf)δ)`，冲量时将A加入实部。采样率16kHz，波形取各组虚部之和，加1e-4标准差的独立audio噪声后clip到[-1,1]。子步冲量按发生时刻精确衰减到采样点，同刻先加入冲量再采样。世界每tick生成800个采样，交付最后256个样本，环缓存、共振状态、样本游标及audio RNG均入snapshot。

Hann periodic=true、n_fft=64、hop=32、center=false、normalized=false、onesided=true，经`log1p(abs(STFT))`得到7×33。幅度界给出`log1p(32)<6`，与U2似然域匹配。初始t=0观测的audio为缺失。

### 4.3 本体感受及坐标

8个通道依次为`q_x,q_y,u_x,u_y,I,2τ/0.05−1,tanh(J_tool,x/4),tanh(J_tool,y/4)`。最后两项来自工具受到的实际反冲量，加入0.005标准差读出噪声并clip到[-1,1]。与8个valid位拼成1×16，invalid值置0。t=0初始观测全部invalid；整模态缺失时省略键。每tick固定抽取两项proprio噪声。

坐标8维采用既有`(x,y,u_seq,asinh(lag_start),asinh(lag_end),spatial_valid,sequence_valid,duration_valid)`，时间单位1秒。RGB的seq坐标为0，spatial/duration valid为1。audio空间坐标为0，seq在7帧上等距取[-1,1]，seq/duration valid为1，其第j帧区间为`[(32j−256)/16000,(32j−192)/16000]`。proprio坐标全部0。

可见静音/黑色保持有效。模态缺失由公开设备计划决定，相机遮罩通过像素内容体现。所有基线接收相同raw媒体、时间、标定、已执行动作及合法记录。

## 5. 公私字段与反馈

模型可见通路由机读合同白名单定义。真对象ID、真实位置/速度、质量、阻尼、半径、声学材质索引、真实分割、任务标签、score_only、分区与seed仅供世界/评分/审计。图像坐标、工具命令、工具反冲量属于合法标定与本体感受。

`U2Packet`继续交付observations、coordinates和6维Outcome。物理接口阶段采用明确的`execution_neutral`结果，数值为`[0,1,0,done,1,1]`，表示已知的中性执行效用及是否终态。任务评分留在score_only通道。实际任务反馈采用独立许可记录，按22.4.0释放；将其接入策略效用时须在22.6冻结对应规则与目标尺度。

相机、audio或合法记录提供新证据时，包可按权限允许写入。仅中性执行Outcome且无新证据的包关闭写许可。帮助服务只引用此前已授权的原始记录，缺少可引用记录时返回固定的缺信息回复；帮助内容及记录引用均计成本。

任务API固定交付`goal_raw/goal_version/request_id/request_raw/world_version/evidence_ids/feedback_rights/deadline`的公开与审计部分，以及评分侧的`family/scoring_rule_id/allowed_answers/oracle_references`。具体A/B/C/D/J实例及合法历史oracle由22.4.2.4冻结。请求使用UTF-8，输出仍受64字节和EOS约束。

## 6. 世界API与事务

`PhysicalWorld.reset(recipe,seed)`创建私有状态，`initial_observation(event_id)`产生t=0采样，`step(action_id,action)`执行act/inspect/ask，`elapse(event_id)`执行无外力tick，`query(id)`查已提交结果，`snapshot()/restore(snapshot)`保存恢复纯世界。每个成功事务递增world_version；物理tick同时递增tick_index与world_time。ask仅提交引用服务事务。

`WorldResult`含固定的`event_id/action_id/kind/world_before/world_after/t_start/t_end/raw_media/observations/coordinates/execution_outcome/public_feedback/cost`。任务oracle保存在独立对象。所有id用于协议与审计。

`PhysicalExecutor`提供runtime使用的execute/query/snapshot，以及恢复工厂。执行器持有唯一可写世界与已提交结果缓存。整个候选step先在私有副本上推进动力学、全部传感RNG及结果，再与action_id、命令摘要共同原子持久化，读回通过后才返回ack。相同id同命令返回缓存，相同id异命令拒绝；失败不增加world_version、效应数或RNG。失去规范持久存储身份时query返回unknown并等待求证。

每份结果、原始媒体和已执行命令共同缓存，首profile最多512个事务、合计16MiB，达到任一上限返回具名预算出口。生命周期成本包含失败副本、保存读回和恢复校验。

## 7. 完整恢复与两处现有缺口

世界snapshot固定保存schema、合同hash、世界ID、版本/时钟、圆片物理/外观/材质数组、相机遮罩、全部命名RNG状态、声学复状态/环缓存/样本游标、已提交id/命令/结果、字节计数和终态。执行器snapshot另绑定规范存储身份、commit generation、source pins及effect_count。

新执行器schema为`taiji-u2-physical-executor-v1`，恢复入口按明确schema分派。当前`U2Runtime.from_envelope`缺省使用ReferenceExecutor工厂，22.4.2.3须增加对应分派及失去持久状态的unknown路径。

当前runtime的外部接收要求pending且有ack，原始链只容纳96条。22.4.2.3增加外生观测意图，先对bootstrap/elapsed事件冻结旧状态预测，再打开payload；该路径仅同化观测，行为信用标记为无自主行动。内部等待、重放或表达的完成仍保留其自身身份。

因果档案采用每页最多96条、总计最多16页和16MiB的首profile预算，活跃计算图继续最多32次原子操作。页间绑定前页head/累计entry count/source/config/主体谱系，原始公开事件可流式重编码。bundle保存档案根和当前页引用，恢复校验全部祖先；缺页停在history_unavailable。训练窗按真实调用预留完整一轮，前缀重建和截断跨度逐项入账。

B的48轮至少需要192条choose/issue/ack/receive记录，通常含处理时钟则为240条。按每轮12操作加8初始操作预留584条，处于16×96总容量内。B/J在分段档案、外生意图及完整主体恢复通过后准入。

## 8. RNG与身份

geometry/material/appearance/image/audio/proprio使用独立CPU torch.Generator。每个子流seed为`SHA256(UTF8("taiji-world-v1|seed|tag"))`前8字节按little-endian转整数并模2⁶³。各RNG uint8状态按原字节保存。动力学自身保持确定性，图像/audio/proprio每个物理tick固定消耗本节规定的噪声量。

世界内部采用CPU float64，像素为uint8，物理状态与所有RNG在采样完成后共同提交。神经token在确定codec末端转换到候选dtype。原始媒体与codec结果分别带摘要、合同身份及来源时间。

实际运行数据通过`get_external_path()`解析到`data/experiments/taiji_u2/22_4_2/<run_id>/`。微型测试在`.seed_test_tmp/`隔离，机读报告写reports，合同写plans/manifests。

## 9. 验收、允许误差及预算

本卡静态核验13类判据，并用单字段突变负对照校验约束。局部代数分别用float32/64，容差1e-5/1e-12。未来同条件确定性恢复要求逐位相同的原始媒体、RNG、结果与下一步；动力学穿透/接触速度使用第3节的数值域。

22.4.2.2必须实际执行16轮可达性、同状态不同动作、同外观不同动力学及恢复接续，22.4.2.4必须达到原有合法历史oracle≥0.95和空白历史≤0.60。图像量化、噪声、遮挡与时间采样导致的不可辨任务按fixture_invalid处理。

22.4.2.1静态运行CPU≤20、wall≤60，记入22.4生成/分区200 CPU既有分项。后续完整实现与生成仍按既有预算定价。成功签`frozen_design_contract_verified`，规格缺项签spec_incomplete，真实数值反例签math_mismatch，工程错误签engineering_failure，超帽签budget_stop。

模型前向、optimizer、项目语料、train/fit/check/final实际world、confirm及封存读取均为0。规格核验和任务、学习、物理能力的后继结论各自留账。
