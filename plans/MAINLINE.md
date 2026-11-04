# 从这里开始

> 文档状态：[当前唯一入口·2026-11-04 单入口收敛]。本文件只保留状态、队首、链接与裁定指针；**唯一主线**全文（状态表、依赖、权威源、冲突消解）在 [MASTER-PLAN](MASTER-PLAN.md)。

**一句话状态**：17 窗口判负（结构主张被 M1 否），18.1 六格判负；包 19（位置计数 + UTF-8 掩码）19.5 判负、19.7 已结案。19.2c 已把"route 穹顶"与"监督无效"分离（成熟父包 route 1.0、span 头学会起点、复合行被冻结 transport 压住）。**19.2e 解决了 0.630/0.979 口径冲突：这是切片偏差，不是定义差异**——48 例切片不含任何 `new_description`，正式全量 0.630/0.583 正确、切片实测池化 0.9792；**19.5 判负不受影响（D16-7）**。强制覆盖探针：B 臂 natural 0.7083 → forced **0.7083**（s4/s16 同为 0.7083）→ **≤0.75，指针路径到顶**；归档 19.5 的 span 指针只有 0.75/0.75–0.96。**19.2f 把损失钉死在新描述半**：B 旧描述 96 例 1.000/0.9375、新描述 96 例 0.2604/0.2292；换序得分差中位 1.19（40/40 对超 θ）但硬 argmax 0/40 改变 → **有序信号已编码、选择饱和**；新描述失败里 route_miss 51.7%。**19.2g 边距训练判该 loss-only 方案无法干预**：训练集 128 个内容起点 0 点激活（gold-decoy 边距中位 8.08 ≫ m=1.0），30 Adam 只训 CE+span；新描述 0.2604→0.2083、route_miss 37→37、换序硬 argmax 0/20→0/20。**19.2h 判零可识别性**：+1,320 参数（+3.56%）的序列表面对齐在 30 步选择段与等额 CE-only 对照臂**逐项相同**（新描述整句 0.2083 vs 0.2083、route_miss 0.3854 vs 0.3854、composite_start 物件 0.3646 vs 0.3646），**Δ 的 bootstrap 区间恰为 [0,0]**；活性门通过（三张量梯度非零、需修 51/96＝53.1%）→ **梯度存在但行为不变**，该参数化退役（D16-10）。**owner 裁定该零结果属"仪器失效"而非结构上限**，立 **D16-12「微调不可用于架构判定」**；**19.2i 经 0 成本同构核查判 `duplicate_objective` 退役（D16-11，省 1,089 CPU）**；**19.2j 从零 A/B 的 1,200 更新全部跑完、4 个 step300 包完整**，但报告序列化在元组键上崩溃，读数与 CPU 计时全丢（工程失败，不是实验失败，D16-15）；只读重建（0 Adam / 311.3 CPU / 319.3 wall）判 **`scoring_form_is_not_the_bottleneck`**：新描述整句 pooled Δ=+0.0312 [−0.0104,0.0677]（等价），物件 route pooled Δ=+0.1146 [0.0625,0.1615]（2000/2000），换序硬 argmax 0–2/20，step300 可辨识性 −0.436；训练 wall 4,961 s、CPU 不可得。累计 3,601.9 CPU 不变（重建另计 311.3），六格 32,256.6 CPU 未花，confirm 0 次读取。**19.2l ② 活性闸已通过**：顺序专用 decoy（描述字符中点两半对调，字节多重集/长度/字节种类完全相同）在 128 个起点上 **激活率 1.0、边距中位 −0.0029**（0 Adam / 16.8 CPU）→ 允许进入 ③ 训练诊断，**等 owner 批准**。

**队首（唯一）**：**19.2l「顺序专用干扰负样本」② 活性闸通过（0 Adam / 128 起点 / 16.8 CPU / 17.7 wall）**。① **decoy 构造**：gold 记录描述按字符中点切两半并对调（`细颈的瓶子 → 的瓶子细颈`），UTF-8 字节多重集、长度、字节种类完全相同，只有顺序不同；每个内容起点 1 条，不做多负样本。② **读数**：训练集 even-stride 64 例 × 2 起点 = 128 个 `score_gold − score_decoy`，退化 0；**激活率 1.0（128/128，门 ≥0.20）**，边距**中位 −0.0029（门 <1.0）**、均值 +0.0288、p10 −0.1504 / p90 +0.2835、min −0.1615 / max +0.4086；物件起点均值 −0.0424、地点起点 +0.1000 → **闸过**。③ **机制提醒（不是判据）**：边距几乎贴在 0，说明冻结池化面对纯字节置换几乎不可分；激活成立不等于梯度可用，若 gold/decoy 关键向量近同，边距项梯度会抵消 → ③ 第一件事必须是首批边距项梯度范数读数。④ **下一步（唯一建议，待批准）**：跑 ③ 训练诊断——1 seed 28031、O 臂、从零、30/300 更新、CE 对照臂 + 唯一变量 order-decoy 边距项；判据换序硬 argmax ≥8/20（**诊断门**）、新描述 route_miss 相对 CE 臂降 ≥0.10、conditional_byte 不回退 >0.05、旧描述不回退；预算 ≤30 独立 Adam/臂（共 60）、≤400 CPU、1 候选包、0 confirm。见 [19.2l 报告](../reports/taiji_package19_2l_order_decoy_activity_20261105.json)＋[manifest](manifests/taiji_package19_2l_order_decoy_activity_20261105.json)。

**继续读**：[MASTER-PLAN](MASTER-PLAN.md)（唯一权威主线：状态表、依赖、权威源、冲突消解、盘点）。
**轨道**：[ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md)（19/20/21 包：步骤、判据、预算、分支）。
**已裁定**：[MASTER-PLAN §5](MASTER-PLAN.md#5-已裁定由-owner-2026-11-04-授权决定原待裁定文字留痕见本条末尾)（M2=①、逐字节门采纳、19 直推、任意切换降级）。
**判据与定理**：[账本](reference/INVARIANT-LEDGER_20261103.md)。**读数边界**：[能力卡](reference/CAPABILITY-CARD_20261103.md)。**流水**：[03](active/roadmap/03_CURRENT_EXECUTION.md)。

## 旧锚点（兼容入口，内容已下沉）

<a id="1-长期目标与当前主计算图"></a>
- **1 长期目标与当前主计算图** → [MASTER-PLAN §0](MASTER-PLAN.md#0-一句话与队首)
<a id="3-唯一执行队列"></a>
- **3 唯一执行队列** → [MASTER-PLAN §2](MASTER-PLAN.md#2-依赖顺序谁先做谁必须等谁)
<a id="r3b-旧基线合同"></a>
- **R3b 旧基线合同**（原合同身份，不发布指令）→ [R3 记录 §6–§7](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)
<a id="4-顺序执行工作卡"></a>
- **4 顺序执行工作卡** → [10 工作包细则](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)
<a id="5-统一验收失败回流与停止"></a>
- **5 统一验收/失败回流与停止** → [MASTER-PLAN §5](MASTER-PLAN.md#5-待裁定只列必须-owner-决定的)＋[10](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)
<a id="09-连续生命与递归进化共同内核"></a>
- **09 连续生命与递归进化共同内核** → [长期发展 V2](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)
<a id="执行位置"></a>
- **执行位置** → [03 当前执行](active/roadmap/03_CURRENT_EXECUTION.md)
<a id="本轮规划边界与下一轮立项"></a>
- **本轮规划边界与下一轮立项** → [MASTER-PLAN §1](MASTER-PLAN.md#1-状态表)
<a id="根目标覆盖与研究准入边界"></a>
- **根目标覆盖与研究准入边界** → [核心需求 L1–L14](active/TAIJI_CORE_REQUIREMENTS.md)

原正文（405 行）逐字节归档：[archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md](archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md)。
