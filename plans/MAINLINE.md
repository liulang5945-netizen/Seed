# 从这里开始

> 文档状态：[当前唯一入口·2026-11-04 单入口收敛]。本文件只保留状态、队首、链接与裁定指针；**唯一主线**全文（状态表、依赖、权威源、冲突消解）在 [MASTER-PLAN](MASTER-PLAN.md)。

**一句话状态**：17 窗口判负（结构主张被 M1 否），18.1 六格判负；包 19（位置计数 + UTF-8 掩码）19.5 判负、19.7 已结案。19.2c 已把"route 穹顶"与"监督无效"分离（成熟父包 route 1.0、span 头学会起点、复合行被冻结 transport 压住）。**19.2e 解决了 0.630/0.979 口径冲突：这是切片偏差，不是定义差异**——48 例切片不含任何 `new_description`，正式全量 0.630/0.583 正确、切片实测池化 0.9792；**19.5 判负不受影响（D16-7）**。强制覆盖探针：B 臂 natural 0.7083 → forced **0.7083**（s4/s16 同为 0.7083）→ **≤0.75，指针路径到顶**；归档 19.5 的 span 指针只有 0.75/0.75–0.96。**19.2f 把损失钉死在新描述半**：B 旧描述 96 例 1.000/0.9375、新描述 96 例 0.2604/0.2292；换序得分差中位 1.19（40/40 对超 θ）但硬 argmax 0/40 改变 → **有序信号已编码、选择饱和**；新描述失败里 route_miss 51.7%。**19.2g 边距训练判该 loss-only 方案无法干预**：训练集 128 个内容起点 0 点激活（gold-decoy 边距中位 8.08 ≫ m=1.0），30 Adam 只训 CE+span；新描述 0.2604→0.2083、route_miss 37→37、换序硬 argmax 0/20→0/20；**地点起点已饱和（top-1 0.9999、边距 12.95）、物件起点未饱和且选错（top-1 0.467、边距 0.828）** → 墙分两半。累计 **2,506.4 CPU**，六格 32,256.6 CPU 未花，confirm 0 次读取。

**队首（唯一）**：**19.2g「记录匹配对比边距」已完成（30 Adam / 1 存包 / 0 confirm，判该 loss-only 方案无法干预）**：在 19.5 B 臂 step300（28031）上只改 loss——每个内容起点对同 world 同类 hard-negative 加 `max(0, m−(score_gold−score_decoy))`、m=1.0、两起点取最差；**训练集抽样 64 例 / 128 点中 0 点激活（gold-decoy 边距最小 5.313、中位 8.08 ≫ m）→ 该边距项在可训数据上恒为零，30 步实际只训了 CE（0.0066→0.0007）与 span**，前后差异只能读作再训练噪声。训练前后：新描述 96 例整句 **0.2604→0.2083**、旧描述 96 例 **1.000→1.000**、全 192 0.6302→0.6042；route_miss **37→37（0.3854）**、route_hit 物件 0.469→0.417 / 地点 0.844→0.906；composite_start 物件 0.385→0.365 / 地点 0.531→0.625；conditional_byte 物件 0.761→0.698 / 地点 1.000→1.000；T2 换序硬 argmax **0/20→0/20**（Δscore 中位 1.243→1.569、top-TV 0.0103→0.0073，更饱和）；top-1 中位 0.9545→0.9658、gold-decoy 边距中位 4.70→7.34。**机制分层（关键）**：地点起点本已饱和（top-1 0.999→0.9999、边距 10.42→12.95），物件起点未饱和且选错（top-1 0.394→0.467、边距 0.695→0.828 < m），route_miss 全在物件侧 → **新描述墙不是单一饱和现象**；地点侧是边距吞顺序，物件侧是描述驱动判别不足。**判据全不过 → 分支 `scoring_head_saturation_is_structural`**。**下一步唯一建议：19.2h 序列表面对齐**（在 `score_rows` 行特征加参数化局部互相关，约 +1,320 参数 / +3.6%，先 30 Adam 选择段诊断），**未启动，等 owner 裁定**。本轮实测 **449.5 CPU / 30 Adam / 1 包 / 0 confirm**（预算 430×3.3=1,419，用 31.7%）；**六格重算价（3.3 系数）111,293.5 CPU、选择段分段价 37,547.6 CPU，仍不放行**。见 [19.2g 报告](../reports/taiji_package19_2g_record_margin_20261104.json)、[ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md) §0/§4。

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
