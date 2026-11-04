# 从这里开始

> 文档状态：[当前唯一入口·2026-11-04 单入口收敛]。本文件只保留状态、队首、链接与裁定指针；**唯一主线**全文（状态表、依赖、权威源、冲突消解）在 [MASTER-PLAN](MASTER-PLAN.md)。

**一句话状态**：17 窗口判负（结构主张被 M1 否），18.1 六格判负；包 19（位置计数 + UTF-8 掩码）19.5 判负、19.7 已结案。19.2c 已把"route 穹顶"与"监督无效"分离（成熟父包 route 1.0、span 头学会起点、复合行被冻结 transport 压住）。**19.2e 解决了 0.630/0.979 口径冲突：这是切片偏差，不是定义差异**——48 例切片不含任何 `new_description`，正式全量 0.630/0.583 正确、切片实测池化 0.9792；**19.5 判负不受影响（D16-7）**。强制覆盖探针：B 臂 natural 0.7083 → forced **0.7083**（s4/s16 同为 0.7083）→ **≤0.75，指针路径到顶**；归档 19.5 的 span 指针只有 0.75/0.75–0.96。**19.2f 把损失钉死在新描述半**：B 旧描述 96 例 1.000/0.9375、新描述 96 例 0.2604/0.2292；换序得分差中位 1.19（40/40 对超 θ）但硬 argmax 0/40 改变 → **有序信号已编码、选择饱和**；新描述失败里 route_miss 51.7%。**19.2g 边距训练判该 loss-only 方案无法干预**：训练集 128 个内容起点 0 点激活（gold-decoy 边距中位 8.08 ≫ m=1.0），30 Adam 只训 CE+span；新描述 0.2604→0.2083、route_miss 37→37、换序硬 argmax 0/20→0/20。**19.2h 判零可识别性**：+1,320 参数（+3.56%）的序列表面对齐在 30 步选择段与等额 CE-only 对照臂**逐项相同**（新描述整句 0.2083 vs 0.2083、route_miss 0.3854 vs 0.3854），**Δ 的 bootstrap 区间恰为 [0,0]**；活性门本身通过（三张量梯度非零、需修 51/96＝53.1%）→ **梯度存在但行为不变**，该参数化退役（D16-10）。累计 **3,493.4 CPU**（含 19.2h 987.0，其中失败尝试 582.2 观测值），六格 32,256.6 CPU 未花，confirm 0 次读取。

**队首（唯一）**：**19.2h「序列表面对齐」已完成（60 Adam / 2 轨迹 / 1 存包 / 0 confirm，判零可识别性、该参数化退役）**。① **活性门通过（D16-8）**：`W_align.weight` 3.04e-6、`offset_embed` 8.33e-5、`W_hidden` 新列 1.33e-5 全部非零；新描述物件侧 argmax 错 **51/96 = 53.1%**（≥20%）。② **与等额 CE-only 对照臂逐项相同（D16-9）**：新描述整句 **0.2083 vs 0.2083**、route_miss **0.3854 vs 0.3854**、composite_start 物件 **0.3646 vs 0.3646**、conditional_byte 物件 **0.6976 vs 0.6976**；配对 bootstrap 2000 轮 **Δ=0.0、CI [0,0]**（三项同）。③ 旧描述 **1.000→1.000**（不回退）、全 192 0.6302→0.6042（两臂同）、换序硬 argmax **0/20→0/20**（Δscore 中位 1.243→1.569、top-TV 0.0103→0.0073，更饱和）。④ 判据 9 门只过 3（`conditional_byte` 地点 1.0、旧描述 1.0、EOS/UTF-8 冻结路径 0→0）→ 分支 **`gain_is_extra_training`** ＋ **D16-10 `zero_identifiability`**：+1,320 参数（+3.56%，head 37,120→38,440）在 30 步选择段**未进入行为通路**（`offset_embed` 初值 ±0.01、余弦特征量级 1e-2 被淹没）→ **该参数化退役**（代码保留 `alignment=False` 开关，历史 bundle 不删、不上调任何能力等级）。⑤ 身份与预检：父/对照 `773dcec2…`、新图 **`4b887487…`**（必变）；identity rescue **8/8**、存包→恢复 **max_abs_diff=0**、删 `W_align.weight` **必拒**（3/3 通过）。**⑥ 唯一建议：19.2i「描述-记录对比监督」**（0 新参数：在 `scores[:R]` 上做 12 路 listwise 交叉熵、两个内容起点取和、τ=0.1 固定；父包/30 Adam/同序不变，对照臂复用 19.2h 同父同序轨迹）——**未启动，等 owner 裁定**。本轮花费 **404.8 CPU（＋首次失败尝试 582.2 观测值）＝987.0 CPU / 60 Adam / 1 包 / 0 confirm**（预算 445×3.3=1,468.5，用 67.2%）；**六格重算价 111,293.5 CPU、选择段分段价 37,547.6 CPU 仍不放行**。见 [19.2h 报告](../reports/taiji_package19_2h_sequence_surface_20261104.json)、[ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md) §0/§4。

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
