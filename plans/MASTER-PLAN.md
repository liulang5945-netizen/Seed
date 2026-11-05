# MASTER-PLAN（唯一权威主线）

> 文档状态：[当前唯一权威主线·2026-11-05 收束版]。本文件维护当前状态、父资产、依赖和唯一队首；判据见账本，能力解释见能力卡，流水见03。历史长段已[逐字节归档](archive/history/n19_to_n20_20261105/README.md)。

## 0. 一句话与队首

**唯一主线**：以原生字节为主体、同一主体内逐字节表达；按“数据前提 → 图验 → 定价 → 恢复预检 → 正式比较 → 条件确认”的窗口制推进，一个窗口判负就归因、修图、再验，不用训练掩盖结构缺口。

**当前窗口**：19.5历史训练窗口判负；19.2b–w整体关闭。19.2v的受限Task1独立确认188/192＝0.9792（正文/选择/起点/来源全对，4例末端模板外溢）；19.2w停止修复在合成/dev过门，未独立确认。P1-confirmed `1d69e945…` 与开发父P1-stopfix-r2 `0d6a5df3…` 分开登记。12张量/46,642元素消融支持匹配定位复制由结构原语承担，不外推全部继承参数无贡献。当前20.0规划与公共规则预审已完成；**20.1已工程交付（`ready`）**：夹具 51 应答轮／纠正面 22／3 会话（10/24/17）／20 条规则，CF1 与 B_EH 同形（3,323,547 参数、新增 0、差异仅 F 可塑），三条泄漏探针全过（静态扫描 pass、真值打乱 765/765 字节一致且 max|Δlogits| 0.0、反查 690/690 落到已释放反馈、truth 0），dry-forward logits [15,257] 且 F 关闭时两图 765/765 逐字节一致；0 训练。详细数据/误差/成本与退出见[包20 rev2](reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md)，旧报告和19.2v确认数字不变。
**队首（下一条要做的唯一事）**：**20.1：规则学习夹具与 CF1/B_EH 参考图已完成工程交付（`ready`，0 训练），等 owner 判断；正式训练未准入，旧 confirm 保持封存。**

## 0a. 长期目标与阶段边界

沿N主线，发展由态极自身可学习、可恢复的认知机制参与决策与表达的原生认知模型，使其以自然中文可靠完成连续任务，跨会话积累和纠正经验，从可核验经验与反馈学习并保持有效旧能力；在授权和资源边界内，无新提示时也能依目标与状态选择继续、求证、求助或等待，并按需要继承式新增、整合或回收单元/连接/群体。各阶段预先声明范围、误差/纠正预算、身份、对照、真实成本和有限出口。

Task1与执行20分别提供有限结构能力与待验证反馈适应，不能代签一般中文、长期参数学习、主动性或规模生长已完成。阶段编号不是根目标完成比例。

## 1. 状态表

状态取值：**活**（有调度效力）/ **死**（原窗口关闭）/ **待裁定**（owner未决）/ **历史**（只作证据）。14–19.2各行保留其原判决；行内当时建议不再发布当前动作，当前执行只有页首队首。

| 线名 | 状态 | 最近读数 | 入口判据 | 出口判据 | 证据指针 |
|---|---|---|---|---|---|
| 09–13 生长/种子/剂量方向 | 死 | 旧剂量臂均未过任务门；仅保留工程资产 | 旧合同 | 旧合同 | [05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md) |
| 14.6 完整动作 | 死 | 0.5844（门 0.80） | 旧合同 | 任务门 | `reports/*14_6*` |
| 15.1 内容转移 R1 | 死 | 0.5625（2400 Adam） | R1 合同 | 任务门 | [N15 规范](reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md) |
| 16.1 选择—解码绑定 | 死 | 0.0500 / 0.0167（1200 Adam） | 16 合同 | 任务门 | `reports/*16_1_compare*` |
| **17 选择—锁定绑定窗口** | **死** | 六格 overall 0.000–0.200（门 0.80），任务门全未过；`n17_negative_closeout`；结构消解被 **M1** 判为不可解读 | 17.2R 图验通过 | 整句 ≥0.80、复制 ≥0.60、结构 A−C>0 | `reports/taiji_selection_routing_17_7_closeout_20261104.json` |
| OP2 数学 **M1** | 生效（永久） | ∂η/∂α ≠ 0：内容保真与记录选择不可结构性分离 | 数学审查 | — | [OP2 规范](reference/SPEC-COPY_OPERATOR_OP2_MATH_20261104.md) |
| OP2 数学 **M2** | **生效（已裁定 ①）** | “任意记录切换 + 逐位恢复”不可同时要求 → 取 ①：保逐位继承，记录切换限定“同一控制范围内” | 数学审查 | owner 裁定 ①（2026-11-04） | [账本 §13](reference/INVARIANT-LEDGER_20261103.md) |
| OP2 数学 **M3** | 生效（永久） | 前向混合权重 ≠ 训练信用；不得据 γ 推断信用饥饿 | 数学审查 | — | 同上 §12 |
| 18.0 数据前提/定价/预注册 | 活（已交付） | 220 名字池、训练/确认不相交、六格裸价 9,109.4 / 保护价 19,220.8 CPU（18.0 口径） | 三条数据前提 | 预注册 10 门冻结 | `reports/taiji_data_premise_18_0_pricing_20261104.json`＋`plans/manifests/…18_0_pricing…json` |
| **18.1 六格正式比较** | **死** | overall 0.000、`new_name_copied` 0.000、unseen 0.000；16,196.3 CPU / 1,800 Adam；`release_confirm_n18_2=false` | 18.0 预注册＋图验 | 10 门 | `reports/taiji_selection_routing_18_1_compare_20261104.json` |
| 18.2 归因（γ 扫描/UTF-8/漂移） | 完成（结论生效） | 复制命中 0.090；b 0.290；b_pure 0.755；c 0.390；c_γ0 1.000；γ 6 档整句全 0 | 18.1 判负 | 四条归因落账 | 账本 [§11](reference/INVARIANT-LEDGER_20261103.md)＋[§14](reference/INVARIANT-LEDGER_20261103.md) |
| **包 19 位置计数＋UTF-8 掩码**（原 18.2 修复包 = 19.1–19.2） | **死（19.5 判负、19.7 结案）** | 19.5 六格 overall 0.000–0.630（门 0.80），最佳格 28031-B；逐字节门物件 0.678 / 地点 0.631；分账 ledger：起点命中 0.347/0.340，起点命中后内容字节 0.941/0.980；29,574.2 CPU / 1,800 Adam；`release_confirm_19=false` | 诊断归因（transport 指针＋解码掩码） | 19.5 逐字节门（物件/地点 ≥0.95、抄回 ≥0.60、整句 ≥0.80）＋预注册 10 门 | `reports/taiji_cursor_utf8_18_2_20261104.json`、`reports/taiji_package19_compare_20261104.json`、`reports/taiji_package19_probe_step300_20261104.json`、`reports/taiji_package19_closeout_19_7_20261104.json` |
| **19.2b 起点监督**（owner 2026-11-04 接受分支） | **停（D16-5 修正后按原文口径未上升，不放行六格）** | 定价 32,256.6 裸价 / 68,061.4 保护价；D16-5 合成核对：`row_logits = transport + span bias` 逐元素最大差 **0.0**、pointer one-hot 全一致、carry 绕过合成；19.5 五有限格三读：β 0.400/0.400、pointer 0.833/0.800、composite 0.767/0.746（route 0.832）；28031-A 全 NaN（2,320,153 参数）已剔除；30 独立 Adam 小验证：`composite_row` +4.17pp 双边后走平，`composite_start` 0.000→0.000 / 0.000→0.021，route 0.005→0.038；累计 953.0 CPU | 19.7 结案＋owner 分支授权＋D16-5 勘误 | 阶段门 `composite_start_hit` ≥0.95（物件/地点）＋route ≥0.95＋conditional 不回退 → 六格 → 最终门；当前未达标，六格不放行 | [D16-5 复合仪器](../reports/taiji_package19_2b_composite_instrument_20261104.json)、[小验证](../reports/taiji_package19_2b_lowupdate_composite_20261104.json)、[19.2b 决策](../reports/taiji_package19_2b_decision_20261104.json)、[定价](../reports/taiji_pricing_19_2b_20261104.json)、[预检](../reports/taiji_preflight_19_2b_20261104.json) |
| **19.2c 成熟父包诊断**（独立诊断包，**不计入** 19.2b 的 60 更新上限） | **完成（架构上限已定位）** | 父包换 19.5 **B 臂 step300**（父包 route 0.998/1.0、composite 0.917/0.979）；同额 60 独立 Adam、仅 selection、挂 32-d span 头（4,194）；`composite_start_hit` 物件 **0.625→0.625（全程恒定）**、地点 **0.792→0.792（u0–u30）→0.812（u45）→0.917（u60）**；`pointer_start_hit` **+0.958 / +1.000**；span CE **6.770→1.541**；route 1.0/1.0 全程；`conditional_byte` 1.0/1.0；320.6 CPU、0 存包、0 confirm | owner 2026-11-04 授权（换父包、不加剂量、不延长轨迹） | `composite_start_hit` ≥0.95（物件/地点）＋`conditional_byte` 不回退（≥0.94/0.98） | **未达标但可分离**：route 不是穹顶（全程 1.0），span 头已学会 gold 起点；复合行被冻结 transport 的行质量压住 → **可加 bias 翻不过冻结 transport，属架构上限**，不是监督失效 | [19.2c 报告](../reports/taiji_package19_2c_mature_parent_20261104.json)＋[manifest](manifests/taiji_package19_2c_mature_parent_20261104.json) |
| **19.2b 残差归因**（0 更新、0 存包） | **完成（句框不是主因）** | 19.5 六格 step300、48 例/格、自由解码；D16-6 格剔除后 5 健康格 pooled：整句 **0.6875**、假设起点=1.0 且抄写=1.0 的理论上限 **0.7417**；六类互斥占比 `frame_byte_wrong` **20.83%**（模型生成 42 / 解码-EOS 8）、`start_miss` 6.25%、`content_byte_wrong` 3.75%、`eos_not_stopped` 0.42%、`route_miss` 0、`order_or_extra` 0；每臂 B **0.979（仅易层 48 例切片；全量正式评 0.630/0.583，D16-7）**、C **0.740**（上限 0.875，起点 13.5pp＋内容 9.4pp）、A 28032 **0.000**（48/48 frame-first 掩蔽 route）；B+C 池化 **0.8594 / 上限 0.9271**；111.2 CPU | 19.5 归档六格（只读） | 六类互斥且合计 100%，另报理论上限 | 完成：**句框在易层只压 2.1pp**；但易层≠任务门，切片口径偏差已由 19.2e 登记为 D16-7；`new_description ~0.26` 不在本表范围 | [残差归因报告](../reports/taiji_package19_2b_residual_attribution_20261104.json)＋[manifest](manifests/taiji_package19_2b_residual_attribution_20261104.json) |
| **19.2e 口径对账＋强制覆盖探针**（0 更新、0 存包） | **完成（口径冲突已解，指针到顶）** | ① 两种评分同一谓词（`raw==target_prefix` 27 字节＋EOS，均走 `prefix=bytes(raw)` 掩码）；48 例切片**不含任何 `new_description`**（96 例全在其余半）；正式全量 **0.6302/0.5833**、切片实测 **1.000/0.9583（池化 0.9792）**、归档易层 1.000/0.9375 → 0.979 = 切片乐观。② 强制覆盖（24 例/格 = 易 12＋new_desc 12）：B natural 0.7083 → forced **0.7083**、s4 0.7083、s16 0.7083；起始物件 0.625→0.75 但内容字节 0.840→0.812；C 对照 0.458→0.500 / 0.458→0.333；归档 19.5 span 指针仅 0.75/0.75–0.96。171.6 CPU | owner 2026-04 对账要求 | 强制后整句 ≥0.80 且逐字节 ≥0.95 → 指针授权；否则指针到顶 | **指针路径到顶**（B 0.7083 ≤0.75）；正式评未被低估、19.5 判负不受影响（D16-7）；转句框/解码归因，不预启动训练 | [19.2e 报告](../reports/taiji_package19_2e_pointer_override_20261104.json)＋[manifest](manifests/taiji_package19_2e_pointer_override_20261104.json) |
| **19.2f 新描述墙归因**（0 更新、0 存包） | **完成（墙在组合匹配，不在接线）** | ① 同一 192 例、正式谓词分层：B 旧描述 96 例 **1.000/0.9375**（池化 0.9688）、新描述 96 例 **0.2604/0.2292**（池化 0.2448）；新描述 route 物件 0.469/0.375、地点 0.844/0.823；composite_start 物件 0.385/0.375、地点 0.531/0.417；起点对后 conditional_byte 物件 0.761/0.829、地点 1.000/0.946。② T2 换序（同字节多重集、只交换两个描述）：表示相对 L2 中位 0.246/0.255、得分最大差中位 **1.19/1.21**、40/40 对超 θ、**硬 argmax 改变 0/40（新）3/40（易）**、TV 中位 0.016/0.059；decode 起点打分（前 3 字节 teacher-forced）改变 0/40（易）/14/40（新）、跟对首实体 40/40→0/40（易）/27/40→5/40（新）。③ 新描述 96 例失败占比：route_miss **51.7%**、content 44.2%、start 3.4%、eos 0.7%、frame 0%、order 0%；route_miss 占 96 例 **39.1%**。**500.5 CPU** | owner 2026-11-04 转向新描述墙 | 换序敏感（θ_repr/θ_score）＋新描述主因 route_miss → 组合匹配容量/目标不足；换序不敏感 → 图接线；否则下游 | 判定 **组合匹配容量/目标不足**（有序信号已编码、边距吞掉顺序）；唯一建议 **19.2g 对比边距**（+0 参数、≤30 Adam / ≤220 CPU、1 包、0 confirm，未启动）；成本超 150 预算 3.3×已登记 | [19.2f 报告](../reports/taiji_package19_2f_new_description_wall_20261104.json)＋[manifest](manifests/taiji_package19_2f_new_description_wall_20261104.json) |
| **19.2g 记录匹配对比边距**（30 Adam、1 存包） | **完成（该 loss-only 方案无法干预）** | 19.5 B 臂 step300（28031）；只改 loss：每内容起点对同 world 同类 hard-negative 加 `max(0, m−(score_gold−score_decoy))`、m=1.0、多起点取最差；+0 参数、不改图、只训 selection。**训练集 0/128 点激活**（边距最小 5.313、中位 8.08 ≫ m）→ 边距项恒零，30 步只训 CE（0.0066→0.0007）与 span。新描述 96 例 **0.2604→0.2083**、旧 96 例 1.000→1.000、全 192 0.6302→0.6042；route_miss **37→37**、route_hit 物件 0.469→0.417 / 地点 0.844→0.906；composite_start 物件 0.385→0.365 / 地点 0.531→0.625；conditional_byte 物件 0.761→0.698 / 地点 1.000→1.000；T2 硬 argmax **0/20→0/20**（Δscore 中位 1.243→1.569、top-TV 0.0103→0.0073）；top-1 中位 0.9545→0.9658、gold-decoy 边距中位 4.70→7.34；地点 top-1 0.999→0.9999（饱和）、物件 top-1 0.394→0.467（未饱和且错）。449.5 CPU / 30 Adam / 0 confirm | owner 2026-11-04 授权 19.2g | 新描述 route_miss ≤0.05、composite_start ≥0.70、conditional_byte ≥0.90、新整句 ≥0.80、旧 96 例 ≥0.9375、T2 硬 argmax 换序 ≥18/20、EOS/UTF-8 不回退、≤30 Adam | **判据全不过** → `scoring_head_saturation_is_structural`；新描述墙分两半：地点侧饱和（边距吞顺序）、物件侧未饱和且选错（描述驱动判别不足）；唯一建议 **19.2h 序列表面对齐**（+约 1,320 参数），未启动；六格重算价 111,293.5 CPU 不放行 | [19.2g 报告](../reports/taiji_package19_2g_record_margin_20261104.json)＋[manifest](manifests/taiji_package19_2g_record_margin_20261104.json)＋`output/taiji_package19_2g_record_margin/28031-B_stop_lock-step30-margin.pt` |
| **19.2h 序列表面对齐**（60 Adam / 2 轨迹 / 1 存包） | **死（零可识别性，参数化退役）** | 父包 19.5 B step300（28031）；新 head `alignment=True`：`W_align` 32×32 共享投影 + 5×8 偏移嵌入 + `W_hidden` 输入 576→580，**+1,320 参数（+3.56%）**；只训 selection、30 独立 Adam；**等额 CE-only 对照臂**（同父/同步/同序/同损失）。活性门通过（行内三张量梯度 3.04e-6 / 8.33e-5 / 1.33e-5，需修 51/96＝53.1%）；两臂**逐项相同**：新描述整句 0.2083 vs 0.2083、route_miss 0.3854 vs 0.3854、composite_start 物件 0.3646 vs 0.3646、conditional_byte 物件 0.6976 vs 0.6976，**Δ=0.0 / CI [0,0]**；旧描述 1.000→1.000、全 192 0.6302→0.6042、换序硬 argmax 0/20→0/20。**987.0 CPU（404.8 实测＋失败尝试 582.2）/ 60 Adam / 0 confirm** | owner 2026-11-04 接受 19.2h＋D16-8/D16-9 硬前置 | 9 门沿用 19.2g 数值；增益必须报 Δ＋bootstrap | **判 `gain_is_extra_training` ＋ D16-10 `zero_identifiability`**：+3.56% 加法特征在 30 步被淹没、未进行为通路 → **退役**；唯一建议 **19.2i 描述-记录对比监督**（0 新参数、listwise 12 路 CE、功能项 330×3.3＝1,089 CPU），未启动 | [19.2h 报告](../reports/taiji_package19_2h_sequence_surface_20261104.json)＋[manifest](manifests/taiji_package19_2h_sequence_surface_20261104.json)＋`output/taiji_package19_2h_sequence_surface/28031-B_stop_lock-step30-align.pt` |
| **19.2j 从零选择段 A/B**（1,200 更新完成；报告序列化崩溃，step300 只读重建） | **完成（打分形式不是瓶颈）** | 2 seed × 2 臂 × 300＝**1,200 Adam 全部跑完**，4 个 step300 bundle 完整（02:10:20–02:13:49）；报告在 `training.first/last` 的元组键上崩溃，中途读数与 CPU 计时全丢（**工程失败，非实验失败**，D16-15）。只读重建（**0 Adam / 311.3 CPU / 319.3 wall**）：新描述整句 O 0.2396/0.2396、N 0.2604/0.2812；新描述 route_hit 物件 O 0.375/0.5312、N 0.6146/0.5208，route_miss O 0.4479/0.3333、N 0.2604/0.3333；旧描述两臂 1.000；EOS/UTF-8 192/192 全合法。Δ=N−O（12 world bootstrap 2000 轮，`n_wins`=Δ>0 轮数）：整句 +0.0208 [0.0000,0.0521] / +0.0417 [−0.0312,0.1042]、pooled **+0.0312 [−0.0104,0.0677]**（1855/2000）→ 等价；物件 route +0.2396 [0.156,0.323] / −0.0104 [−0.042,0.021]、pooled +0.1146 [0.0625,0.1615]（2000/2000）；物件起点 pooled +0.0990 [0.0417,0.1562]。换序硬 argmax O 2/0、N 0/0（远低于 18/20）。step300 可辨识性 **log10 per-param RMS 比 −0.436（聚合 −1.208）→ 可辨识** | owner 2026-11-05 处置：训练成果有效、记账失败；只读重建、修序列化、立 D16-15 | 预注册 8 门；pooled 整句 Δ 下界 ≤0 且非全门通过 → 判等价 | **分支 `scoring_form_is_not_the_bottleneck`**（两 seed 同判）；7 门中只过 2（旧描述不回退、EOS/UTF-8）；**训练 wall 4,961 s（00:51:08→02:13:49），CPU 不可得＝仪器缺口**；step50/100/200/250 读数 `unrecoverable`、step100 梯度比 `inferred=过闸、measured:false`；唯一建议 **19.2k「span 指针从零可训」**（未启动，等裁定与计价）；confirm 0 | [19.2j 报告](../reports/taiji_package19_2j_from_scratch_selection_ab_20261104.json)＋[manifest](manifests/taiji_package19_2j_from_scratch_selection_ab_20261104.json)＋`output/taiji_package19_2j_from_scratch_selection_ab/`（4 包） |
| **19.2l 顺序专用干扰负样本**（② 活性闸通过；A 结构证明完成；B 已跑完并判 `washout_at_readout_proven`） | **A 推翻"O 臂梯度恒零"前提；B 判 readout 层冲掉（差在没穿过 argmax）** | **② 活性闸**：decoy＝gold 记录描述按字符中点切两半对调，字节多重集/长度/字节种类完全相同、只有顺序不同；每起点 1 条。训练集 even-stride 64 例 × 2 起点＝**128 读数、退化 0**：**激活率 1.0（128/128，门 ≥0.20）**、边距**中位 −0.0029（门 <1.0）**、均值 +0.0288、p10 −0.1504 / p90 +0.2835 → **闸过**（0 Adam / 16.8 CPU）。**A 结构证明（0 Adam / 21.9 CPU）**：命题"袋装打分器 ⟹ 置换负样本边距项梯度恒 0"数学成立，**但当前 pipeline 不满足前提**——冻结 transport 对置换敏感，`max|Δkey|` 中位 **1.491**（min 0.994 / max 1.838，128/128 全不同）；**O 臂梯度范数 1.5965（非 0）、N 臂 1.6015**，新增块仅 **0.0089（0.56%）**；N 首批 **2.1515 > 1e-6**；N 初始化可辨识性 **log10 RMS 比 −1.425（不可辨识；step300 才 −0.436）**。**B 已跑完（D16-17 量级闸：开跑前 866.6 ≤ 2×1,200 → 直接跑；60 Adam / 1 包 / 0 confirm / 784.9 CPU / 802.1 wall / 占用 65.4%）**：两臂 30 更新/臂。step0（共用快照）：新描述 route_miss 0.5417、route_hit 物件 0.0 / 地点 0.0833、conditional 0.0 / 0.25、completed 0.0、旧描述 0.0、换序硬 argmax 4/20。CE 对照 step30：route_miss **0.4896**、route_hit 物件 **0.3542** / 地点 **0.7813**、completed **0.1875**、旧描述 **0.9479**、conditional **0.7608 / 0.9797**、argmax **3**。边距臂 step30：route_miss **0.5938**、物件 **0.1354** / 地点 **0.7188**、completed **0.0521**、旧描述 **0.9688**、conditional **0.7436 / 0.9598**、argmax **1**。Δ＝边距臂−对照（96 例新描述、12 world bootstrap 2000 轮）：route_miss **+0.1042 [0.0521,0.1563]**（变差）、物件 route_hit **−0.2188 [−0.3021,−0.1458]**、地点 −0.0625 [−0.1667,0.0417]、completed **−0.1354 [−0.2083,−0.0625]**。三读：`key_diff_after` 中位 **1.4912（基线 1.4912，降幅 0.0）**；`score_margin` 中位 **−0.0048 → 1.562**（配对 bootstrap **+1.5667 [1.4905,1.6412]**，128 读数，激活率 1.0→0.0）、首批 decoy 边距 0.0339 → **1.5533**；`gradient_share_by_param` **30/30 全程**：W_hidden **58.98%**、w_out **38.92%**、W_hidden.bias 1.69%、offset_embed 0.33%、W_align 0.084%、k_null 0，新增块合计 **0.416%**。诊断门 5 项过 3：`conditional_byte` 不回退 >0.05（物件 0.7608→0.7436、地点 0.9797→0.9598）、旧描述不回退（0.9479→0.9688）过；换序硬 argmax ≥8/20（实测 1）、route_miss 相对 CE 降 ≥0.10（实测 −0.1042）不过。包：`28031-N-alignment-step30-order-decoy.pt`（sha256 `5c1c2c14…`），save→load 往返 **max_abs_diff 0.0** | owner 2026-11-05：先做 A 结构证明，B 改 N×margin vs N×CE-only；A 读数出来再问 B；**B 改量级闸（D16-17），1,200 上限、≤2× 直接跑** | ② 激活率 ≥0.20 且中位 <1.0；A 需 ①`max\|Δkey\|` 与 ②梯度范数成对读数，`≤1e-12` 记结构零、`>1e-6` 才可开训；B 需 ≤30 Adam/臂、1 包、0 confirm；**预算闸：预估 >2×1,200 才允许停（D16-17）** | **② 闸过；A 判 `frozen_transport_is_order_sensitive`**：O 臂不是纯袋装打分器（D16-16）；**B 判 `washout_at_readout_proven`**：key 未降、边距已拉开 >1.5，但换序硬 argmax 4→1、route_miss 反而 +0.10 → 差在打分差**没穿过 argmax**；任务面同时变差（completed −0.1354、物件 route −0.2188）→ 该 loss 在 30 步剂量下**有害无益**；下一步唯一允许＝readout 层顺序敏感重排（成对比较/top-1 抑制），禁止加步数、禁止动上游 key | [② 报告](../reports/taiji_package19_2l_order_decoy_activity_20261105.json)＋[A 报告](../reports/taiji_package19_2l_margin_gradient_20261105.json)＋[B 报告](../reports/taiji_package19_2l_order_decoy_diagnostic_20261105.json)＋[B manifest](manifests/taiji_package19_2l_order_decoy_diagnostic_20261105.json)＋账本 §16.9–§16.11 |
| **19.2n 同池孪生设计闸**（0 Adam / 0 包 / 0 confirm） | **完成（判 `twin_not_constructible`：池内不存在同池孪生）** | 192 dev 例 × 2 内容起点＝**384 读数（96 唯一 world×key×起点组合）**：池内可构造孪生 **0/384（0.0）**、实体受限孪生 **0/384**、池内字节多重集相同的 key 对 **0**、每起点孪生数 **{0:384}**。线级孪生（记录行内描述↔名字互换）**384/384 同长同字节多重集**，但 **0 条在册**；互换后首实体 **128/384 仍是原描述（key 不变→重复项、分数差恒 0）**、**256/384 变成名字（名字从不是候选 key→池外）** → 能作为不同 key 进池的 **0**。跨记录互换描述 **0/336 改变 key 集**。`score_gold − score_twin` 测量集为空 → 未加载任何包、未做前向。**0.203 CPU / 0.99 wall** | owner 2026-11-05 裁定 19.2m 方向否、D16-19 受理条件 | 池内可构造比例 ≥0.60 且边距中位 <1.0 → `twin_gate_pass`；<0.60 → `twin_not_constructible`；边距中位 ≥1.0 → `twin_already_separated` | **`twin_not_constructible`**：池内对手不存在（不是“训练学不会”）；唯一允许＝另立数据前提提案（key 改整行、池扩为描述×名字叉积），禁止在现有 8 条池上再提池外负样本/改 readout 判据 | [19.2n 报告](../reports/taiji_package19_2n_twin_pool_20261105.json)＋[manifest](manifests/taiji_package19_2n_twin_pool_20261105.json)＋账本 §16.12 |
| **19.2o provenance 零更新补算**（0 Adam / 0 包 / 0 confirm） | **完成（L1 通过；L2 目标未达成，差 1,986 字节）** | 19.2j 4 个 step300 包 × 96 例 = **384 例、768 个内容跨度、6,912 字节**。L2：`provenance_bytes` **4,926（0.7127）**、`misbound_bytes` **0**、`fabricated_bytes` **1,986（0.2873）**，`accounting_closed = true`；目标 `fabricated_bytes = 0` **未达成**。按年龄：**旧描述 3,456 字节 100% 可溯源、0 编造**；全部编造落在**新描述**半（新描述 3,456 字节中编造 **57.5%**）。四格（字节/占比）：选对且抄对 4,650（67.3%）、选对抄错 666（9.6%）、选错但抄自己选中记录 276（4.0%）、选错且编造 966（14.0%）、截断未写完 354（5.1%）；跨度 534/60/0/115/59。**关键机制事实：`wrong_route_right_content = 0`**（选错却抄对 0 次）→ 内容抄写绑定在所选记录上。L3：selection **0.8112**、随机基线 0.1111（非空 0.125）、可达上限 1.0、相对上限 0.7715–0.8184、逐包 bootstrap 下界最低 **0.7396**。L1：4 包 `file_sha256` 与 `candidate_sha_v2` 全等于 manifest、UTF-8 非法 **0/384**、confirm **0**、未存包（恢复引用 19.2l-B 往返 0.0）；首跑 identity 字段误存整块 dict，已用 `--identity-only` 零解码复核（0.08 CPU）。**141.5 CPU / 147.0 wall** | owner 2026-11-05 指令（D16-20 三层验收） | L1 全绿（零容忍）；L2 报 `provenance_rate` / `fabricated_bytes` / `misbound_bytes` 且完整划分；L3 报 selection/随机/上限三数，阈值待裁定 | **L1 通过；L2 未达 `fabricated_bytes=0`**；编造只在新描述、且与“选错”绑定 → 唯一建议 **19.2p 候选池口径提案**（key 改整行），未启动、等裁定 | [19.2o 报告](../reports/taiji_package19_2o_provenance_20261105.json)＋[manifest](manifests/taiji_package19_2o_provenance_20261105.json)＋账本 §16.13 |
| **19.2p–19.2v 结构实现线**（19.2p 提案闸 → 19.2r 匹配闸 → 19.2s 扰动闸 → 19.2t 零训练装配 → 19.2u 停止边界＋消融 → 19.2v 封存确认） | **完成并整体关闭（任务 1＝绑定复制＋选择，由结构实现达成；训练权重贡献 0）** | 19.2p 判 `proposal_not_admissible`（0 前向 / 2.34 CPU）；19.2r 描述串唯一出现 384/384、纯扫描选择/整句 1.0、`ByteMatchRoute` 65,807 参数注入后 gold 192/192（1.88 CPU）；19.2s ε 8 档 / σ 5 档全 1.000、随机初始化第 0 步已 1.000、无崩塌点（182.45 CPU）；19.2t 零训练装配 L1 全 1.000、整句 γ=0 **0.3854**、失败全在字节位 15（130.05 CPU）；19.2u 停止边界结构原语 → 整句 **192/192**、消融（transport/span/score_rows 随机化）**0.9479**、失败全在模板字节位 1（256.72 CPU）；19.2v confirm 首次单读 **188/192＝0.9792**（旧 95/96、新 93/96）、L1 全 1.000、4 例失败全为位 27 多写 1 个模板字节"到"、与 dev Δ **−0.0208**（144.06 CPU）。`candidate_sha_v2` `773dcec2`→`95099768`→**`1d69e945`**（参数 3,563,859 不变、`byte_match` 65,807） | owner 2026-11-05 逐包授权（每包单独立项、零训练） | 每包单一问题；L1 机械不变量全 1.000 零容忍 | **19.2 系列整体关闭（19.2b–19.2v）**：匹配层与停止边界均为结构原语，训练对本任务贡献 0；**本任务此后不得再以训练立项（D16-26）**；下一个能力必须带**非算法残余** | [19.2r](../reports/taiji_package19_2r_byte_match_gate_20261105.json)、[19.2s](../reports/taiji_package19_2s_perturbation_gate_20261105.json)、[19.2t](../reports/taiji_package19_2t_zero_training_assembly_20261105.json)、[19.2u](../reports/taiji_package19_2u_stop_boundary_ablation_20261105.json)、[19.2v](../reports/taiji_package19_2v_confirm_holdout_20261105.json)＋账本 §16.14–§16.20 |
| **19.2w 边界残差复现与修复** | **完成（`fixed`；封存集未复核）** | 冻结外合成：尾类 64 例 1 例外溢（1.56%）、对照 64/64；位 27 EOS 0.3492 vs 顶位 0xE5 0.3968，随后补齐三字节 → 外溢为模板续写；1,536 条记录中"到"0 次 → 非复制通路。修复 `copy_stop_boundary.sequence_stop`（最后一跳字段停止边＝序列终点；0 参数 / 0 样例常数 / 无字符特判）。三道门：合成尾 64/64、对照 64/64、dev 192 全 L1 1.0000、11 类末字符各 8 例全 0 外溢。290.51 CPU | owner 2026-11-05 授权（0 confirm / 0 训练） | 合成侧同量级复现＋修复 0 参数 0 特判＋dev 不回归 | `candidate_sha_v2` `1d69e945…`→**`0d6a5df3…`**（参数 3,563,859 不变）；新装配态 `assembled_stopfix_r2`；confirm 0.9792 不改写 | [19.2w 报告](../reports/taiji_package19_2w_boundary_residual_20261105.json)＋账本 §16.21 / D16-27 |
| **20.1 规则学习夹具＋CF1/B_EH 参考图** | **工程交付完成（`ready`；0 训练，等 owner 判断）** | 夹具 4 world／204 轮（51/world）、纠正面 22/world、会话 [10,24,17]、反馈 2 批 40 行 1,560 B（v1 20 行 780 B、v2 20 行 780 B）、反例 7、changed_query 25（6.25/world）、20 条规则；两图 **3,323,547 参数 / 135 张量 / 新增 0**、同形（名称/形状/dtype 全同、参数量差 0）；CF1 `candidate_sha_v2` `a8a1e169…`、B_EH `3e98f65d…`；探针：静态 pass（应答模块不 import 夹具、无真值/求解器名）、真值打乱 **765/765 字节一致**（max\|Δlogits\| 0.0）、约束解码反查 **690/690 落到已释放反馈**（truth 0、generate 0）；dry-forward 102 轮 logits [15,257] 全有限、F 关时两图 **765/765** 一致、会话恢复 4/4 精确；**159.53 CPU / 164.55 wall**（上限 500 / 4500 s） | owner 2026-11-05 阶段20 rev2 定稿授权（只做 20.1） | 信息门＋无 solver 泄漏＋强基线有动态范围；0 训练 / 0 Adam / 0 confirm / 两图各 1 份 `initialized_untrained` | `verdict=ready`，19 个门全 true；装配态 sha256 CF1 `b62d6fe6…`（16,645 B）、B_EH `64cb327e…`（16,651 B）；**20.2 未准入** | [20.1 报告](../reports/taiji_package20_1_rule_learning_fixture_graph_20261105.json)＋[manifest](manifests/taiji_package20_1_rule_learning_fixture_graph_20261105.json)＋账本 §16.22 |
| **D16-11 重复目标判定**（判据侧，2026-11-04 生效） | **生效（永久）** | 新损失/CE 变体报批前必须先与在跑损失做 0 成本同构核查（公式/输入张量/归约/目标）；同一目标族即退役。首号实例：19.2i ＝ `selection_loss` 同族（仅 τ 1→0.1、步 18→2 的**更弱**差异）→ 退役，省 1,089 CPU | owner 2026-11-04 立纪律 | 同构 → 退役、不占预算 | 账本 §16.5 |
| **D16-12 微调不可用于架构判定**（纪律侧，2026-11-04 生效） | **生效（永久）** | ≤60 Adam 微调只许做行为校准（门控/掩码/温度），**不得**证实或证伪结构性改动；架构问题必须"同初始化、同数据顺序、从零训练、只差一个变量"，并报 Δ＋bootstrap 与可辨识性闸门。依据：19.2g 零梯度、19.2h Δ=[0,0] 两次零结果**均不得读作结构上限** | 同上 | 违反 → 结论作废 | 账本 §16.5 |
| **D16-8 活性检查门**（判据侧，2026-11-04 生效） | **生效（永久）** | loss 干预开训前必须报「触发占比 ≥20%」＋「梯度范数非零」；参数化干预的"触发"读作需修样本占比（19.2h 51/96＝53.1%）；首号实例＝19.2g 的 0/128 激活 | owner 2026-11-04 立纪律 | 两项任一不过 → 禁止开训 | 账本 §16.1 |
| **D16-9 零干预对照臂**（判据侧，2026-11-04 生效） | **生效（永久）** | 任何 ≤60 Adam 小验证必须跑等额 CE-only 对照（同父/同步/同序/同损失）；增益报 Δ＋bootstrap；首号实例＝19.2h（Δ 恰为 [0,0]） | 同上 | 无对照臂 → 增益不可归因 | 账本 §16.2／§16.4 |
| **D16-10 零可识别性判定**（判据侧，2026-11-04 生效） | **生效（永久）** | 干预臂与对照臂逐例读数完全一致且 Δ 区间恰为 [0,0] → `zero_identifiability`；不得读作结构上限/训练不够，禁止靠加剂量救；只允许改参数化或换干预 | 19.2h 实测 | 判定 + 归因（剂量/初始化 vs 插入位置） | 账本 §16.4 |
| 旧能力保持 | 可立保持合同，未实测学后保持 | 已有Task1确认参考父；开发父r2另标工程身份 | 20.1先冻结新保持面、20.2核验完整父包 | 账本§10硬/软两层＋Task2合法历史能力另测 | [包20](reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md) |
| 经验写入面 | 设计中，未在本候选验收 | 20的F反馈绑定/三会话合同 | 20.1正式图及状态清单 | 反馈适应、版本纠正、保持和恢复合取 | 包20§3–8 |
| 18窗口旧待办 | 历史/取消重复调度 | 有效职责由Task1结构资产及20承担 | 不重开18窗口 | 旧负结果保留 | 原窗口报告 |
| **S19 跨session经验与事实修订** | 长期责任，20覆盖有限子集待验 | 尚无本范围任务通过 | 包20三会话/版本协议 | 只按实测范围签收 | 系统计划§6、包20 |
| **S20 反馈塑形/保持/可塑扩容** | 长期责任，20先验反馈F学习 | 生长未启用 | 先有可核验学习父与残差 | 功能贡献/保持/费用 | 系统计划§7 |
| **S21 世界模型/主动活动** | 未调度 | 不以活动次数或空转签收益 | 20成立后执行21研究同任务主动求证 | 真实信息/效用/成本改善 | 系统计划§8 |
| **S22 继承式规模生长** | 未调度 | 启动信用不等于成熟贡献 | 学习父＋可复现容量压力＋同容量对照 | 成熟功能增量/保持/成本 | 系统计划§9 |

## 2. 依赖顺序

1. 本轮20.0只完成已有证据评估、计划收束、任务/算法草案与公共组合枚举。owner最新规划请求替代旧“待命、不写提案不给建议”指令；不借此启动训练或重读封存数据。
2. 20.1夹具与单一图 → 20.2信用/完整保存恢复 → 20.3真实定价与冻结 → 20.4有界对照 → 20.5新独立确认 → 20.6全局收束。任一步不成立按具名出口结案，不继续19.2x补丁链。
3. 旧包20重复抽取/50步训练与旧21纯登记草案被20 rev2吸收，快照归档。Task1继续用作结构保持面，不再用其成绩证明学习收益。
4. 20若成立，执行21才研究同任务中主动求证/重放选择；更大结构生长须有学习父、可复现残差及同最终容量对照。S19–S22是长期能力责任轴，不是执行编号或完成比例。
5. M1/M2/M3按相应方程与控制域保留。Task1确认与r2修复身份不混用，20的F控制不冒用旧19同名臂的归因。仍缺的是学习后保持实测与新反馈图，不再是“没有任何可登记父能力”。

## 3. 单一权威源（每条事实只允许一处正文）

| 事实类别 | 唯一权威文件 | 其他文件只保留 |
|---|---|---|
| 当前状态、队首、待裁定 | 本文件 | 一行指针 |
| 推进轨道（步骤、判据、预算、分支） | [ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md) | 一行指针 |
| 判据阈值、硬不变量、否定定理 M1/M2/M3 | [账本](reference/INVARIANT-LEDGER_20261103.md) | 一行指针 |
| 读数解释边界（数字该怎么读） | [能力卡](reference/CAPABILITY-CARD_20261103.md) | 一行指针 |
| 逐日执行流水与回执 | [03](active/roadmap/03_CURRENT_EXECUTION.md) | 一行指针 |
| OP2 形式化 | [OP2 规范](reference/SPEC-COPY_OPERATOR_OP2_MATH_20261104.md) | 一行指针 |
| 系统能力责任轴（S18–S22） | [系统计划](reference/PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md) | 一行指针 |
| 工作包出口/失败回流细则 | [10](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) | 一行指针 |
| 17 机制细则（已结案） | [17 设计](reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md)＋[SPEC-N16-01](reference/SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md) | 标 `historical` |
| 数据前提定义 | `taiji/data_premise_18_0.py`＋18.0 定价报告 | 一行指针 |
| 证据数值 | `reports/*.json`（只读，不改字节） | 一行指针 |
| 文档索引 | [PLAN_INDEX](active/PLAN_INDEX.md) | 只做指针，不写结论 |

## 4. 冲突消解（唯一正确版本）

| # | 冲突 | 旧版本（标 superseded） | 唯一正确版本 | 证据 |
|---|---|---|---|---|
| 1 | 17 结构分离是否成立 | “内容保真通道独立可控/可分离” | **不成立**（M1：∂η/∂α≠0）；只可作整条选择支路代理信用 | 账本 §12 M1、L4 已改 |
| 2 | γ 门归因 | “门是最大压制/主因是 γ 门” | **错误**：门只放大**已对准**的指针；γ=0 只说明前向质量压在生成侧；按 M3 不得推断信用饥饿 | 账本 §14.4、γ 扫描、18.2 |
| 3 | 硬不变量表述 | “任意记录切换 + 逐位恢复，无条件” | **同一控制范围内**逐位一致；两者不可同时要求（**M2 已裁定 ①**） | 账本 §10/§13 |
| 4 | 逐字节门是否采纳 | “未采纳/只是诊断” | **已于 2026-11-04 正式采纳为任务判据**：物件/地点 ≥0.95、抄回 ≥0.60、整句 ≥0.80；不追溯 18.1；18.2 的 ≥0.95 是**诊断门**，不是任务门 | 账本 §11/§13、§14 |
| 5 | `prefix_cursor` 是否应非零 | “恒 0 是正常状态面” | **错误**：恒 0 是缺陷；18.2 起必须随接受字节递推并进 bundle；读数 A 18/18、B 16/16 | 能力卡 §8、18.2 报告 |
| 6 | 17.4 “任意记录切换中断续推一致” | 无条件跨记录续推 | 限定为同一可达控制范围内的切换一致 | 账本 §13 表 ① |
| 7 | 18 是否已达标 | “18 已接近达标/复制只是没被用上” | **18.1 判负**；18.2 只证明复制通路的结构可达性（gold 记录+gold 起点），不是任务能力 | 18.1 报告、18.2 §14.2 边界 |
| 8 | 旧能力保持是否完成 | 无父、永久不可测或NLL漂移即保持 | **已有受限Task1确认父，学后保持仍未实测**；20还须测试Task2-v1合法历史能力，结构保持不能代替学习保持 | 账本§10、包20 |
| 9 | 19 号编号撞车（跨 session 经验 vs 位置计数包） | “19 = 跨 session 经验” | **包 19 = 位置计数＋UTF-8（含 19.1–19.7）**；系统计划长期阶段改用 **S19–S22** 并冻结调度 | [ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md) §0 |

## 5. 已裁定（由 owner 2026-11-04 授权决定，原“待裁定”文字留痕见本条末尾）

| # | 议题 | 裁定 | 生效范围 | 证据/留痕 |
|---|---|---|---|---|
| 1 | M2 选 ① 还是 ② | **① 保逐位继承**（记录控制限定“同一控制范围内”） | 所有恢复判据、T8、能力卡跨记录表述 | 账本 §13（含原“建议 ①、待裁定”留痕） |
| 2 | 逐字节门是否正式采纳 | **正式采纳**为下一包任务判据：物件/地点 ≥0.95、抄回 ≥0.60、整句 ≥0.80；**不追溯 18.1** | 包 20 起；18.2 的 ≥0.95 仍是诊断门 | 账本 §11/§14、ROADMAP §2 |
| 3 | 19 是否达标后立即训练 | **直接推进**：19.3 定价 → 19.4 预检 → 19.5 六格，不再逐级等放行；任一步不达标按轨道分支 | 包 19；越线（20,000 CPU / 30,000 wall）不停但必须留 `watchline` | ROADMAP §1、03 流水 |
| 4 | “任意记录切换”是否永久放弃 | **永久放弃为能力声明**，降级为上限研究方向；旧表述留痕不删 | 能力卡可验收项与对外声称 | 能力卡“能力声明修订”、账本 §13 |

**留痕**：裁定前本表四项均为“待裁定 + 建议”，其中建议 ①、建议采纳逐字节门、建议不立即训练、建议放弃声明；owner 2026-11-04 将第 3 项改判为“直接推进”，其余与建议一致。

## 6. 盘点（计划/判据/结论文档）

统计（2026-11-04 收敛后实测）：`plans/` 共 **1,063** 个文件（827 md + 232 json + 4 其他）；其中 `reference/` 672（进入当前决策集 8 份 + 本次旁挂 4 份 + 轨道 1 份，其余 659 只作历史证据）、`active/` 20、`manifests/` 218、`archive/` 150；`reports/` 共 **118,077** 个文件，其中 16–18 命名决策报告 **44** 份。

| 路径 | 用途 | 最后更新 | 调度效力 |
|---|---|---|---|
| `plans/MAINLINE.md` | 单入口 | 2026-11-04 | **active（唯一入口）** |
| `plans/MASTER-PLAN.md` | 唯一权威主线（本文件） | 2026-11-04 | **active** |
| `plans/active/roadmap/03_CURRENT_EXECUTION.md` | 执行流水 | 2026-11-04 | active |
| `plans/active/roadmap/02_GATES_AND_CI.md` | CI/门禁规则 | 2026-10-30 | active |
| `plans/active/roadmap/01_SCOPE_AND_PHASES.md` | 范围/阶段 | 2026-10-30 | active（范围） |
| `plans/active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md` | 工作包出口/回流 | 2026-11-01 | active |
| `plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md` | 旧主线计划 | 2026-10-30 | **superseded → 本文件** |
| `plans/active/roadmap/04_EXECUTION_PLAN.md` | 旧路径兼容入口 | 2026-10-22 | active（别名，不发布队首） |
| `plans/active/roadmap/05_TECH_DEBT_REGISTER.md` | 技术债登记 | 2026-10-30 | active（登记） |
| `plans/active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md` | 旧决策论证 | 2026-09-15 | historical |
| `plans/active/roadmap/07_MINI_MODEL_DELIVERY.md` | 交付计划 | 2026-10-30 | active（长期） |
| `plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md` | 上游同步 | 2026-10-30 | active（工具） |
| `plans/active/roadmap/11_CI_FLOAT_EVIDENCE_FOR_OWNER_20261025.md` | CI 浮点证据 | 2026-10-25 | historical |
| `plans/active/roadmap/12_LINE2_EVIDENCE_CLOSEOUT_20261025.md` | Line2 结案 | 2026-10-25 | historical |
| `plans/active/PLAN_INDEX.md` | 文档索引 | 2026-11-04 | active（只做指针） |
| `plans/active/TAIJI_CORE_REQUIREMENTS.md` | L1–L14 需求 | 2026-10-31 | active |
| `plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md` | 原生架构 | 2026-10-31 | active |
| `plans/active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md` | 长期发展（约束） | 2026-11-01 | historical（长期约束，不调度） |
| `plans/active/ARCHITECTURE_DIRECTION_2026_08.md` | 旧方向 | 2026-08 | historical |
| `plans/active/SEED_ARCHITECTURE.md`、`SEED_DEVELOPMENT_ROADMAP_2026_08.md` | 项目背景 | 2026-08 | historical |
| `plans/reference/INVARIANT-LEDGER_20261103.md` | 判据/定理 | 2026-11-04 | **active（判据唯一源）** |
| `plans/reference/ROADMAP-NEXT_20261104.md` | 推进轨道（19/20/21 包） | 2026-11-04 | **active（轨道唯一源）** |
| `plans/reference/CAPABILITY-CARD_20261103.md` | 读数边界 | 2026-11-04 | **active（解释唯一源）** |
| `plans/reference/SPEC-COPY_OPERATOR_OP2_MATH_20261104.md` | OP2 形式化 | 2026-11-04 | active |
| `plans/reference/PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md` | S18–S22能力责任 | 2026-11-03 | active |
| `plans/reference/REVIEW-PACKET_20261102.md` | 17 时代评审 | 2026-11-02 | **superseded**（17 已结案） |
| `plans/reference/CONSOLIDATION_20261103.md` | 上次收敛 | 2026-11-03 | **superseded → 本文件** |
| `plans/reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md`、`SPEC-N16-01_…DRAFT_20261103.md` | 17 机制细则 | 2026-11-03 | historical（17 死） |
| `plans/reference/PRICING-17_3_SELECTION_ROUTING_20261103.md` | 17 定价 | 2026-11-03 | historical（价随图改作废） |
| `plans/reference/ATTRIBUTION-ARCH-VS-ALG_20261103.md`、`MATH-REVIEW_…_20261103.md`、`DESIGN-CANDIDATE-COMPOSITION_20261101.md` | 归因/数学/14–16 设计 | 2026-11-03 前 | historical |
| `plans/reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md` | 执行20 rev2设计与详细子卡 | 2026-11-05 | active（草案，未训练准入） |
| `plans/reference/` 其余 612 个文件 | N1 及更早 | ≤2026-11-02 | historical（不调度） |
| `plans/manifests/` 16–18 预注册（18.0 定价、18.1 preflight/compare、18.2、17.2R identity 等） | 冻结判据/价格 | 2026-11-04 | **active（冻结，不得改字节）** |
| `plans/manifests/` 其余约 200 个 | 旧窗口 | ≤2026-11-02 | historical |
| `plans/archive/` 150 文件 | 历史快照 | ≤2026-11-03 | archived |
| `reports/taiji_selection_routing_18_1_compare_20261104.json` | 18.1 正式判决 | 2026-11-04 | **active（判负证据）** |
| `reports/taiji_cursor_utf8_18_2_20261104.json`（＋A/B 逐例） | 18.2 验证 | 2026-11-04 | **active** |
| `reports/taiji_data_premise_18_0_pricing_20261104.json` | 18.0 定价 | 2026-11-04 | **active** |
| `reports/taiji_preflight_18_1_20261104.json` | 18.1 预检 | 2026-11-04 | active |
| 其余 16–18 报告（44 份，按 `_16/17/18_` 命名） | 探针/预检/结案 | 2026-11-04 | 证据（只读） |

## 7. 收敛与清理记录

- 原 `MAINLINE.md` 正文（405 行 / **177,021 字节**）**逐字节归档**至 [archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md](archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md)，`MAINLINE.md` 只保留单入口与锚点桩。
- 新增旁挂（不改旧字节，共 **1,356 字节**）：`CONSOLIDATION_20261103.md.superseded.json`（317）、`REVIEW-PACKET_20261102.md.superseded.json`（373）、`09_NEXT_MAINLINE_PLAN.md.superseded.json`（282）、`PRICING-17_3_SELECTION_ROUTING_20261103.md.superseded.json`（384）。
- 删除：**0 个文件 / 0 字节**。现有“死草案”均为被引用的兼容入口或历史证据，按“保留全部判定/判负报告与证据 bundle”处理；无引用且方向已死的独立草案本次未发现。
- 新增唯一轨道 [ROADMAP-NEXT_20261104.md](reference/ROADMAP-NEXT_20261104.md)（**11,395 字节**）：19/20/21 包步骤表、六项字段、预算上限、决策树与本轮不做清单。
- 合计新增字节：177,021 + 1,356 + 11,395 = **189,772 字节**（1 个归档文件 + 4 个旁挂 + 本文件 + 轨道文件）。

## 8. E1–E11数学审查输入（2026-11-04）

本节保留数学审查交付时点的设计来源；其中当时队首/预算语句不再调度，当前阶段见§9和页首。

[14项意见的后续裁定](reference/MATH-ADJUDICATION_CORE_GRAPH_20261104.md)已纠正全域rank≤2、独立注入轴必要性、重复复制不可表达、KKT矛盾及OP2合法域的推广。生长保留“启动不等于功能增量”的强限定。统一在线目标继续作为候选，不由本次审查升为唯一v1.3默认。数学主张登记在[账本§17](reference/INVARIANT-LEDGER_20261103.md#17-e1e11数学审查的主张边界)。

设计后继M-GRAPH-1在[系统计划§17](reference/PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md#17-e1e11数学裁定与单一候选图规格)，先闭合单一候选图、父可塑集合、信用/保持单位和约束更新。此处只登记数学设计责任，不另发布训练队列、不改变19.2j已到达的价格出口或请求预算。

后续旧field/fabric附件按[数学裁定§9](reference/MATH-ADJUDICATION_CORE_GRAPH_20261104.md#9-补充附件旧fieldfabric底座的容量与统计界)分轨处理，撤回容量/JL/归一化的错误定量迁移。它不构成E1–E11新结论或新的执行队列，不改变M-GRAPH-1及当前价格出口。

## 9. 2026-11-05最新进度评估与阶段20

Task1能力与父身份以能力卡C12/包20§2为准；19.2v官方确认188/192，r2只挂工程修复证据。父资产必须包含根bundle、65,807参数的匹配模块装配态、停止协议和版本，不把265KB装配文件当完整主体。参数总量先分项，20.1实际去重后再冻结。

包20的任务、CF1反馈写入与独立无F基线、51轮分母/22首答、错误与纠正预算见[详细计划](reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md)；[草案manifest](manifests/taiji_package20_learning_correction_draft_20261105.json)明确training_admitted=false，数据/价格/候选身份尚未物化。公共规则静态预审见[回执](../reports/taiji_package20_plan_review_20261105.json)，不能当模型能力。

本轮没有读取模型权重或新旧封存数据，没有模型前向、优化器或运行替换；既有工程/训练成本保留原记录，19.2j缺失CPU不补0。旧队首与重复待办逐字节归档，不删除历史失败或冻结证据。
