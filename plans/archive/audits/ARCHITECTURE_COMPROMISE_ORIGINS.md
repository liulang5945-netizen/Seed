# 架构妥协审计：早期机制与历史状态

> 本文由原总路线图按职责拆分而来。原始行号：1–529；本文件属于当前计划资料或历史证据，具体身份以本文件所在目录为准。
> 这是原审计的早期机制、状态和历史记录部分。

## 原始架构妥协点审查报告

> 梳理整个项目中所有"为了易于实现采取的妥协方案"，按上限损失严重性排序。
> 每个妥协点给出：当前实现 → 妥协原因 → 上限更高方案 → 提升幅度。
>
> 调研范围：共振场核心 + 训练流水线 + 推理运行时，共 90+ 妥协点。
> 本报告聚焦**系统性妥协**（影响全局上限），局部小妥协见归档。

---

## 📌 人脑动态神经元：热插拔三机制（2026-08-06 实施中）

**动机**：用户提出"人脑神经元动态增加/减少不影响工作，可正常对话输出，且可多线程处理不同任务"——态极需支持运行时增删神经元而不打断推理，并支持多任务并行。

| 人脑机制 | 态极实现 | 状态 |
|---------|---------|------|
| 神经元增删不影响工作中推理 | **快照隔离**：`ensemble.forward/forward_train` 入口 `nmap = dict(self.neurons)` 浅拷贝，全程用快照；增删只改原 dict，被删 neuron 在快照中仍持引用不崩 | ✅ 已完成（commit bfccaf8） |
| 增删互斥防交错 | **热插拔锁**：cortex 增删接口（`add/remove/isolate/revive_neuron`）用 `threading.RLock` 串行化；推理读走快照不拿锁（读-写无争用） | ✅ 已完成（commit bfccaf8） |
| 混合规格种群共存 | **热插拔补投影**：`ensemble.add_neuron` 在 `field_dim ≠ unified` 时自动补建 `CrossSpecProjector`（正/反向）——修既有 bug（混合种群下新 neuron 缺投影层 → 推理 3072-vs-2048 RuntimeError）；同时**放宽 field_dim 校验**（原只对首个 neuron 校验，混合种群会误拒） | ✅ 已完成（commit bfccaf8） |
| 训练与推理并发（学习时正常对话） | **训练/推理分离（影子权重 COW）**：推理（generate/generate_multimodal）**去训练锁**（原 acquire(timeout=10) 阻塞对话）；训练侧 `_train_cortex_neurons` 在 `_clone_module` 克隆副本上训练（live 权重训练全程稳定），结束一次性写回 + 恢复引用；`_clone_module` 用配置重建 + load_state_dict（deepcopy 因 RotaryEmbedding._cache_lock 不可 pickle 会崩） | ✅ 已完成（本 commit） |
| 多线程并行处理不同任务 | **任务级并行**：① ensemble `field` 改属性——推理 forward 期间返回 thread-local 独立共振场（`_get_task_field`，继承 W_cond + gamma gate，跨任务互不污染）；② forward scratch（round_scores/_router_*/_logits_keep_ids）全部 thread-local（`_fstate`），写穿 `_last_forward_round_scores` 供 sleep_engine 分裂选择；③ forward_train 显式用默认场；④ API SSE 聊天 `asyncio.to_thread` 移同步 generate 出事件循环（并发请求真正并行）；`active_nids` 按域路由不同 neuron 子集（已有） | ✅ 已完成（本 commit） |

**快照隔离已验证**（`verify_hotswap_snapshot.py` 全通过）：
- 推理线程持续 forward，主线程并发 add/remove/isolate/revive → 推理 8 次 forward 全部正常、分数有限
- 混合规格热插拔：field_dim=256 neuron 加入 field=512 → 投影层自动补建（in=256/out=512 断言）
- 隔离/复活语义：pop 保留引用可复活，复活后推理正常
- 回归：旧 3072-vs-2048 崩溃场景（静态混合规格推理）分数有限

**训练/推理分离已验证**（`verify_train_infer_separation.py` 全通过）：
- 训练周期全程 live 权重完全稳定（推理读到稳定权重）——影子隔离核心契约
- 写回后 live == 影子（训练生效）+ 训练期间移除的 neuron 不复活
- 推理线程 25 次 forward 全部正常（训练 COW 周期全程不崩溃、分数有限）
- dict 引用不变（ensemble.neurons is cortex.neurons 贯穿全周期）

**任务级并行已验证**（`verify_task_parallel.py` 全通过）：
- 三线程 barrier 同步并发推理，全部不崩溃、分数有限
- 路由隔离：每任务 final_scores keys 只含本任务 active_nids（跨任务无污染）
- 并发 top-1 与串行基线完全一致（field 隔离生效；neuron 不应期共享为符合人脑语义的微小调度差异）
- task field thread-local 缓存复用断言

**真实集成验证**（`verify_hotswap_integration.py` 全通过，真实 zh_general compact 权重）：
- 三线程并发真实推理 ✓；推理线程中热插拔（同规格 compact + 跨规格 1024→2048 投影层自动补建）✓；隔离/复活 ✓；真实 COW 周期（live 稳定 → 写回生效）✓
- **已知限制（待讨论）**：`ensemble.add_neuron` 保留 hidden_size 校验——standard(768)/compact(512) 混合 hidden 种群会被拒。当前生产统一 compact 无实际影响；若未来引入 expert(1024)/foundation(384) 混合种群，需评估 forward 路径（embed_adapter 各自投影 → 理论上兼容）后放宽

**并发容错补充**：`_update_channel_usage` 两处 `post_neuron` 改 `.get()` + None 跳过（推理中 side_channel 清理的 post 神经元可能已被移除）。

---

## 📌 弱神经元剔除（凋亡）v2：人脑分层凋亡（2026-08-06 重构，commit 6efd24f）

**动机**：原 ApoptosisTracker 固定 PPL>200 + 激活率<5% 两个绝对阈值——① general 256K 空间与域空间 PPL 口径完全不同（固定 200 会误杀全部 general 空间 neuron，与当前基座训练直接冲突）；② 均匀激活假设与"5 联合 > 5"分工路由冲突（域 neuron 只在自己域激活是设计意图）；③ 永久删除无恢复。

**人脑机制映射**：

| 人脑机制 | 态极实现 |
|---------|---------|
| 突触修剪先行（Synaptic Pruning）| 弱 side_channels（`_channel_usage` 长期低）先被修剪，神经元本体保留（`prune_synapses`）|
| 活动依赖存活（use it or lose it）| 激活率作为生存信号，种群相对归一化 |
| 神经营养竞争（营养=网络贡献）| 协作边际贡献（A/B 剔除实验）+ 网络中心度（side channel 出入度）入生存分 |
| 凋亡级联（启动→执行→清除）| active → candidate → isolated → trial → dead，多阶段可取消/可复活 |
| 成熟度保护 | 幼稚态（maturity_ratio<0.5）不参与凋亡竞争 |
| 抑制性神经元保护 | inhibitory 不把"网络贡献"作为存活要求（权重转移给活动+能力）|
| 空间自适应 | PPL 用**种群内百分位**（同空间内排序），不跨空间比绝对值 |

**v2 实现（已验证）**：
- [lifecycle.py](file:///e:/taiji-neuron/taiji/resonance/lifecycle.py) `ApoptosisTracker`：多维生存评分（activity/ppl 分位/contribution/connectivity/redundancy 惩罚，缺失自动降权）+ 状态机（`step_population`）+ `prune_synapses` + `cleanup_neuron` 改移入 `_recycle_bin`（不直接删）
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py)：`isolate_neuron`（摘除路由保留 ckpt）/ `revive_neuron`（重载 ckpt 复活）/ `get_isolated_neurons`
- [sleep_engine.py](file:///e:/taiji-neuron/taiji/life/sleep_engine.py) Phase 4：采集多维信号（activity/ppl/connectivity/maturity/inhibitory）+ 级联动作（隔离/试复活/复活/凋亡补偿新生）+ 突触修剪
- 兼容：`record_ppl`/`check_activation`/`get_apoptosis_candidates` 保留（verify_apoptosis.py 全通过）

**验证**：状态机全链路 active→candidate→isolated→trial→dead ✓；复活 ✓；成熟度保护 ✓；抑制性保护（exc 0.31 vs inh 0.80）✓；**general 空间安全（4 域 PPL 全 >200 仍全 active，旧固定阈值会误杀）** ✓；突触修剪 ✓；旧接口兼容 ✓

**待注入信号（评估基础设施就绪后）**：contribution（协作 A/B 剔除实验）、redundancy（field_vector 相似度）

---

## 📌 递归设计检查（2026-08-06 补充）

**结论：项目"递归"（taiji/life 递归改进 + 递归蒸馏）两条回路——任务进化回路活着，递归改进回路是死的。**

**✅ 已修复（commit 7e3a3ea）**：
1. **删除废弃死代码**：`recursive_improver.design_next_generation` 及 13 个私有辅助（~300 行）、`evolution_engine.execute_generation_transition`/`_design_to_model_config`/`_validate_student`/`EVOLUTION_PATH`/`_get_current_generation`/`_get_next_generation`
2. **修潜伏 bug**：`_get_next_gen_name` 双同名定义（L623 被 L740 覆盖 → design_next_generation 必 TypeError）——方法已随死代码删除
3. **激活死配置**：`loss_plateau_steps` 从定义未用 → `record_sleep_training` 记录 loss 历史 + `check_evolution_ready` 平台检测（std < 0.05）
4. **接入真实调用方**：`chat_strategies._record_recursive_strategies` 每次推理记录 prompt/tool_choice/reflection 策略到 RecursiveImprover（输入环复活）；功能验证：12 条记录 → 产出 2 条提案（prompt + tool）
5. **sleep_engine Phase 5 重构**：design_next_generation 调用 → 生成 `next_training_data_spec.json`（训练数据建议，消费方 = 跨域协作层训练）
6. **防噪**：prompt 模式提取过滤 >80 字符整句（中文无空格分词噪声）

**修复前问题（已解决）**：
- 输入环断开：`record_strategy` 零调用方 → 分析恒空
- 输出环废弃：design/execute 均 deprecated/NotImplemented
- 单体变大叙事残留（0.5B→7B）与 BODY_LIFE 决策 3 冲突

---

## 📌 态极递归流程（神经元架构下，2026-08-06 定义）

```
                        ┌────────────────────────────────────────────┐
                        │ 推理时（循环起点）                           │
                        │ chat_strategies：ReAct + 工具 + 直接生成     │
                        └───────┬────────────────────────────────────┘
                                │ 同时记录两路
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   ┌─────────────────────┐              ┌──────────────────────┐
   │ 策略环（输入）        │              │ 任务环（输入）         │
   │ record_strategy      │              │ record_task_success/  │
   │ prompt/tool/reflection│             │ failure               │
   └──────────┬──────────┘              └──────────┬───────────┘
              ▼                                    ▼
   ┌─────────────────────┐              ┌──────────────────────┐
   │ RecursiveImprover   │              │ EvolutionEngine      │
   │ 策略记录（jsonl）     │              │ 成长值/失败率/知识饱和/ │
   └──────────┬──────────┘              │ loss 平台             │
              │ 睡眠 Phase 5             └──────────┬───────────┘
              ▼                                    │ check_evolution_ready
   ┌─────────────────────┐              ┌──────────▼───────────┐
   │ 策略改进提案         │              │ 能力扩展信号（ready）  │
   │ analyze_and_improve │              │ ① neurogenesis 信号   │
   │ → EventBus +        │              │ ② 训练数据建议         │
   │   best_strategies   │              └──────────┬───────────┘
   └─────────────────────┘                         ▼
                                        ┌──────────────────────┐
                                        │ next_training_data_   │
                                        │ spec.json（弱点+建议） │
                                        └──────────┬───────────┘
                                                   ▼
                                        ┌──────────────────────┐
                                        │ 协作层训练（数据环）    │
                                        │ train_multi_domain_  │
                                        │ foundation.py（基座）  │
                                        │ train_cross_domain_  │
                                        │ collab.py（协作层）    │
                                        └──────────┬───────────┘
                                                   ▼
                                        ┌──────────────────────┐
                                        │ 能力扩展             │
                                        │ 新 neuron / 更强协作  │
                                        │ 5 联合 > 5（涌现实证） │
                                        └──────────┬───────────┘
                                                   │ 再次投入使用
                                                   ▼
                                              回到推理时（递归）
```

**三环说明**：
1. **策略环**（推理时 → 睡眠分析）：记录实际使用的 prompt/工具/反思策略 → 睡眠时统计成功模式 → 低效工具降权提案（`web_fetch 成功率 0%` 实测产出）
2. **任务进化环**（持续）：任务成功/失败 → 阶段升级（infant→adult）→ neurogenesis 信号（睡眠创建新神经元）+ 进化报告
3. **数据环**（睡眠 → 训练，递归核心）：`check_evolution_ready` 满足 2 条件（成长值/失败率/知识饱和/loss 平台）→ 生成 `next_training_data_spec.json` → 跨域协作层训练 → 能力扩展 → 重新投入推理

**废弃（已删除）**：代际变大（0.5B→7B）、design_next_generation、execute_generation_transition——神经元架构的递归 = 协作层数据闭环，非整体替换模型。

---

## 📌 文档维护规则（2026-08-07 新增，解决"过时信息干扰"）

1. **唯一权威**：本文档以"**当前执行状态**"段为唯一当前事实源。以下所有"历史状态"段落均为归档参考（保留演进记录），**不以它们为准**。
2. **过时即覆盖**：任何"下一步/待决策"叙事若与"当前执行状态"冲突，以最新为准（例：~~推理侧 division 分工路由~~ 已被"域判别路由 loss"方案替代）。
3. **误用即修复**：代码层面"机制说明不足"的修复（如融合主路径收敛、函数状态标注、训练可复现配方）已直接写在代码 docstring 中——**代码注释是机制的第一权威**，本文档记录决策背景。
4. **训练命令**：可复现配方记录在 `train_cross_domain_collab.py` docstring，不在本文档维护（避免双源漂移）。

---

## 📌 当前执行状态（2026-08-08 更新）

**"共享 general lm_head 统一输出空间"：基座训练完成 ✅ + 产物验证 ✅ + 路由适配完成 ✅**

**架构落地（已完成，commit 5bb522b 等）**：
- ✅ [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py#L160-L164)：`ResonanceNeuron.__init__` 支持注入 `shared_lm_head`——所有 neuron 共享同一个 general 256K lm_head，直接预测通用 token（无词库转译投影稀释）→ 路由置信度信号保留
- ✅ [train_multi_domain_foundation.py](file:///e:/taiji-neuron/scripts/training/train_multi_domain_foundation.py)：`--target-space general` 完整适配
  - `create_shared_lm_head()`（512→256K，131M params）；neuron 创建注入共享 head
  - general 模式：输入/目标都 general 编码（`batch_align_and_embed(batch, general_sp, general_sp, emb)`），全文本 loss（无 answer marker）
  - 优化器三分：neuron 主体 lr / embedding 独立低 lr / **共享 head 独立 lr（--lm-head-lr）**
  - checkpoint 瘦身：`_strip_shared_head()` 剥离 131M head 出 neuron ckpt（共享 head 独立存 `shared_lm_head.pt`，避免每域 ~525MB 冗余）
  - 回读验证支持 `lm_head_path` 注入（最终 + 配对校验两段都传，防止 16K 域 head 算 256K 目标崩溃）
- ✅ **冒烟验证**（3 步/域 × 4 域，batch 2）：管线跑通 + checkpoint 回读正确；loss 从随机水平（12.9-13.2 ≈ ln256000）三轮内降到 10.3-12.5，**学习正常**
- ✅ **正式基座训练完成**（21:23）：`data/foundation_v1_general`（4 域 × 600 步 = step 2400，batch 8），`foundation_history.json` 完整记录。**注意：4 个域 neuron 是全新随机初始化训练（compact，非继承旧 5 神经元/旧域基座）**——统一空间要求所有 neuron 直接预测 general 256K token，旧域 lm_head（code 12K 等）输出空间不一致无法复用
- ✅ **collab/eval 加载路径预修复**（commit 5bb522b）：`load_neuron` 支持注入 `shared_lm_head` + 新增 `load_shared_lm_head()`——general 基座（ckpt 已剥离 131M head）可正确加载并输出 256K 空间 logits

**产物回读验证通过**（`verify_foundation_general.py`，修复 tokenizer 口径后）：
- **回读 PPL**（best ckpt + 最终 embedding，collab/eval 口径）：**code 6.1 / math 21.1 / zh 274.9 / en 178.2**——code/math 健康，zh/en 因 general 256K 空间全文本 CE（无 SFT masking）偏高，属预期口径（不是坏 ckpt）
- **配对校验**（best 权重 ↔ best 步 embedding）：code 14.1 / math 41.5 / zh 514.7 / en 187.7
- **输出空间验证**：4 域全部输出 general 256K logits，top token 正常（code '<0x0A>'、math '4'、zh '、'、en '.'）

**🔧 修复训练脚本回读 bug（commit 00cda08）**：`verify_checkpoint` 目标 tokenizer 传错——L335/L341 传 `domain_sps[d]`（域词表），与训练（general_sp 编码）不一致 → 目标编码错位，训练脚本自身打印的"最终回读" PPL 是假值（400 万级）。已改为 `general_sp, general_sp`（输入/目标都 general，与训练循环 L267 完全一致）。

**✅ 路由适配统一空间完成**（`verify_unified_space_routing.py`，commit 4dc5db3）：
1. **新增 division 融合模式**（[ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py) `_division_logit_fusion`）：同 vocab（256K）场景 per-position 硬路由——每个 token 位置交给 max-prob 最高的 neuron，直接复用原生置信度（无转译投影）。`division_norm` 为 per-neuron 归一化变体（实验）
2. **修复架构 bug**：`forward` 的 active_ids 原为 set（迭代顺序不定）→ round_logits/all_logits dict 顺序随机 → 路由监控/诊断索引错位。改为有序 list（保持 nmap 顺序），路由信号顺序确定性
3. **验证结果**：
   - [2] 域内分工全对角占优：code 文本→code 0.66、math→0.54、zh→0.46、en→0.34（分工方向正确，路由信号在统一空间可用）
   - [3] 基线 PPL（协作层未训练时）：个体最优（code 19.5/math 56.3/zh 82.6/en 108.5）> soft（共振分融合）> division 裸 > division_norm——**跨 neuron 的 logits 尺度天然不可比**（code 平均 max-prob 0.566 vs zh 0.374，尖锐者抢走非自身域位置但预测错误），启发式归一化无效（过度平坦化）
   - [4] 生成冒烟正常（中文 prompt 回显 + 多 neuron 续写）
4. **关键结论**：division 分工路由是"路径打通 + 信号可用"的适配（域内分工正确），PPL 校准需**协作层训练**（C12 对比约束让共振分对齐 NLL 排序 / Sparse Router 学习路由）——即下一步

**✅ 混合阵容协作层训练支持完成**（train_cross_domain_collab.py，commit c3a6a68）：
- **回应核心设计**：不同词表 neuron 通过**词库转译**协作是既有核心能力（EMERGE +4.0% 实测），统一空间（共享 lm_head）是演进而非否定。协作层训练**同时包含旧 5 对话 neuron + 新 4 general neuron**
- `--dialogue-ids/--dialogue-dir/--dialogue-max-texts/--dialogue-data-dir`：旧对话 neuron（zh 50K 域空间）加入阵容——保留各自域输出头（跨 vocab 协作），输入用 ckpt 内 home embedding（shared_embedding_state），经词库转译矩阵（zh 50K→general 256K，sparse.mm）投影到统一目标空间
- dialogue 数据桶（data/simple_zh，短答案 ≤150 字）加入域轮转，general 目标空间统一编码
- 日志频率 %50→%10（原 epoch_ppl 用 loss_history[-1]，无记录时假打印 PPL=1.0）
- **冒烟验证通过**（9 neuron 含 standard/compact 混合规格 + 转译 + 融合 + loss 下降 + checkpoint 可加载）：43 step / 7.1min，math loss 5.82 → dialogue 7.28（下降中）；PPL 高是协作层未训练自然状态
- **性能基准**：batch1×seq32 ≈ 10s/step（CPU + 256K vocab 固有成本）；正式训练建议 batch2×seq64 ≈ 40s/step

**后续链（当前：域判别路由 loss 重训中）**：
1. ✅ ~~路由适配统一空间~~（division 模式 + set 顺序修复 + 基线记录）
2. ✅ ~~混合阵容支持~~（旧 5 + 新 4 词库转译联合训练管线，冒烟通过）
3. ✅ **正式协作层训练完成**（collab_v1_mixed.ckpt.pt，540 step × 2 epochs ≈ 1.7h，9 neuron 混合阵容）
4. ✅ **负 EMERGE 根因诊断 + 评估口径修复**（2026-08-07，verify_collab_mixed.py）：
   - **评估口径修复**：① cap 加错函数（ensemble 而非 solo）→ 移正；② 无固定 seed + n_eval=8 → 采样噪声致结果不可复现（zh solo 855↔43364 波动 50 倍）→ `random.seed(42)`；③ `load_dialogue_texts_multi` 用 `list(set())` 去重 → set 迭代顺序跨进程随机 → 改插入序去重
   - **scores 诊断（关键结论）**：`result["scores"]`（LOO cosine 共振分）排序——code 文本 top1 是 **math**（-0.418 > code -0.429）、math 文本 top1 是 **zh**、zh 文本 zh 反而最低；且所有 scores 为负且极接近（差 0.02-0.05）→ `trust=softmax(scores/0.15)` **几乎无区分度 → 校准实际失效** → 路由退化为 max-prob → 弱 neuron 抢位（code 文本 code 仅 0.700，math 抢 0.200）
   - **body 微调无损确认**：协作层训练前 code solo（8 条随机 general 空间）45 → 训练后 23.99（verify_unified_space_routing L147 注释佐证）——训练有效，负 EMERGE 纯粹是路由校准问题
   - **上界实验（[4b]，trust_override 硬门控）**：已知域硬门控下协作追平 solo（code -18.0%→**-1.4%**、math -6.8%→**-0.7%**、zh +0.5%→**+1.4%**、en -4.0%→**-0.2%**）——**路由是唯一瓶颈，修复路由即可消除负 EMERGE**；且门控下协作 ≈ solo（域内任务协作目标是"不伤害"，跨域才是涌现价值）
5. ✅ **域判别路由 loss v1 已训练（270/540 步，中断半成品）**（collab_v2_routing，2026-08-07）：
   - 方案：训练时显式约束"本 batch 域 neuron 的共振分 scores 最高"——`routing_loss = -log_softmax(scores/0.15)[domain_idx]`（温度与推理 router_temperature 一致），`--routing-loss-weight 0.5`，仅 4 个 general 域生效（dialogue 桶无单一目标 neuron 跳过）
   - 理由：scores 是 LOO cosine（共振协调度），数据少时学不到"域内 neuron 应胜出"→ 显式注入域判别目标比加大数据量（10× 训练时间）更高效直接
   - ensemble.py 新增 `forward_train(trust_override=...)` + `_confidence_routing_fusion(trust_override=...)`（上界诊断用，向后兼容 None）
   - 训练配置与 v1 一致：batch2×seq64×max_texts100×dialogue150×2epochs≈540 step（~1.7h CPU）
6. ✅ **routing_loss 未生效诊断 + C13 修复（2026-08-07）**：
   - **第一层：训练中断（执行层）**——日志显示训练实际跑到 step 360+（进入 E2），但最终 ckpt 只保存到 **270 步**（E1 结束保存点；训练脚本每 500 步 + epoch 结束才保存）。**评估用的是半成品 ckpt**，计划 540 步只完成一半。训练进程无报错突然中断（疑似手动/超时被杀）
   - **第二层：LOO 梯度泄漏（机制层）**——`scores[n] = cosine(vec_n, normalize(field_state - vec_n))`，`field_state` 含全部 9 neuron 的 vec → 提高 code 的 score 时梯度也流到其他 8 个 neuron。实测：RL→其他 8 neuron 梯度 L1=57.5，是 code 自身 15.2 的 **3.8 倍**。尝试 detach field_state 无效（泄漏反而升到 125.8）——**真正泄漏源是 round 2 场条件化注入**（neuron 第 2 轮输入 fs=field_state 携带所有 neuron round 1 的梯度）
   - **第三层：结构性偏向（根因）**——code 文本 scores 排序 `en 0.365 > math 0.155 > code 0.138`。en 是 general 空间高锐度 neuron，其 vec 主导场方向 → **LOO cosine 天然偏向强 neuron**（C12 score_dim 设计解决但未启用）
   - **C13 修复（用户决策：per-neuron 域判别 head，上限最高）**：每个 neuron 新增 `domain_score_head: Linear(hidden→1)`，round 1 独立前向（无场注入）输出"当前样本属于本 neuron 域"的 logit；routing_loss 与推理 trust 都改用它（softmax(domain_logits/temp)）。round 1 独立 → 梯度只流向自身 + softmax 分母的合理负监督
   - **修复验证**（diag_routing_gradient.py）：① domain_logits 排序正确（code 文本 code=0.87 第一，en=0.44 第二，随机 head 已有域判别力）；② 错误泄漏消除（RL→其他 8 neuron 从 125.8 降到 8.37，剩余为 softmax 分母的合理负监督）；③ CE 梯度增强（4.1→15.9，code 被正确路由选中）
7. ✅ **collab_v3_c13 完整训练 + 评估（2026-08-07 完成）**：
   - **训练**：540 步 × 2 epochs 完整完成（E1 PPL≈1174 50.9min → E2 PPL≈177.6 48.4min），`data/neurons/collab_v3_c13.ckpt.pt`（epoch=2, total_steps=540）
   - **域内 EMERGE（8 文本/域，seed 42 固定）**：

     | 域 | v2 | C13 | 门控上界[4b] |
     |----|-----|------|------|
     | code | -43.1% | **-17.7%** | +0.7% |
     | math | -21.6% | -28.2% | +0.4% |
     | zh | -40.3% | **-19.0%** | -3.5% |
     | en | -0.4% | **+0.7%** ✅ | +0.4% |
     | dialogue | — | **+67.1%** ✅ | — |

   - **结论**：① C13 方向正确——code 改善 25.4pp、zh 改善 21.3pp、en 转正，domain_logits 路由显著优于 LOO cosine；② 负 EMERGE 未完全消除（code -17.7%/math -28.2%/zh -19.0%）；③ [4b] 硬门控上界 code/math/en ≈ +0.5% → **融合层无损，剩余负值是路由仍不完美**（soft 融合 vs 门控差 18-29pp）；④ **zh 例外**：门控下仍 -3.5%（PPL 512 vs 415）——zh 域融合/转译本身有损，独立于路由；⑤ dialogue +67.1% 大幅正向（协作对对话域显著增益）
   - **遗留问题**：① 路由精度仍有提升空间（门控上界未达）；② zh 域门控 -3.5% 需单独排查（融合/转译损耗）；③ math 域 C13 后 EMERGE 反而微降（-21.6→-28.2），domain_logits 对 math 判别力不足
8. ✅ **路由错误模式诊断（2026-08-07，diag_route_errors.py，seed 42 逐文本分类）**：
   - **汇总**：code 正常 6/路由错误 2；math 正常 3/**路由错误 5**；zh 正常 3/**路由错误 5**；en 正常 8；dialogue 门控口径不适用（5 个均匀 gate 融合爆炸）
   - **根因 1：判别器以"表面语言"为域信号，而非"域语义"**——math 文本（英文 GSM8K）5/8 判给 en（en logit 0.29-1.25 vs math -0.30-0.26），soft PPL 暴涨 10 倍（128-446 vs gate 29-106）；code 域 2 条 Java/自然语言文本也判给 en
   - **根因 2：en head 系统性高偏置**——en 对纯中文 zh 文本也给 0.79-1.26 高分（与 en 文本上的 0.67-1.47 无差），说明 en head 学到的主要是偏置、权重贡献小 → softmax 温度 0.15 极尖锐 → zh batch 上 en 一旦不是赢家 softmax≈0 → **非赢家梯度消失，en 高偏置压不下去**
   - **zh 门控也差（gate 1436-5618 vs solo 996-5306）**：zh 域融合/转译本身有损（与 [4b] -3.5% 一致），独立于路由，需单独排查
9. ✅ **C14 实施 + 训练 + 二次诊断（2026-08-08）**：
   - **实施**：① domain_score_head 升级 MLP(hidden×2→128→1)+GELU+mean/max 双 pooling（[neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py)）；② routing_loss 温度参数化（--routing-loss-temp，0.15→1.0）
   - **c14 训练完成**（540 步，E1 PPL≈419.6 优于 c13 的 1174，E2 PPL≈137.2）→ 评估：code -17.7→**-11.6%** ✓、zh -19.0→**-6.9%** ✓、dialogue +67.1→**+71.2%** ✓、en +0.2%、**math -28.2→-60.2% ❌ 恶化**
   - **c14 二次诊断（diag_route_errors）**：math **8/8 路由错误**、code 4 路由错误（c13 为 2）；zh 7/8 正常（c13 为 3/8）、en 8/8 正常。关键证据：math 判别器给分不低（0.15-1.02）但 en 总高 0.5-1.0（1.25-1.90）——**根因：判别器 lr 太低**（domain_score_head 属 body 走 body_lr_ratio=0.1 → lr=1e-4），MLP 131K 参数 540 步远不够收敛；zh 靠语言粗信号勉强学到、math-vs-en 需语义判别学不到
   - **c14b 修复 + 重训中**（2026-08-08）：domain_score_head 参数从 body_params 分离 → adamw 主 lr（1e-3，快 10 倍）。冒烟验证：body 222→195、adamw 36→63（9 neuron × 3 判别器参数全部移入）✓
10. ✅ **C14b 训练 + 诊断：判别器方案深层困难（2026-08-08）**：
    - **c14b（判别器 lr 1e-3）训练完成** → 评估全恶化：code **-61.3%**、math **-67.4%**、zh **-31.2%**、en +0.2%、dialogue +67.2%（C14: code -11.6/zh -6.9/dialogue +71.2/math -60.2）
    - **c14b 诊断**：code 8/8、math 8/8、zh 7/8 路由错误（C14 时 zh 7/8 正常）。**判别器 logit 爆炸**：en 对所有文本输出 6.3-8.4（C14 时 1.2-2.1），code/math/zh 判别器全在 0 附近
    - **根因 1（尺度游戏）**：routing_loss = -log_softmax(domain_logits/temp)[domain_idx] 只约束**排序**不约束**尺度** → 判别器通过膨胀自身 logit 降低 loss（对任意输入都给大正数即赢）→ lr 1e-3 下 en 判别器权重爆炸，压偏置梯度（相对）拉不动
    - **根因 2（判别任务不对称）**：C13/C14/C14b 三次 en 判别器全是默认赢家——math/code 文本是**英文**，en 的"英文性"覆盖它们；math vs en 需语义判别（难），en 只需识别英文（易）→ 判别器学习不平衡
    - **根因 3（判别器输入不可比）**：每个 neuron 判别器输入是**自己的 round 1 表征 h**（不同空间），softmax 在 9 个不可比分数间比较（同 C12 跨 embedding 不可比问题）→ en neuron 表征空间与自身判别器对齐好，天然占优
    - **教训**：C14（MLP + lr 1e-4 + temp 1.0）是三版最佳（zh 修好、code 略降、math 仍错）；判别器 lr 提高无益反害。路由信任依赖"域标签预测"是间接代理，需重新设计
11. ✅ **决策：D 方案（预测质量路由，C15）实施中（2026-08-08）**：
    - **用户决策**：D. 预测质量路由（替代 C13/C14 域标签判别，域判别方案三次迭代判定失败）
    - **实施**（commit 03c3f1f）：① `domain_score_head` 废弃 → `quality_head`（MLP 结构不变），round 1 独立前向输出"我对当前样本的预测质量"；② 监督从域标签 CE 改为 **contrastive_loss**（KL(softmax(quality/1.0) || softmax(-NLL/0.5))）——NLL 是客观预测质量，训练时可得，推理时 quality_logit 直接可用；③ 替代 C12 的 field_score_proj 条件（从未启用：训练时 score_dim=None → contrastive 恒 0）；④ quality_head 走 adamw 主 lr
    - **验证**（verify_contrastive.py）：contrastive_loss 非零（1.78-4.41）+ quality_head 梯度流动（10.8-22.9）✓；冒烟 5 步内 zh/en 文本 quality 排序已与 NLL 对齐（code/math 需正式训练收敛）
    - **为什么能修**：① 无判别任务不对称（监督目标是客观 NLL 而非域标签）；② KL 对齐固定分布（softmax(-NLL/0.5)）→ logit 膨胀会增大 KL → 天然防尺度游戏；③ 所有域 batch 生效（含 dialogue）
12. ✅ **生成质量验证链建立 + 崩坏层定位（2026-08-08，用户关键质疑"生成质量没有测试"）**：
    - **用户质疑成立**：C13→C15 全部基于 PPL/EMERGE 纸面指标迭代，真实生成从未系统测试
    - **建立 gen_test_collab.py**：真实推理路径（ensemble.forward）+ 采样/argmax + solo 对照 + subset 阵容对照
    - **实测证据**（collab_v3_c14b）：
      | 阵容 | 生成质量 |
      |------|---------|
      | 混合 9 ensemble（argmax） | 全崩：回显 + 答：空行 |
      | solo code（argmax） | 全崩：仅回显/重复标点 |
      | solo zh_std0（采样 0.9） | 崩坏：句子断裂 + 来 来 来 退化 |
      | 旧 5 + c14b 注入 | 崩坏：clusters/colonial 随机英文词重复 |
      | **旧 5 原始权重（cross_spec_dialogue）** | **正常：4/5 通顺对话**（你好，很高兴你们 / 可以学习基本代码） |
    - **崩坏源定位**：collab 训练产物（body 尾层微调 + 混合 side_channels）破坏个体生成能力；旧 5 原始协作权重生成正常 → **PPL 改善是混合数据 next-token 拟合的假象，与生成能力负相关**。路由不是生成崩坏主因
    - **方法论修正**：真实生成测试（采样 + 多 prompt）为第一验证指标；PPL/EMERGE 降级为辅助诊断；每次训练后必须过生成测试
13. **方向决策（用户确认，2026-08-08）**：按生成质量优先调整——暂停路由迭代，先解决 collab 训练破坏生成的问题。候选：
    - A. **collab 训练冻结 body**：不微调 body 尾层（unfreeze_layers=0），只训 side_channels/融合层——保持个体生成能力（协作层只增强不破坏）
    - B. **分解验证**：旧 5 + c14b 只注 side_channels vs 只注 body，细分离破坏源
    - C. **collab 产物不覆盖 body**：生成时用原始 body + 训练后的协作层
13b. ✅ **分解验证完成（2026-08-08，B 方案实施）**：gen_test_collab.py 新增 `--inject`（side/scale/body/cross 选择性注入）+ `--no-lmhead`（body 注入时跳过 lm_head）。c14b 终态分解（subset=dialogue，unified=3072 与训练一致）：
    - 全注入（终态）→ 崩（clusters/clicking 重复）＝ 复现 C14b 崩坏
    - body+cross 注入 → 崩（同重复崩坏）
    - body+cross --no-lmhead（排除 lm_head 微调）→ 仍崩 → lm_head 非唯一破坏源，body 主干（layers.4/5 + domain_score_head）也参与
    - side,scale,cross（排除 body）→ **无法归因**：判别器（domain_score_head）在 body_state 里，排除 body = 判别器未注入 = 路由全判 en 失效；生成为英文胡言而非重复崩坏（body 缺失使重复崩坏消失，但判别器失效掩盖真相）
    - **结论**：破坏源 = collab 训练的 body 微调整体（尾层 layers.4/5 + norm + field_write + lm_head + 判别器耦合其中），无法用当前注入粒度再细分（判别器耦合在 body）
    - **修复方向收敛**：候选 A（冻结 body）——collab 训练冻结 body 主干，判别器（quality_head）拆为独立分量单独解冻+保存，协作层只训 side/scale/cross_spec
    - **关键事实**：body_state 含 lm_head.weight（256000×512）→ collab 训练微调了共享头，混合数据污染共享输出分布
14. ✅ **C16 实施：LoRA 保护 body + quality_head 独立分量（2026-08-08，commit 0a33a26）**：
    - **用户决策**：LoRA 微调尾层（个体能力零破坏起点 + 保留适配能力） + quality_head 拆独立分量
    - **neuron.py**：`LoraPair`（B 初始 0 → 输出恒 0 → 向后兼容）+ `enable_lora(rank, layers)`（默认尾层 2 层，嵌套 ModuleDict），forward 标准路径在 attention/ffn norm 后输入注入低秩增量、dendritic 路径退化块级
    - **train_cross_domain_collab.py**：`--lora-rank`（默认 16）→ body 全部冻结（含 lm_head，body=0 可训练），LoRA + quality_head 走 adamw 主 lr；save_checkpoint 拆 `head_state`（quality_head）/`lora_state`，body_state 空
    - **gen_test_collab.py**：--inject 支持 head/lora 分量
    - **冒烟验证**（48 步）：loss 5.68→4.96 正常下降；ckpt head/lora_state 结构正确（lora b 非零=学到增量）；**生成测试：无重复崩坏（回显+句号，=原始 body 行为）+ 路由按语言正确（英文→en、中文→zh）**——对比 c14b 的 en 全抢占，C16 修复生效
    - **正式训练数据规模决策（2026-08-08）**：首次全量配置（3000 条/域 + 1000 对话，seq128 batch4）实测 CPU 稳态 16s/step → 全训练 ~29h，远超验证目标。**决策：缩小至 200 条/域 + 300 对话**（C14 崩坏在 100 条/域时已完整复现，LoRA 保护是否避免崩坏无需大语料验证）→ 每 epoch 270 step × 2 ≈ 540 steps，预计 3-4h CPU。旧 ckpt（29h 配置 step 3000 遗留）已清理，重训覆盖 collab_v3_c16.ckpt.pt
    - **✅ 正式训练完成 + 生成测试（2026-08-08）**：540 step × 2 epochs 完整完成（E2 完成 PPL≈743.9，耗时 73min/epoch）。ckpt 结构检查全过：9 阵容完整、body_state 空（保护生效）、head_state 9 个、lora b 全非零（学到真实增量）、cross 投影 8 组。
    - **❌ 生成测试发现新问题：路由全 code 独占**——gen_test_collab（--inject side,scale,cross,head,lora，全 9 阵容）所有 prompt（code/math/zh/dialogue/en）路由权重 code=1.00，其余 0.00 → 生成空洞（zh/dialogue/en 乱码片段）。**但无 c14b 式重复崩坏**（clusters/colonial 无限重复消失）→ LoRA 保护 body 生效，问题在路由层
    - **根因诊断（diag_c16_quality.py，逐文本 9 neuron quality_logit + NLL + maxprob 对照）**：
      - **code quality_logit 爆炸（6.6~16.6），其余全 -2~-7** → trust=softmax(q/0.15) 下 code=1.000 one-hot
      - **真实 NLL 列（shift 预测）code 全局最低**：math 文本 code=6.3 < math=24.4；中文对话 code=10.8 < zh_aug2=15.8；en 文本 code=3.5 < en=26.2
      - **根因：C15 contrastive 监督（KL(softmax(q/1.0)‖softmax(-NLL/0.5))）的 NLL 跨 neuron 不可比**——共享 general 256K 空间英文主导，code 域=英文代码最匹配 + 分布最锐利 → code 对几乎所有文本（含中文对话）NLL 全局最低 → 监督目标天然 one-hot 偏 code → quality_head 忠实学到"code 永远质量最高"（并非学坏，是监督目标本身失效）。与 C12 跨空间不可比、C14b 判别器尺度爆炸同源（空间不对称）
    - **修复方向（2026-08-08，用户跳过提问→自行收敛为 C16d）**：
      - **C16b z-score 验证**（EMA per-neuron 标准化，[ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)）：zh/dialogue 文本路由修复 ✓（code 不再独占），但 **math/en 英文文本被 dialogue neuron 抢**（转译 NLL 基线数百~上万，z 系统性负）
      - **C16c LOO 融合增益实验**：contrib=NLL(去i)-NLL(全)，全部 uniform——base 融合在 max-prob 路由下被转译 neuron 抢位质量差，边际贡献全负，信号不可用，弃
      - **C16d 最终方案：gate+z-score**——z-score（相对自身水平）解决 code 独占 + 绝对质量 gate（batch 最优 ×50 排除 NLL 基线巨大的转译 neuron）防 dialogue 抢英文。验证（verify_c16b_contrastive，25 步 EMA 预热）：code→code ✓ math→math ✓ zh/dialogue→zh_aug ✓ en→math（en 域数据不足致 math 相对提升更大，信号语义正确）。重训中（collab_v3_c16 覆盖，2.5-3h）
      - **经验固化**：NLL 跨 neuron 不可比是 C12/C14b/C16 三次失败的共同根因（跨空间地位：native general vs 转译；英文 vs 中文匹配度）。绝对 NLL 偏锐度、纯 z-score 偏高基线、LOO 依赖 base 融合质量——gate+z-score 是当前可用组合
    - **🔴 范式转向（2026-08-08，用户决策：停止 token 级路由修补，转向任务级路由）**：
      - **定性**：C12-C16 全部撞在同一堵墙——"统一空间 + 全局 token 级竞争"范式本身与生物机制相悖。人脑分工是**解剖结构**（面孔→梭状回/语言→布罗卡区），路由是**任务级**（前额叶执行控制切换任务模式），竞争只发生在**同功能内部**（侧抑制），跨脑区协作是**层级流水线**（前馈预测+反馈误差），从不做"跨脑区全局 token 竞争"
      - **已终止**：C16d 重训（gate+z-score，刚起步 step 10）已停止，定时任务已删除。token 级融合训练线（train_cross_domain_collab.py 的 quality_head 路由）废弃
      - **保留资产**：① LoRA 保护 body 原则（body 冻结 + 低秩增量，C16 已验证零崩坏）；② quality_head 结构（升级为**回合级**质量信号）；③ cortex 已有回合级判定基础设施（_infer_domain 启发式 + domain 参数 + hybrid 共振校验 + _fingerprint_route prototype）；④ C18 客户端链路（assemble_cortex + chat/feed/sleep）
      - **下一步**：C19 任务级路由设计（历史记录，见 `plans/archive/implementation/BIO_INSPIRED_ARCHITECTURE_PLAN.md` 2.7 节）
15. ✅ **C17 实施：新生神经元无缝衔接 IntegrateEngine（2026-08-08，设计→代码→冒烟）**：
    - **设计**：`plans/archive/implementation/BIO_INSPIRED_ARCHITECTURE_PLAN.md` 2.6 节——人脑神经发生 4 阶段（静默→蒸馏→验证→固化/凋亡），参考"沉默突触/关键期可塑性/use-it-or-lose-it/关键期关闭"
    - **ensemble.py**：`_confidence_routing_fusion` 的 conf 乘 `maturity.get_resonance_weight`（新 neuron 融合权重 0.1→1.0 渐进，静默期不参与输出；成熟 neuron 返回 1.0 不受影响；maturity=None 时跳过向后兼容）
    - **integrate_engine.py（新）**：`IntegrateEngine.integrate(new_nid)`——影子 COW 训练（复用 sleep_engine._clone_module/staticmethod _copy_shadow_back），只训新 neuron 协作层（side_channels/quality_head/LoRA，body 冻结 C16 延续），loss = CE + 邻居蒸馏 KL（DISTILL_TEMP=2.0）+ contrastive，每步 tick maturity，maturity≥0.8 跑 ablation（临时从 ensemble 弹出对比 CE）→ commit / apoptosis 信号
    - **sleep_engine.py**：`_integrate_new_neuron` helper，挂载到两个 neurogenesis 创建点（域错误率 + 孤立检测）后
    - **冒烟验证**（verify_integrate.py）：影子 COW ✓ / 训练循环（CE+蒸馏+contrastive）✓ / 静默期（maturity 压低融合）✓ / ablation 真实对比 ✓ / 决策 ✓ / 写回 live ✓
    - **关键发现**：MaturityTracker 已有 get_resonance_weight（0.1→1.0）与 get_lr_multiplier（3×→1×）——静默期机制基础天然存在，C17 只需接线（fusion 注入 + 训练循环）
16. ✅ **C18 实施：协作层注入运行时 cortex（2026-08-08，客户端端到端前置）**：
    - **需求**：客户端端到端测试前置——C16 训练产物（collab_v3_c16.ckpt.pt，9 阵容）需注入运行时 cortex.ensemble 才生效；原 `_load_collab_weights_into_cortex` 只支持 final artifact 格式（side_channels/cross_spec 无 _state 后缀）+ 缺 head/lora
    - **loader.py**：`_load_collab_weights_into_cortex` 扩展——① `_pick` key 兼容（side_channels_state/cross_spec_state 训练格式 + side_channels/cross_spec 旧格式）；② head_state 注入（quality_head 独立分量）；③ lora_state 注入（enable_lora 后 load，rank 从 a.weight 推断）；④ body_state 语义更新（C16 冻结 → 空 dict 不注入，保持原始 body 保护个体能力；旧 C13/C14 微调格式仍兼容）
    - **assemble_cortex**：新增 `collab_name` 参数（默认 cross_spec_dialogue.pt，可指定 C16 训练 ckpt）
    - **验证**（verify_collab_runtime.py）：head 注入 5/6 ✓、lora 注入 5 个 b 非零 ✓、body 0 分量 ✓、生成不崩 ✓
    - **意义**：客户端端到端 = assemble_cortex(collab_name="collab_v3_c16.ckpt.pt") 后纯客户端驱动（chat/feed/sleep API），无需额外接线
17. ✅ **C19 实施：任务级路由（Executive Control Routing）冒烟验证通过（2026-08-08，verify_c19_executive.py）**：
    - **回合级判定**（[cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) `_executive_route`）：混合信号 = 启发式 `_infer_domain` 定基础域 + quality_head 回合级聚合（round1 全量 neuron 的 quality_logit → per-domain mean）→ 显著占优（>1.5×）切换。**复用 C16b 教训**：quality_logit 跨 neuron 不可比（未校准 code head 恒 16.9 vs 其他 -2~-5）→ per-neuron EMA z-score + 成熟度门（count<20 回退纯启发式，回退安全；C20 训练期间 EMA 自然积累后混合信号自动生效）
    - **修复 blocker 1：quality_logits 只收集 1 个**（[ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)）——final 聚合误用共振过滤后的 `active_ids`（收敛后只剩最强 1 个）→ 改 round1 快照 `round1_active_ids`，9 个全收集 ✓
    - **修复 blocker 2：生成 OUT_OF_RANGE**（[cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) `_generate_p7`）——**架构收敛遗留**：2026-08-07 共享 general lm_head 后 logits 统一 256K 空间，但 `_generate_p7` 仍用 domain tokenizer decode（2026-07-31 per-neuron lm_head 时代的正确做法）→ general 空间 id 越界 domain vocab。已收敛：生成/decode 全程 general 256K（identity 回填 + general_sp.DecodeIds），domain 只负责激活 neuron 选择
    - **executive 生成**：leader 限定 dominant 域内最强 neuron（不跨域抢占），回合内稳定生成不融合
    - **验证结果**：quality_logits 9/9 ✓；判定 code→code / zh→zh / dialogue→zh / en→en ✓（math→en 为启发式误判，C20 quality 修正场景）；executive/fusion 生成无 OUT_OF_RANGE ✓
    - **已知限制**：executive 生成质量仍碎片（C16 基座在 general 空间中文能力弱，非 C19 机制问题），C20 回合级训练 + 基座能力提升解决
    - **下一步（C20）**：回合级监督训练——候选 neuron 轮流主导生成完整回复 → 融合 NLL 评估 → 标签训练 quality_head（回合级对齐），使 quality 信号成熟并修正启发式误判（如 math→en）
18. ✅ **C20 实施：回合级质量监督训练（2026-08-08，验证通过 5/5）**：
    - **监督粒度升级**（[ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py) `forward_train` 新增 `answer_mask`）：per_neuron_nll 只对 answer（回复）部分算回合级 NLL——prompt 部分无区分度（输入即答案），answer 才是"谁能生成好这个回复"的回合粒度真实质量。C16d 全序列 NLL 被 prompt 稀释。
    - **train_round_level_quality.py**：只训 quality_head（body/LoRA/side 冻结，C16 保护延续）；监督 = C16d 复用（per-neuron EMA z-score + 绝对质量 gate）作用于回合级 NLL；warm start 从 collab_v3_c16.ckpt.pt；**同域 batch**（混合域 batch 被低 NLL 域拉低 gate 阈值 → dialogue 转译 neuron 全排除，监督失效）
    - **正式训练**：1100 steps（200 条/域 + 300 对话 × 2 epochs，~2h CPU）
    - **✅ 验证**（verify_c20_round_quality.py）：C16 head code 恒高 16.91（未校准）→ C20 head 校准后回合级判定 **5/5**：code→code / math→math（修正 C19 误判）/ zh→zh / dialogue→zh / en→en
    - **关键调优**：_executive_route 切换条件加 z 绝对差阈值（≥0.7σ）——纯 1.5× 比例对接近 0 的 z 太宽松（en 回合 zh 0.49 vs en 0.04 也满足 → 错误覆盖启发式正确的 en）；0.7σ 实测区分 math（差 0.95 切 ✓）/en（差 0.45 不切 ✓）
    - **已知限制**：zh general 是"全能型"（多文本 NLL 低）→ quality z 系统性偏高，靠 0.7σ 显著门防错误覆盖；生成质量仍碎片（C16 基座限制，非 C20 范围）
    - **产出**：collab_v3_c20.ckpt.pt（head_state 分量，C18 注入格式兼容）
    - **🔴 端到端生成验证（2026-08-08）：碎片根因确认 = 统一空间收敛未完成**——executive 生成仍碎片（dialogue→zh 判定后 leader 输出 `clusters clicking deadline` 等）。逐 neuron lm_head 检查：**5 个 dialogue neuron lm_head vocab=50000（zh 域空间，C16 转译设计保留域头），4 个 general neuron=256000**。C19 生成已统一 general_sp decode → leader 选 dialogue neuron 时 zh 空间 id 用 general 词表解析 → **完全错位** → 碎片。且 zh general（general 空间）只会回显+句号（foundation 全文本续写训练，无 SFT 问答能力）。**没有任何 neuron 在 general 256K 空间拥有中文问答生成能力**。general tokenizer 中文覆盖 0 unk（词表无问题）。
    - **修复方向（C21）**：dialogue neuron 迁移到 general 256K 空间——注入 shared_lm_head + general 输入表 + 中文 SFT（general 编码）训练 body，让对话能力在 general 空间表达；统一空间后转译层废弃，leader 生成空间自洽

**✅ 多模态集成层收敛完成**（2026-08-07，commit 3e4efc0）：
- **问题诊断**：`mm_lm_heads` 独立 codebook 输出头与"共享 general 256K lm_head"架构矛盾——输入投影进 shared 空间、输出却跳去独立 codebook 空间，输入输出割裂
- **收敛方案**：废弃 `mm_lm_heads` 独立头 + `mm_logits_modality` 机制，多模态输出统一走共享 general lm_head（与文本同构）
- **核心改动**：
  - [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py)：删除 `mm_lm_heads` 属性 + `register_modality_lm_head`/`compute_mm_logits` 方法，`auto_register_modalities` 只注册输入投影
  - [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)：删除 `mm_logits_modality` 参数及传递，`logits` 统一走 `compute_logits`（共享 general head）
  - [sleep_engine.py](file:///e:/taiji-neuron/taiji/life/sleep_engine.py)：target 映射到 general 词表 codec 段（`base + codec_index`），删除 `mm_lm_heads` 依赖
  - [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py)：生成路径 mask 到 codec 段采样 + 映射回 codec 索引自回归
- **modality 映射**（general 词表预留段）：
  - image: `[1000, 9191]`（base=1000, size=8192）✅ 支持训练 + 生成
  - audio: `[9192, 13287]`（base=9192, size=4096）✅ 支持训练 + 生成
  - video: 无预留段 ❌ v1 仅支持输入编码（生成需 v2：codec token 并入 general 词表）
- **验证通过**（`verify_multimodal.py` + `verify_mm_ensemble_train.py`）：
  - 单元通路：image/audio/video forward 返回 `[1, L, 256000]` logits（general vocab）✅
  - ensemble 训练闭环：image 2 rounds loss 12.69→12.24（下降），audio 1 round loss 14.92 ✅
  - loss ≈ ln(256000) ≈ 12.45 符合未训练 uniform 分布预期
- **后续（v2）**：codec token 并入 general 词表 + mm 统一序列训练（图文双向），video 生成支持

**背景**：跨域协作 v3 + rounds=2 已达成域内 EMERGE +4.0% 全域转正（"5 联合 > 5" 域内实证）；通用空间投影极限诊断结论 = 静态稀疏投影到 256K 摧毁置信度（max-prob≈0.001）→ 共享 general lm_head 免投影是架构上限最高的解

---

## 📌 历史状态（2026-08-06 19:40 跨域实测线）

**跨域协作 5>5：域内全正（+4.0%）✅，通用空间探针确认架构方向**

**最新实测（v3 权重 + rounds=2，n-ppl 5）**：
- ✅ **域内 EMERGE 平均 +4.0%（全域转正）**：code +1.9% / math +3.6% / zh +6.9% / en +3.3% → "5 联合 > 5"域内实证
- ✅ **通用空间生成探针**：zh 提问 → general 256K 空间分工路由，输出变通顺中文（zh neuron 被选中；code 域路由输出 code 碎片）——证明通用空间让 zh 理解与 code 表达共存，路由结构正确
- ❌ **无配对数据 → 模型只会续写中文，未学会"切到 code 表达"**：通用空间路径正确但需配对训练数据

**下一步（唯一建议）**：构建配对数据（zh_sft 中 52 条含代码样本做种子 + code_sft 指令机器翻译成中文配对），在 general 输出空间训练协作层，让侧通道学会"zh 理解写入场 → code neuron 读取场生成"的转换 → 语义涌现

**已完成链条**（全部提交）：
基座重建（foundation_v1）→ 跨 vocab 硬路由融合（EMERGE -4228%→-5.9%）→ 协作层 v3（-5.9%→+0.9%）→ v3+rounds=2 全域 +4.0% + 通用空间探针

---

## 📌 历史状态（2026-08-06 19:10 跨域实测线）

**跨域协作 5>5 实测：域内涌现达成 ✅，跨域语义桥接待决策**

**最终实测结果（cross_domain_v3 训练权重 + rounds=2 场条件化）**：
- ✅ **域内 EMERGE 平均 +0.9%（转正）**：code +2.1% / math -0.2% / zh -0.8% / en +2.5%
  - 协作不再破坏个体，code/en 还超越个体 → "5 联合 ≥ 5"（个体基线 code 5.4 / math 12.2 / zh 66.6 / en 62.0，均远低于随机）
- ⚠️ **跨域生成（zh→code）结构性涌现**：输出呈 code 形态（#注释、"for i in range"、"#include"），单个体无法做到（code neuron 读不懂中文、zh neuron 输出不了代码）——但**语义未桥接**（生成的代码与提问无关）

**关键决策点（下一步方向）**：
1. **场状态桥接**：训练时显式用"zh 理解 → code 生成"的配对数据（双语指令对），让侧通道学会把 zh neuron 的理解写入场、code neuron 读取场生成 → 语义涌现
2. **通用词表输出空间**：融合输出改到 general 256K 空间（各 neuron 投影到 general，路由与生成都在通用空间），zh 理解与 code 表达天然共存 → 更大改动
3. **接受当前里程碑**：域内 5≥5 已实证 + 结构性跨域涌现，语义跨域作为后续独立课题

**已完成链条**（全部提交）：
- 基座重建（foundation_v1，verify_v3 损坏诊断+按对话配方重训）→ 跨 vocab 硬路由融合（EMERGE -4228%→-5.9%）→ 协作层 v3 训练（EMERGE -5.9%→+0.9%）

---

## 📌 历史状态（2026-08-06 18:30 跨域涌现训练线）

**跨域协作 5>5 实测：基座重建 ✅ → 融合修复 ✅ → 跨域涌现训练中**

**实测里程碑**：
1. ✅ **基座重建**（foundation_v1）：verify_v3 不可用（训练 embedding 丢失），按对话配方重训 4 域 neuron + 共享 embedding（best 权重早停保存 + 配对校验）。个体基线：code PPL 9.3 / math 38.2 / zh 345 / en 138（均远低于随机）
2. ✅ **跨 vocab 融合修复**（commit 060cc1d）：软加权平均被域外 neuron 投影的极端幅值淹没（zh max|logit|=60828 vs code ±14，0.07×60000=4200 主导）→ 改为**按位置硬路由**，置信度 = min(原生 max-prob, 投影后 max-prob)（原生项=域门封顶，投影项=目标空间信息量）
3. ✅ **域内 EMERGE -4228% → -5.9%**：code -1.2% / math -8.4% / zh -0.2% / en -13.7%（协作不再破坏个体；en 偏差源于 code/en 英文域重叠竞争）
4. ⏳ **跨域涌现训练中**（cross_domain_v3，epochs 4）：新融合下 n_rounds=2 场条件化给侧通道梯度（R2 注入让 zh 理解写入场、code 生成读取场 → 桥接）

**跨域生成（zh 提问→code）当前状态**：路由已让输出变为 code 侧 token 碎片（非 token 汤），但 zh 理解未桥接到 code 生成 → 需训练好的侧通道/场条件化

**待决策/下一步**：
- 协作层重训完成后用 n_rounds=2 + 训练权重评估，验证场条件化能否桥接 zh→code

---

## 📌 历史状态（2026-08-06 17:00 跨域诊断线）

**跨域协作 5>5 实测：基座诊断 → 重建 → 验证中**

**核心诉求**：5 个域神经元（code/math/zh/en）联合 > 5 个独立之和（涌现）。

**诊断结论（verify_v3 基座不可用）**：
- verify_v3 多域神经元由 train_neurons_from_scratch 训练，输入是**域编码**（p7 input_ids max 11912 < code vocab），但训练时 `create_shared_embedding` 随机初始化且**从未保存** → 评估时任何管线（域编码/通用编码/joint 变体）PPL 均≈随机（loss 9.4-10.6 vs 随机 9.39）
- 已验证：code/math/zh/en 的 neuron_*.pt 与 *_joint.pt × shared_embedding*.pt 全部组合，loss 均≥8.9
- **根因**：域编码输入 + 丢失 embedding = 不可复现的模型（权重对着随机向量训练，向量已丢失）

**正确配方（对话管线实证，PPL 2.2 可复现）**：
- 输入 = general_sp 编码 → 共享 embedding（保存）→ neuron；标签 = 域 tokenizer
- 冒烟验证：对话 embedding 基座 + code 数据，code neuron 500 步 loss 9.4→2.7（PPL 15）✅

**行动**：
1. ✅ 新建 `train_multi_domain_foundation.py`：按对话配方重训 4 域 neuron + 联合训练共享 embedding（对话基座 warm-start），保存时回读验证 checkpoint（用户规则：训练前确认保存正确）
2. ⏳ 运行中：`data/foundation_v1`（600 步/域，~25min，含周期保存+回读验证）
3. ⏭ 基座完成后：重跑跨域协作训练（train_cross_domain_collab.py）+ 评估（_eval_cross_domain_collab.py 已验证 API 兼容）

**参考基线（同域 5>5 已实证）**：5 个 zh 对话 neuron 协作 PPL 29.7 vs 个体 95，EMERGE 65.7%（对话管线）

**待决策/下一步**：
- 基座训练完成后自动继续跨域协作训练 → 评估 5>5

---

## 📌 历史状态（2026-08-06 15:26 对话线）

**EOS + 短答案筛选重训已完成，评估中**：

**训练结果**（2026-08-06 15:26 完成，总步数 39480）：
- 最终 PPL=3.0（E1 331 → E4 15.5 → E6 6.3 → E8 3.0），远优于上一轮 E6 32.9
- 产物：`cross_spec_dialogue.pt`（推理用）+ `cross_spec_dialogue.ckpt.pt`（可续训）

**评估诊断**（2026-08-06）：
1. ✅ **发现并修复 eval 加载缺陷**（commit b5a1c80）：`load_cross_spec_weights` 此前只加载 cross_spec 投影层，**丢失 side_channels/scale_bias/body_state**（协作层训练核心产物）→ eval 用"训练前"参数，PPL 虚高 705。修复后训练分布 PPL 2.2 复现训练（口径一致）
2. ✅ **修复推理路径 gap**（缺口 N，commit 待）：推理 "soft" 此前走 per_position（entropy 启发式），无视训练学的共振分 → 训练分布 PPL 2.2 vs 推理 12.6。A/B 实验证实共振分融合生成明显更通顺 → forward 统一 "soft" = 共振分 softmax 融合（与训练 forward_train 对齐）。修复后 eval 协作 PPL 1516 → 240.7，EMERGE 70.8%
3. ⚠️ **分布泛化 gap 仍存**：训练分布 2.2 → eval 分布（无筛选长答案）240——训练数据（≤150字短答）与 eval 分布差异大，有过拟合风险
4. ⚠️ **生成知识性错误仍存**：语言流畅度已达标（"你好，很高兴。我是一个人工智能助手"），但事实错误（"法国首都是伦敦"）——模型容量/数据上限，非路径问题

**待决策/下一步**：
- 数据侧提升：更高质量/更长答案/更多样的 SFT 数据后重训（治模型知识上限）
- 或接受当前质量，进入跨域协作层训练（code/math 加入）

**根因诊断**（API 实测质量不达标的三个核心缺陷）：
1. ❌ **训练数据未加 EOS**：`batch_align_and_embed` 只产生 domain_targets，无 EOS token → 模型永不自然停止
2. ❌ **训练数据严重未充分利用**：仅加载 15000/88730（17%），其他 6 个文件完全未用
3. ❌ **训练/生成长度严重不匹配**：alpaca-zh 答案 200-500 字，生成 max_tokens=60（约 80-100 字）→ 模型学成了"长文本续写"而非"简短问答"

**三项核心修复**：
1. ✅ **EOS 注入**（[translator.py:498-521](file:///e:/taiji-neuron/taiji/resonance/translator.py#L498-L521)）：`batch_align_and_embed` 追加 EOS token（截断前注入，保证 EOS 始终在末尾）。smoke test 验证：末尾 target=3(EOS)，sft_mask=True ✓
2. ✅ **短答案筛选**（[utils.py:283-338](file:///e:/taiji-neuron/scripts/training/utils.py#L283-L338)）：`load_dialogue_texts_multi` 加 `max_answer_chars=150` 参数，筛选答案≤150字的样本，匹配生成 max_tokens=60
3. ✅ **数据量提升**：从 15000 条（仅 alpaca_zh 第一个文件）→ 19745 条（7 个文件全部加载，去重后），多样性显著提升

**重训参数**：
- 从头训练（不 --resume）：加 EOS 后训练任务本质变了，不继承"无 EOS 长答案"偏见
- epochs=8, batch=4, lr=1e-3, max_texts=88730, max_answer_chars=150
- 总步数 39480，ETA 约 301 min/epoch
- 备份：cross_spec_dialogue.pt.pre_eos_finetune

**自适应激活设计已完成**（2026-08-05）：详见 [§4.0c](#40c--自适应激活设计r1-软路由--top-k-稀疏路由2026-08-05-设计)。Probe-based Sparse Router 方案落地，待训练完成后实施。

**并行工作完成**（2026-08-05，训练期间开展 4 项，全部提交）：
1. ✅ **eval_dialogue.py 支持任意 checkpoint**（commit 96b9ecd）：`--ckpt_path` 参数，用于训练完成后对比 held-out PPL 判断过拟合早停
2. ✅ **稀疏 vs 稠密对比脚本**（commit 658d546）：`compare_sparse_dense.py`，同 checkpoint 双 ensemble 对比协作 PPL/EMERGE/激活数/速度（smoke 已验证）
3. ✅ **跨域神经元 Step 2 数据准备**（commit 5d98f95）：`p7_{domain}_mixed_tokenized.pt`（6000 条/域，域 SFT + 英文对话），train_neurons_from_scratch.py 支持 `--data-suffix mixed`
4. ✅ **API 集成修复**（commit 858c3e1）：新建 `taiji/core/config.py`（TrainingConfig + 6 个接口），memory_watchdog 补 `force_memory_refresh`/`get_memory_status_dict`，API 29 路由可正常启动

**API 路径五项修复（已完成，commit e94ca1d + 6426797）**：
1. 训练/推理 embedding 错配 → per-neuron shared embeddings 注入 cortex
2. 解码 byte fallback → `domain_sp.DecodeIds`
3. 默认参数宽松 → 60/0.55/15/soft
4. EOS 缺失 → 温和 EOS bias(+0.5)（临时方案，现已被训练时 EOS 注入替代）
5. 跑偏截断 → 连续 3+ 非中文 token 回退截断

---

## 🧠 神经元综合体饱和点判断标准（2026-08-04 决策）

**何时触发新神经元进化**——三个量化指标：

| 指标 | 饱和标准 | 当前值 | 状态 |
|------|---------|--------|------|
| PPL 收敛 | 连续 2 epoch Δ<0.5 | epoch 9→10 Δ=-14.2 | ❌ 远未饱和 |
| EMERGE 递减 | 连续 2 次评估 Δ<5% | 21.7% vs 22.7% | ❌ 协同收益稳定 |
| 参数效率 | body 梯度范数持续 <1e-4 | 仍在下降 | ❌ 未饱和 |

**触发时机**：当前轮训练后（PPL 预计 < 20），如果 API 质量仍不达标，就是触发新神经元的时机。预计 1-2 轮训练后（3-6 天）。

**新神经元方向（上限最高）**：跨域扩展（en_dialogue）。理由：
1. 当前全是 zh 神经元，跨域协作是"小神经元匹配大模型"的核心
2. 已有跨域 tokenizer 基础设施（en 16K vocab）
3. 跨域协作能显著提升综合能力（不同视角的共振）

**技术路径**：neurogenesis + lifecycle + establish_topology_channels 自动重建 + finetune_cross_spec 微调

---

## ⚠️ 架构本源定性（2026-08-04 认知重构，详见 §4.0）

**核心定性（重构）**：**涌现已存在**——单个神经元 PPL ~20-42 无法正常对话，5 个协作 PPL 15-33 能正常对话，这正是涌现的定义。"单神经元较强"是效率优势（不需要巨大协作层就能涌现），不是劣势。人脑类比不是唯一标准。

**唯一核心缺陷**：自适应激活不足（协作层稠密，R1 软路由需要强化为 top-K 稀疏路由）。

**决策时点**：
- **Step 1（进行中）**：zh 综合体 + EOS+短答案重训 → 验证能正常对话（涌现的输出验证）
- **Step 2**：加入 code/math/en 等特定能力神经元，测试跨域涌现（§4.0b 候选 1）
- **Step 3**：强化自适应激活（R1 → top-K 稀疏路由），提升协作效率
- 方向 B 定位修正：不再是"上限更高的备选"，而是"探索另一种涌现机制的实验"，优先级降低

**详见**：[§4.0 涌现已存在](#40--架构本源定性涌现已存在核心缺陷是自适应激活2026-08-04-认知重构) | [§4.0b 涌现深化探讨](#40b-涌现的深化探讨--新能力方向2026-08-04-认知重构) | [§6 方向 B](#六方向-b-备案小神经元--强协作架构2026-08-04-设计)

---
