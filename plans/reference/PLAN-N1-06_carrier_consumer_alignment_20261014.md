# PLAN-N1-06 · N1 载体与消费实施附件（2026-10-14，C0 冻结版 r1）

本附件落实 [开发计划修订](PLAN-N-01_development_revision_20261014.md) 的 C0，冻结 C1 的实现身份供 owner 决策。本文只有代码核对和容量测算，没有新行为结果；不构成动码、计算或训练授权。继承 [PLAN-N1-05](PLAN-N1-05_persistent_stepwise_carrier_prereg_20261014.md) 五条载体判据和默认中性守卫；消费与信用的新协议须按本附件列明的差异另冻版本，不追改 [PLAN-N1-04](PLAN-N1-04_s1b_reactivation_implementation_prereg_20261014.md)。

## 1. 信息流与继承范围

生产者记录实际生成步的皮层上下文投影→按 episode 存入独立载体→checkpoint 保存→独立进程恢复→读取 episode 轨迹→消费阶段检索/选择并再激活→实际生成与来源记录。

载体记录的是**生成时状态轨迹**；它和实际观察经历的绑定须单列。每个 episode 的来源应能回查到输入经历及记录时刻，并标明轨迹为生成过程产物。保存这条轨迹不赋予它事实身份，也不改变奖励。C1 只证明可记录、可恢复、可供给，不能宣布问答或记忆能力成立。

## 2. C1 实施约束

| 项 | 新附件约定 | 继承或替代关系 |
| --- | --- | --- |
| 开关 | `persistent_stepwise_carrier` 为运行时显式属性，默认关，不加入 TaijiConfig | 继承 05；04 未来消费版本同样采用运行时属性，替代原“配置开关”措辞 |
| 开关 | `Taiji.persistent_stepwise_carrier` 运行时属性只控制生成时的记录，默认 `False`；不加 `TaijiConfig` 字段，也不保存开关状态 | 继承 05；载荷恢复后记录仍关闭；只读的 episode 查询 API 可显式调用，不受记录开关隐式影响 |
| 来源与 episode | `SeedRuntime.chat()` 已创建 `InputFrame(input_id, modality, source, timestamp, provenance, confidence)`；沿 `Seed.generate_input` → `TSKV8Adapter.generate_input` 传给 `Taiji.generate`。carrier 用已保存的单调 serial 生成唯一 episode id（`input_id` 是来源字段，不假定它全局唯一），并原样保存全部帧来源元数据 | 缺 `input_id`/来源字段或 serial 撞号时响亮拒绝；不从答案文本/题目标签补造来源 |
| 写入 | 在 `Taiji.generate` 真实发射环 `for` 内，先处理 stop boundary；每个实际追加的答复字节，在 `observe(next_symbol)` 前记录 `cortical_cue()`，即该字节发射时的上下文；每轮暂存，正常结束时整轮提交 | 挂点 `taiji/model.py:3443-3504`；向量来自 `:779-786`，与 `fabric.step` 所需上下文同形；不记录未发射的 stop 符号，不从测试答案补造轨迹 |
| 载荷与兼容 | 在 Taiji v10 核心 payload 里加可选顶层键 `persistent_stepwise_carrier`，值含 `format=taiji-persistent-stepwise-carrier-v1`、`version=1`、`context_dim`、持久 `next_episode_serial`、episodes、来源元数据及 float32 投影矩阵；空且关闭时不输出该键；旧载荷无键按空载体恢复 | 写入 `_checkpoint_core()`、读取 `Taiji.restore()`；`TSKV8Adapter.native_checkpoint()` 将其放入 kernel，`Seed.checkpoint()` 的兼容 payload 也同源带入。保持 `TaijiConfig` 严格相等检查不变 |
| 读取与反证 | 按明确 `episode_id` 返回克隆的有序向量；缺项返回空/拒绝，绝不回退 `last_event`、剪贴板、缓存或现场重算 | 读取口由本载体提供；取值直接喂 `fabric.step(..., episodic_feedback=...)`，并实测空载体负反证 |
| 保存与恢复 | `SeedRuntime.save()` 原子写；独立进程用 `SeedRuntime.load()`（`weights_only=True`）重载，逐条核数量、shape、dtype、内容；运行时记录开关恢复默认关，审计通过明确 episode id 调只读查询 | 锚点 `api/seed_runtime.py:296-340,1503-1531`；禁止同进程 restore 冒充持久性 |
| 容量与隔离 | 基座 `output/a31_chunked_self/checkpoint.pt` 的 region sizes 为 `[256,192,128]`，故 `cortical_context_dim=1152`；正式面前 96 manifest items 实有 288 turns。冻结 C1 上限为 288 episodes × 每 episode 256 步＝73,728 vectors，float32 原始上限 339,738,624 B（324 MiB）；超出即失败，不做静默 FIFO/截断。各臂使用独立 checkpoint/output | 仅是 C1 取证上限，不是长期产品保留策略；C2 前须另定生产保留/逐出政策。每臂披露真实 episode/step/字节量及前后权重摘要 |

C1 正式面保留 05 的 96 个 manifest items、`max_length=256` 与旧档身份；读 manifest 确认其前 96 项共 288 turns（每项最多三轮），不能再把 96 items 写成只有 96 generations。所用 manifest 为 `plans/manifests/r2_copy_surface_extension_v3_position_random.json`（SHA-256 `B162FF091630F64A8CAC986184729FF7894887F4806C923CC12B301CE5AE4FEE`），母档为 `output/a31_chunked_self/checkpoint.pt`（SHA-256 `CA2628077B21BC4C9A6F2FC06FF410D70311B1218A30B8794ECFD7E5A027D8BB`）。≤30 分钟 CPU 的预算不得省去新进程恢复；324 MiB 原始向量上限尚未实测，起跑前需做不改母档的容量/耗时核算；预计超预算则回到预算决策，不裁短轨迹来凑预算。

### 2.1 C1 实施触点与取证命令

1. **写入链**：输入来源从 `api/seed_runtime.py:493-513` 的 `InputFrame` 贯穿 `seed/model.py:194-217`、`taiji/adapter.py:11137-11172`，进入 `taiji/model.py:3265-3504`。`Taiji.generate` 当前 reset episode 固定为 `generation`；C1 开启时由 carrier 持久 serial 生成唯一 episode id，并把 `InputFrame` 元数据随轨迹保存，避免直接拿可能复用的 `chat:{tick}` 当唯一键。若直接调用生成环而未提供来源帧，开关打开时响亮拒绝。捕获位置在 `:3487` boundary stop 检查之后、`:3491` append 与 `:3495` observe 之前，保证投影对应这次实际发射的字节。
2. **状态链**：`PersistentStepwiseCarrier` 记录结构落在 `taiji/memory.py`，由 `Taiji` 持有；不得复用 `EpisodicField.last_event` 的池化字段。公开只读口定名 `read_persistent_stepwise_episode(episode_id)`，只返回该 episode 的有序克隆向量，缺项返回空。核心载荷挂在 `taiji/model.py:3643-3680`，旧键缺失的默认恢复落在 `:3712` 的 `restore()`；原生 envelope 由 `taiji/adapter.py:12464-12620` 包含 kernel，Seed 信封在 `seed/model.py:225-259` 同时保留原生和兼容两条载荷。API 保存入口 `api/seed_runtime.py:1503-1531` 是原子保存，恢复入口 `:296-340` 会新建模型后调用 `Seed.from_checkpoint`。
3. **C1 守卫与正式面命令**（下列新开关/参数由批⑴实现；本轮未运行）：

   ```powershell
   python -m pytest tests/taiji_native/test_persistent_stepwise_carrier_contract.py -q
   python scripts/training/probe_taiji_a30_stop_failure.py --checkpoint output/a31_chunked_self/checkpoint.pt --manifest plans/manifests/r2_copy_surface_extension_v3_position_random.json --limit 96 --max-length 256 --persistent-stepwise-carrier --carrier-checkpoint-out output/n1_carrier_open/checkpoint.pt --out-report reports/taiji_n1_carrier_open_20261014.json
   python scripts/training/probe_taiji_a30_stop_failure.py --checkpoint output/a31_chunked_self/checkpoint.pt --manifest plans/manifests/r2_copy_surface_extension_v3_position_random.json --limit 96 --max-length 256 --out-report reports/taiji_n1_carrier_off_20261014.json
   python scripts/training/audit_taiji_n1_persistent_stepwise_carrier.py --checkpoint output/n1_carrier_open/checkpoint.pt --report reports/taiji_n1_carrier_open_20261014.json --expect-episodes 288 --expect-context-dim 1152
   python scripts/training/audit_taiji_n1_persistent_stepwise_carrier.py --checkpoint output/a31_chunked_self/checkpoint.pt --expect-legacy-absent
   ```

   第一条钉载体合同及互斥拒绝；第二条沿既有 `runtime.chat()` 生成链开载体并另存 checkpoint；第三条同母档、同题面做默认关闭对照；第四条由独立解释器验证恢复内容、episode 清空后不回退来源、以及反馈向量可通过 `fabric.step` 形状校验；第五条检验当前旧档无载体键可加载且默认关闭。96 items 的运行还必须满足 05 的 tail/≥0.95 覆盖率条件。文件名与参数在 C1 实施中按本文冻结；改名或改变容量/数据身份须先升版本。

## 3. 生产与消费的组合边界

原 05 的载体旗标与再激活旗标互斥仍生效。第一份消费协议采用**分阶段组合**：先开启生产旗标记录并保存→退出→新进程恢复→生产旗标保持关→显式开启消费旗标读取现存载体。不得现场取消互斥，也不得用临时开发回放队列替代持久源。

未来一边记录一边再激活属于另一组合版本，须新增覆盖、非干扰、episode 可见性与默认中性守卫。本附件不批准那项扩展。

## 4. C2 行为判别规格的必填项

消费协议冻结前列明：同 bundle 身份、模型可见的检索线索、episode 选择方法、投影来源、回放长度、状态重置及推理预算。诊断阶段可用已知 episode 做特权定位，但正式能力臂必须由模型当时可见信息选择；oracle 读数不能计能力。

对照按一个因素逐项设：真实轨迹 vs 清空（有无供给）；真实轨迹 vs 错 episode（内容依赖）；真实轨迹 vs 同内容乱序（顺序依赖）；真实轨迹 vs 同剂量池化重放（逐步轨迹贡献）。“清空”不作为预算等量的纯结构比较。每对记录实际重放步数/计算量；无法等量时给质量—成本曲线，不宣称纯架构胜利。

复述、世界转移、身份同构逐项读完整正确输出、停止、事实翻转、未知拒答和错误类型；恢复/跨会话隔离单列。所有零训练消费臂 `learn=False` 且前后权重摘要相同。首步分布改变只判接入，不代替行为收益。具体分母、数值阈值、散布或重复资格在消费版本起跑前冻结，本附件不临时发明通过线。

## 5. C3 信用实现的设计前置

代码核对：`taiji/fabric.py` 的 `step`、`taiji/model.py` 的 `observe` 与 `generate` 均在 `@torch.no_grad()` 下；现产品路径不能直接取 04 J-N1d-2 所需 autograd 首步梯度。因此这条旧门标为**当前实现不可测**，不能说梯度为零而判机制无效，也不能用参考通路的梯度补发产品通过。

下一信用版本先画实际计算/更新链，列离散选择、detach、保存恢复和资格迹。采用高上限的可比较设计：隔离可微序列参考通路作为信用可达性的诊断对照，产品局部规则必须另用真实更新读数和行为对照证明贡献；参考成功不等于产品信用成功。是否把可微路径变成实际候选属于新架构/训练决策，本轮不采用。

04 “只重放首步应使首步梯度为零”的负对照不成立：保留的首步仍可能影响答案。新梯度协议需在该状态后切断信用并保持同一前向值/状态和更新预算；局部信用协议需切断实际资格/调制通道而保持前向可比。门分别冻结，不能混用指标。C1/C2 可在信用实现尚未裁定时验证，但不能提前结项整个 N1。

## 6. 当前决策与验收范围

PLAN-N-01 的 C0 现已完成：精确触点、payload 版本、容量上限、数据身份和复算命令均在本版冻结。下一步是 owner 决定是否批准 C1 批⑴载体实现及批⑵ ≤30 分钟 CPU 零训练面；在批文前不改产品/探针代码，不启动面。检查点保存和独立进程恢复列为先决验收，不能被同进程读回替代。

C1 通过只解除“无持久载体”这一前置。C2 消费、C3 信用、权重训练、默认产品挂载和新算力不在本次建议范围。失败依原五条载体判据返回，不采用池化多步近似冒充逐步载体。
