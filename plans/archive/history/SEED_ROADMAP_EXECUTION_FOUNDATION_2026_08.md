# Seed / Taiji 路线执行记录：前期 Gate 与路线校准

> 本文由原总路线图按职责拆分而来。原始行号：1209–1489；本文件属于当前计划资料或历史证据，具体身份以本文件所在目录为准。
> 这是 2026-08-27 至路线校准前的执行日志，不再提供新的下一步。

## 16. 当前唯一下一步

**已完成（2026-08-27）：仓库可发现性元数据三项全部落地并通过程序化复核。** 由用户在 GitHub 网页端写入（agent 令牌缺 `administration:write`，详见 14.10），agent 侧用 `gh repo view --json description,repositoryTopics,homepageUrl,usesCustomOpenGraphImage,openGraphImageUrl` 复核：

- description：与定稿文本**逐字符相等**，252 字符（350 上限内）。
- topics：**集合相等**，13/13，missing 0、unexpected 0。GitHub 返回时按字母重排，故只判集合不判顺序。清单为 `predictive-coding` `cognitive-architecture` `episodic-memory` `hebbian-learning` `local-learning` `online-learning` `computational-neuroscience` `neuromorphic-computing` `sparse-neural-networks` `world-models` `pytorch` `deep-learning` `artificial-intelligence`（选取依据与实测仓库计数见 14.10）。
- social preview：`usesCustomOpenGraphImage: true`，`openGraphImageUrl` 指向 `repository-images.githubusercontent.com/1301491809/69a87c39-…`，即 GitHub 已完成 CDN 转存。图源为 `frontend/public/social-preview.png`（1280×640，139.9 KB），由 `scripts/make_social_preview.py` 生成，改文案后重跑即可重出图；脚本内置两道非零退出自检（墨迹 bbox 须落在四边 100px 安全边距内、产物须 <1 MB），首版两个已修正的缺陷记录见 14.10 上游条目。图不走仓库文件系统，仓库内保留 PNG 与脚本仅为可复现。
- homepageUrl：**刻意留空**。项目暂无独立站点，填 README 锚点等于制造一个自指链接，对访客无增量信息。

后续若要让 agent 自行改这类元数据，唯一有效做法是换执行主体：提供带 `Administration: Read and write` 的 fine-grained PAT 作为 `GH_TOKEN`（或给该 App installation 补 Administration 权限），届时下面两条命令即可放行。

```bash
gh repo edit liulang5945-netizen/Seed --description "Byte-level predictive-coding kernel that learns online from local prediction errors: no backpropagation, no attention matrix, no optimizer. Sparse fixed-fan-in synapses, slot-free distributed episodic memory, lesion-controlled reproducible experiments."

gh api -X PUT repos/liulang5945-netizen/Seed/topics \
  -f "names[]=predictive-coding" -f "names[]=cognitive-architecture" \
  -f "names[]=episodic-memory" -f "names[]=hebbian-learning" \
  -f "names[]=local-learning" -f "names[]=online-learning" \
  -f "names[]=computational-neuroscience" -f "names[]=neuromorphic-computing" \
  -f "names[]=sparse-neural-networks" -f "names[]=world-models" \
  -f "names[]=pytorch" -f "names[]=deep-learning" \
  -f "names[]=artificial-intelligence"
```

注意 social preview 例外：GitHub 从未提供该字段的 REST/GraphQL 接口，**任何令牌都写不了**，只能人工在 Settings → Social preview 上传，换 PAT 也不能自动化这一项。

**已完成：`TSKV8Adapter.step_cross_region_network()` 已把 growth/pruning/split/merge 所需的 activity、route evidence、prediction error、learning gain、holdout transfer 和资源压力接入可 checkpoint 的 runtime observation；Gate 为 `reports/taiji_runtime_structure_20260826.json`。无 expected activity 时不伪造 growth supervision，route credit 来自实际 target activity，runtime tick 不直接改变 topology。**

**已完成：真实 runtime evidence 已汇聚为有边界、可 checkpoint、按 substrate 去重的 `StructuralProposalCandidate` 队列；候选覆盖 split、region/connection prune 和兼容 region merge，保存证据、source tick、priority、参数与 resource cost，且不会绕过 ledger 直接改变 topology。**

**已完成：candidate 可幂等 materialize 为 pending `StructuralTopologyProposal`，candidate→proposal lineage 随 native checkpoint 恢复；materialization 不改变 live topology。**

**已完成：split、region-prune、connection-prune、merge candidate 已接入统一 holdout validator dispatch；验证只更新 pending proposal 的 validation score/status，未验证 candidate 仍被 commit gate 阻断。**

**已完成：统一 commit/rollback dispatcher 已按 topology role 路由 candidate，依次执行 holdout score、budget、trial checkpoint、live topology mutation 和 latest-change reverse rollback；runtime Gate 覆盖 commit 后拓扑变化、父结构恢复和 checkpoint continuation。**

**已完成：candidate queue 已与真实 holdout 数据绑定为逐项 fail-closed maintenance cycle；两个 runtime candidates 完成 materialize、holdout、commit、rollback，缺数据/异常/预算不足不会绕过 ledger，`StructuralMaintenanceResult` 随 native checkpoint 恢复。**

**已完成：直接 neuron birth (`add`) 已纳入同一 candidate contract；`TSKV8Adapter.step_adaptive_neuron_region()` 从真实 standalone region tick 生成带 substrate/evidence/source tick/priority/resource cost 的候选，统一 materialize、holdout validator、commit/rollback dispatcher 和 native checkpoint 均已通过 `reports/taiji_runtime_structure_20260826.json` 的 direct-add 子门禁。**

**已完成：maintenance cycle 已具备显式 candidate dependency/conflict 判定；反向输入仍按依赖拓扑顺序执行，依赖失败会阻断下游，同一 substrate 的竞争变更全部 `failed_closed`；不同 neuron identity 的 `add` 可并存并按依赖连续出生，队列只对同一目标 unit 去重。**

**当前唯一下一步：建立三层以上自适应区域的规模化结构维护 Gate，覆盖跨区域 route、混合 add/split/prune、资源竞争、checkpoint continuation 和拓扑不变量。**

**已完成：三层自适应区域规模化结构维护 Gate 已通过；`source→relay→target` 显式 route 在 connected split 后保留并按受影响边展开，standalone neuron `add` 可与 network split 混合进入同一 maintenance cycle，checkpoint continuation、资源预算和双向 rollback 均通过。**

**当前唯一下一步：对已落地的 native sparse neuron/network runtime 做 CPU/CUDA 实际热点剖析，建立跨设备 checkpoint 恢复与数值一致性基线，再决定是否需要 fused/sparse kernel。**

**已完成：native sparse neuron/network runtime profile 已执行并通过；本机为 `torch 2.13.0+cpu` 且无 CUDA，报告明确记录 CUDA 未执行。CPU region/network 热点、CPU checkpoint roundtrip 和 continuation 均通过，CPU 实测分别约为 `18,735 ticks/s` 与 `5,291 ticks/s`，主要热点为 `aten::_to_copy`、`aten::to`、`aten::index`。**

**当前唯一下一步：消除 native tick 中可避免的设备/标量转换与临时分配，复跑同一 profile 并比较热点/吞吐；在获得 CUDA-capable 主机前不写 fused/sparse kernel。**

**已完成：native tick hardening 已落地并通过；稀疏输入仅在设备不一致时转换，norm 常量按 device/dtype/limit 缓存，network 的 zero scratch vector 按区域复用且不进入 checkpoint。重跑 profile 仍通过 CPU profile、checkpoint roundtrip 与 continuation，热点收敛到 `aten::index`、`aten::sum` 及少量 copy；本机仍无 CUDA，故没有伪造 CUDA 结论。**

**当前唯一下一步：将当前 profile 固化为稳定的性能回归基线，并在 CUDA-capable 主机上复跑同一 workload；在此之前不引入自定义 fused/sparse kernel。**

**已完成：profile 固定 workload 与 manifest 已提交，CPU profile 报告、checkpoint roundtrip/continuation 和 network scratch 复用回归测试均已固化；吞吐仅作为本机观测，不钉死为跨设备硬阈值。**

**当前唯一下一步：在 CUDA-capable 主机上复跑同一 workload，完成跨设备输出/checkpoint continuation 验证，再依据真实热点决定是否引入 fused/sparse kernel。当前 CPU-only 环境不宣称 CUDA 已验证。**

**已完成：`taiji/` 的 autograd 学习平面已整体替换为原生局部信用分配（详见 14.5）。8 个模块 13 处 `loss.backward()` 全部迁移至 `taiji/local_learning.py`，`no_autograd_parameters` 从假绿转为真检查并通过；原生性契约 15/15、8 道阻塞 verify 全 pass、`tests/` 437 passed / 5 skipped、lint 三件套与版本一致性全绿。`LocalAdam` 保留 Adam 更新式，因此未连带改变任何已调优的学习率。**

**已完成：迁移已获 CI 实证。`02f6602` 的 Linux job 上 8 道阻塞 verify 全部转绿，其中 `no_legacy_or_transformer_dependency` 从 `ffe1da2` 的 false 变为 true——该项此前红了 7 次连续构建，根因确为 autograd 学习平面，不是环境差异（详见 14.7）。同一构建暴露出唯一剩余红点 `tests/test_terminal_input_normalization.py` 的 3 条平台未钉定用例，与本次迁移无关，已按 14.5 修复并在本机复现 Linux 条件验证。**

**已完成：`test` 转绿后首次真正执行的 `build-frontend` / `docker-build` 双红已定位并修复（详见 14.8）。两者因 `needs: test` 在此前 7 次连续红期间一直是 skipped、从未运行，故属被遮蔽的既存缺陷而非本次回归。`docker-build` 的 `COPY data/` 已收敛到项目既有的挂载约定；`build-frontend` 的 2 个 high CVE 已用 `overrides` 全树强制到安全版。前端四道门禁本机全绿：audit 退出 0（余 2 moderate 不阻塞）、eslint 0 errors / 17 warnings、vitest 19 files 160 passed、build 成功且 `dist/index.html` 存在，`npm ci --dry-run` 退出 0 证明 lock 与 package.json 同步。Docker 侧因本机无 Docker 未做构建验证，改以 `COPY` 源跟踪文件数静态审计替代，并已如实记录。**

**已完成：`build-frontend` 已在 CI 实证转绿（`32982579047`，1m37s），5 个上游 job 亦全绿（test 3.10 13m25s、test 3.12 13m8s、test-windows 6m27s、两个 startup smoke）。`docker-build` 的 `COPY data/` 根因确认修复——`Build image via docker compose` 与 `Verify Docker image metadata` 均已打勾；但其后从未运行过的 `Startup smoke and healthcheck` 暴露出下一层缺陷 `ModuleNotFoundError: No module named 'seed_platform'`。根因是 Dockerfile 手工 `COPY` 清单漏了 pyproject 已声明的 `seed_platform*`，而 `packages.find` 静默跳过缺失目录使 `pip install` 仍退出 0（详见 14.8）。修法为补齐 `COPY` 并加构建期导入断言 `RUN python -c "import api.app"`，使漏拷贝此后在 build 层即刻失败；清单与 pyproject 已复核对齐（`MISSING: none`），断言语句本机实测退出 0 证明不会误红。**
**已完成：CI 的「基线不可复现」根因已修。`ci.yml` 原只钉 `ruff`/`black`，`mypy`/`pip-audit` 浮动，导致门禁数字在无人改代码时自己漂移（核心 47→63、全仓 259→281）；现已钉 `mypy==2.3.1`、`pip-audit==2.10.1`，并确立通用规则「凡把工具输出数字当阈值的门禁，工具本身必须钉版本」。同时 mypy 核心门禁由 advisory 升为**棘轮 blocking**（初始 `MYPY_CORE_BASELINE=63`，超基线即 `exit 1`，解析不到数字亦 `exit 1` 绝不静默放行，低于基线打 `::notice::` 提示下调；当前基线已收紧为 0）。双矩阵实测否证了「报错数随 Python 版本变化故不能设阈值」——3.10 与 3.12 的核心数 63、全仓数 281 完全相同（详见 14.2）。停机期三次假红已判定为平台产物并记入 14.11。**

**已完成：核心 mypy 类型债已归零。2026-08-27 在当前固定工具链 `mypy==2.3.1` 下，`python -m mypy --follow-imports=silent seed taiji` 对 44 个源文件报告 0 错误；修复覆盖 checkpoint/state_dict 类型收窄、可选值边界、结构化参数契约、局部 GRU 学习张量和可替换语言器官协议。定向 ruff 通过，相关回归 46 passed。`.github/workflows/ci.yml` 的 `MYPY_CORE_BASELINE` 已同步从 63 收紧至 0。**

**已完成：核心类型债提交 `11ca75c` 已在当前 `main` 与 `origin/main` 同步；本地 `mypy==2.3.1` 对 `seed taiji` 的 44 个源文件保持 0 错误。GitHub Actions 实跑仍未因 CLI 未认证而声称完成。**

**已完成：P2/A1 感知训练—运行时边界合同已修正。`LearnedPerception.fit_predictive()` 现在复用与 `observe()` 相同的 prediction-error、surprise baseline、hysteresis 和 maximum-duration 边界时钟；训练使用动态 assembly 的每个活动前缀监督，运行时与训练不再分别使用固定滑窗/可变切段。新增训练 rollout 与 runtime boundary 的回归测试；定向 P2 回归 8 passed，完整 `tests/taiji_native` 为 192 passed、1 skipped，另有 2 个 Windows pytest 临时锁 setup error。旧 next-byte A1 在真实 manifest 上仍未通过，说明评测任务本身还需继续提高语义层级，不能把这次合同修复冒充 Gate 通过。**

**已决定：CUDA 相关 profile、跨设备 checkpoint 和 fused/sparse kernel 暂缓，直到具备 CUDA-capable 主机；本轮继续推进 CPU 可验证的 Taiji 能力，不修改 CUDA 结论。**

**已完成：动态 assembly pooled state 已从无序均值提升为可学习的顺序敏感读出。`assembly_recency_logit` 通过正值 softplus 增益学习当前活动的 recency 权重；训练、运行时和 checkpoint 共用同一池化公式，不新增固定词表或 Transformer 组件。训练暴露统计又形成连续 novelty 信号，参与 boundary competition 并随 checkpoint 保存、在线更新；checkpoint 往返、mypy 0、ruff 通过，P2 定向回归 9 passed。**

**已完成：A1 边界合同已改为以 marker 位置的 boundary rate 作为因果指标；整段 aggregate boundary rate 仍保留为诊断，因为插入 marker 会改变序列长度并重排邻近 assembly。**

**已完成：A1 Gate 已收紧为所有 seed 的最差 generalization、marker score/rate 和 random-chunk drop，而不是只看 primary seed；正式报告 `reports/taiji_a1_perception_20260827.json` 如实为 `gate_passed=false`：primary gain=`+0.0089`，但最差 seed gain=`-0.0398`，最差 random-chunk drop=`+0.0030`，marker score/rate 最差仍为 `+0.1299/+0.1095`。这关闭了“单个幸运 seed 收口”的评测漏洞，同时证明 novelty 已修复边界响应但未完成稳健组合迁移。**

**已完成：A1 predictive temperature 从 `0.15` 调整为 `0.5`，同一 32/16 smoke manifest 在严格最差 seed 口径下通过：三 seed 的 generalization gain=`+0.0022/+0.0035/+0.0231`，random-chunk drop=`+0.0135/+0.0092/+0.0116`，marker score/rate 最小=`+0.2025/+0.4262`，报告为 `reports/taiji_a1_perception_20260827.json`。**

**已完成：独立规模化验证已执行于 `shared_core` 的 128 train / 64 holdout manifest，报告为 `reports/taiji_a1_perception_shared128_20260827.json`；跨 seed std=`0.0048`、marker score/rate 最小=`+0.2872/+0.5606`，但最差 generalization gain=`-0.0058`、最差 random-chunk drop=`+0.0023`，Gate 仍为 `false`。这说明 temperature 修复和边界 novelty 已有效，但 assembly 的组合迁移和 lesion 抗性尚未规模化稳定。**

**已完成：顺序敏感 assembly 已加入可 checkpoint 的多步预测信用分配；`multi_step_prediction_weight=0.05`、`horizon=4` 纳入 A1 默认，误差沿连续 transition 展开并回写 assembly/transition/embedding 的原生局部梯度。smoke 32/16 在三 seed 下 Gate 通过；128/64 独立 manifest 仍为 false（最差 gain=`-0.0010`、最差 random drop=`-0.0001`），故没有用多步模块掩盖规模化失败。**

**已完成：跨 assembly 边界结构已落地。训练先按与 runtime 相同的闭合 boundary 映射出“边界后的下一段”，该段不跨越下一条 runtime boundary；可选的 boundary-after 多步 CE 已实现但默认权重为 `0.0`，默认使用跨后续 assembly 的多步对比负样本（`cross_assembly_negative_weight=0.01`），对不同 boundary 后段的上下文进行显式正/负匹配。该目标通过 native local-credit 路径回写 embedding/transition，未引入 token 表、固定段表或 Transformer。checkpoint 往返、定向回归 `10 passed`、核心 mypy 0、Ruff 和 Black API 检查均通过；relation subgate 复核为 true。**

**已完成：A1 感知 Gate 已在两级规模正式通过。smoke `32/16` 报告 `reports/taiji_a1_perception_20260827.json` 的最差泛化=`+0.00310`、最差 random-chunk drop=`+0.00578`、marker score/rate 最小=`+0.1734/+0.3567`、cross-seed std=`0.00608`；`shared_core` `128/64` 报告 `reports/taiji_a1_perception_shared128_20260827.json` 的最差泛化=`0.0`、最差 random-chunk drop=`+0.00527`、marker score/rate 最小=`+0.2161/+0.4483`、cross-seed std=`0.00834`。两份报告均为 `gate_passed=true`；完整 `tests/taiji_native` 为 `193 passed, 1 skipped, 2 errors`，两个 error 仍是 Windows pytest 临时目录锁 setup 权限问题，未进入测试体。**

**已完成：P2→P3 lineage contract 已接入 runtime。`WorkspaceState` 与 `WorldState` 新增可选的 `percept_event_id`、`percept_assembly_id`、`percept_boundary_closed`；`TSKV8Adapter.observe()` 在生成 lineage 后同步写入两者，`observe_event(world_state=...)` 与 `settle_action(world_state=...)` 的外部状态替换都会保留当前来源，native checkpoint/restore 可恢复。closed boundary 若缺 event/assembly ID 会 fail closed；定向 world/concept/v1 回归 `21 passed`，Ruff、核心 mypy 0 通过。该改动只建立 provenance contract，不把 lineage 存在冒充为跨 episode 能力。**

**已完成：P2→P3 perception-to-world closure Gate 已通过。`scripts/training/eval_taiji_p2_p3_closure.py` 用 64 train / 32 新对象与新候选组合 holdout、3 seeds 驱动真实 `TSKV8Adapter`；每个样本先经历两次 raw observation 和 boundary-closed percept，再把 learned/none workspace 选择绑定到 `TaijiWorldState` 的对象—关系 transition。报告 `reports/taiji_p2_p3_closure_20260827.json` 的 learned route/world transition 最差均为 `1.0`，none lesion 最高为 `0.0`，lineage 最差为 `1.0`，192 次 boundary-closed assembly、三 seed checkpoint continuation 和 world checkpoint roundtrip 全部通过；shared16 relation subgate 复核为 true。该 Gate 关闭 provenance-to-world 的窄闭环缺口，不宣称长程世界模型或开放域语义智能。**

**已完成：P2→P3 variable-horizon continuation Gate 已通过。`scripts/training/eval_taiji_p2_p3_variable_horizon.py` 在 64 train / 32 holdout、3 seeds、3/4/5 个 closed assembly 上驱动真实 `TSKV8Adapter`；learned route/world success、lineage、两步 history、checkpoint continuation、`TaijiWorldState` roundtrip 和 runtime `WorldDynamicsLearner` online calibration 均为 `1.0`，workspace lesion route 为 `0.0`。第一步后通过真实 bridge observation 消费 pending experience，再从 native checkpoint 续接第二步；第二步使用训练 schema 未见的 `secured` relation，`assembled → secured` progression Gate 为 `1.0`。新增同 tick `TaijiWorldState.synchronize_observation()` 只同步观察快照、不伪造 action transition，并保持历史 checkpoint 连续。该 Gate 证明变量时长与跨 checkpoint 的两步因果续接已成立，但未知 relation 当前只保证被保存和传递，不宣称 world learner 已完成开放集关系预测。报告为 `reports/taiji_p2_p3_variable_horizon_20260827.json`，manifest 为 `reports/taiji_p2_p3_variable_horizon_manifest_20260827.json`。**

**已完成（旁支，不改主线）：公网 demo 的前置核查已出结论，记入 14.13。`taiji/` 对 torch 的依赖是"张量库"而非"自动微分引擎"（0 处 `.backward()` / 0 处 `torch.optim` / 0 处 `torch.autograd`，77 处 `no_grad`），算子面为 43 个初等 torch 函数 + 8 种 nn 层，`taiji.model.Taiji` 的推理闭包仅 8/35 模块 6623 行，checkpoint 为自定义纯 dict 格式且零 `torch.save`。故 WASM 内核 demo 技术上可行；但 Pyodide+shim 与 Rust/C++ 重写的二选一尚未尽调，未做决定。本项不占用主线唯一下一步。**

**已完成：P2→P3 open-set world schema evolution Gate 已通过。** `WorldSchema` 现在以语义 feature key 扩展 object、numeric attribute、relation、action kind、actor、target 和 parameter；`WorldDynamicsLearner` 可在运行时注册新边界、按语义迁移旧 input/output 权重、保留 checkpoint 并通过真实 `WorldTransition` 的 outcome feedback 在线校准。`TSKV8Adapter` 的 runtime 适配路径不会把 `action_symbol` 等控制元数据误注册为 world 参数；`begin_episode()` 保留 world/events/concepts/calibration 和已学网络，只清理 episode transient。`TaijiWorldState.advance_observation()` 补齐了无动作感知跨 tick 的 owned snapshot 推进，action history 仍只记录真实 transition。正式报告 `reports/taiji_p3_open_set_20260827.json` 在 64 train / 32 holdout、3 seeds 下通过：learned route/world、relation progression、open object/relation/action、lineage、跨 episode、checkpoint、四 transition ownership、history/roundtrip、calibration 均为 `1.0`，workspace lesion route 为 `0.0`；完整 native 回归为 `203 passed, 1 skipped`，另有 2 个 Windows pytest 临时目录锁 setup error，未进入测试体。该 Gate 证明的是可扩展 schema 与窄因果闭环，不是开放域语义理解或通用智能。

**当前唯一下一步：建立跨 episode 的 Taiji world schema registry 与可回滚生命周期 Gate。** 将当前单 learner 的即时扩展提升为可审计 registry：统一 canonical identity/alias、关系 predicate 的新增与冲突、slot confidence、checkpoint lineage、版本回滚，以及在资源上限下的 merge/prune/tombstone；用混合旧/新 schema 的 holdout 和矛盾 outcome 验证不会静默覆盖旧知识，CUDA 继续暂缓。

**已完成：P3 world schema registry lifecycle Gate 已通过。** `WorldSchemaRegistry` 为 schema 提供 revision proposal/commit/rollback、canonical object alias、slot confidence、矛盾 feedback 记录、资源预算、prune+tombstone 和 checkpoint lineage；`WorldDynamicsLearner` 保存每个 revision 的网络快照，可在不重置旧权重的前提下回滚 schema 与网络，`TSKV8Adapter` 原生 checkpoint 同时恢复 registry 和多版本快照。正式报告 `reports/taiji_p3_schema_registry_20260827.json` 在 3 seeds 下全部通过：alias 稳定与冲突拒绝、旧/新 schema 混合预测、旧权重保留、矛盾反馈 fail-closed、预算阻断、prune/tombstone、rollback 和 checkpoint rollback 均为 `1.0`。生命周期单测 3 passed；核心 mypy 0、Ruff/Black 通过；native 全量为 `208 passed, 1 skipped`，另有 2 个既有 Windows pytest 临时目录锁 setup error，未进入测试体。该 Gate 验收 schema 生命周期安全，不宣称开放域关系语义已解决。

**已完成：P3 registry adjudication 已接入真实 adapter `WorldTransition` outcome 闭环。** `WorldSchemaRegistry` 对不含 tick/event 噪声的 semantic before/action 生成稳定 evidence key，并保存 after-state/reward/success outcome signature；跨 episode 的一致结果会增加主假设置信度，矛盾 after-state 会写入 conflict ledger 且不覆盖既有证据。`WorldDynamicsLearner.online_update()` 在本地信用分配前执行 adjudication，矛盾样本 fail-closed，不增加 `online_updates`；`TSKV8Adapter` 的 `WorldCalibrationTrace.calibration_applied` 与实际接受的 update 对齐，并把 accept/reject 计数和 registry evidence 一同纳入 native checkpoint。真实 adapter Gate `scripts/training/eval_taiji_p3_transition_adjudication.py` 在 seeds `11/29/47` 全部通过：跨 episode confidence=`1.0`、relation-specific contradiction、prediction calibration、no-update-on-reject、registry/network checkpoint 和 continuation 均为 `1.0`；报告/manifest 为 `reports/taiji_p3_transition_adjudication_20260827.json` 与 `reports/taiji_p3_transition_adjudication_manifest_20260827.json`。定向闭环为 `23 passed`；native 全量为 `211 passed, 1 skipped, 2 errors`，2 个 error 仍是 Windows pytest 临时目录锁 setup 权限问题，未进入测试体；核心 mypy 0、Ruff/Black 通过。该 Gate 证明真实反馈的证据一致性与 fail-closed ownership，不宣称随机世界建模或开放域关系语义已解决。

**已完成：P3 多假设 outcome ledger Gate 已通过。** transition evidence 不再只保留单一结果签名，而是按 semantic context 记录多个 after-state/reward/success 假设、evidence count 和 outcome probability；一过性矛盾进入 `conflicted`，两个及以上候选都达到重复支持后进入 `stochastic`。`WorldDynamicsLearner` 只在当前样本属于拥有明确 lead 的主假设时更新，平票或少数结果 fail-closed；`TSKV8Adapter` 的 calibration trace 反映真实接受/拒绝结果。3 seeds 的 deterministic/stochastic lesion、relation-specific holdout、跨 episode、registry/network checkpoint continuation 全部通过，stochastic 主假设 confidence=`0.6`；报告/manifest 仍为 `reports/taiji_p3_transition_adjudication_20260827.json` 与 `reports/taiji_p3_transition_adjudication_manifest_20260827.json`。新增 ledger 单测与真实 adapter Gate 通过；CUDA 继续暂缓。该 Gate 证明有限证据下的结果不确定性边界，不宣称已完成概率世界预测。

**已完成：P3 ledger outcome 分布已接入 world prediction 与 uncertainty。** `WorldSchemaRegistry` 保存每个 outcome hypothesis 的 reward/success 统计，并统一返回 `unseen`、`deterministic`、`conflicted`、`stochastic` 四种不确定性语义；`WorldDynamicsLearner.predict()` 对已观测 context 使用 ledger 的经验 outcome estimate，对未见 context 保留网络预测但标记最高 uncertainty。`WorldPredictionRecord`、native checkpoint 和规划器候选都传递同一 uncertainty；真实 adapter Gate 验证 known=`0.0`、conflicted=`1.0`、stochastic=`0.5`，3 seeds 全部通过，核心 mypy 0、Ruff/Black 通过。该 Gate 关闭了“模型没见过”和“环境本身多结果”混为一谈的接口漏洞，但不等价于概率校准质量已达标。

**已完成：P3 ledger-driven probability calibration Gate 已通过。** `scripts/training/eval_taiji_p3_probability_calibration.py` 使用独立 relation-specific holdout，只测量不写回 transition ledger；3 seeds 的 success Brier=`0.24`、binary NLL=`0.6730`、组级 confidence coverage=`1.0`、reward MAE=`0`，holdout evidence count 未变化。未知 target/`tracks` relation 保持 `unseen/1.0`，已观测随机 context 保持 `stochastic/0.4`，native checkpoint continuation 与无 world learner 的 planner lesion 全部通过；报告/manifest 为 `reports/taiji_p3_probability_calibration_20260827.json` 与 `reports/taiji_p3_probability_calibration_manifest_20260827.json`。该 Gate 证明测量边界和 ledger 不泄漏，不宣称开放世界概率预测质量。

**已完成：P3 outcome hypothesis 分布已接入多步 imagined rollout 的风险敏感规划。** `PlanningCandidate` 与 `ImaginedRollout` 现在携带 `uncertainty_mode`；`TSKV8Adapter.predict_world_candidates()` 和 `imagine_world_rollout()` 将 world learner 的 `unseen/deterministic/conflicted/stochastic` 语义传入规划层，`GoalPlanner` 按可配置 multiplier 对不同风险类型施加惩罚，并在单步与多步 rollout 使用同一公式。概率校准 Gate 已扩展为 3 seeds 的风险敏感单步/多步选择、relation holdout、ledger 隔离、checkpoint continuation 与 world-model lesion，全部通过；报告/manifest 为 `reports/taiji_p3_probability_calibration_20260827.json` 与 `reports/taiji_p3_probability_calibration_manifest_20260827.json`。该 Gate 证明风险语义已贯穿 imagined scoring，不宣称真实环境已自动执行整条随机 rollout。

**当前唯一下一步：建立 stochastic/conflicted rollout 的真实逐步执行闭环。** 让真实 environment 每一步的 after-state 回写 ledger 与 prediction trace，成功/失败或歧义时按 risk mode 中止或重规划，并验证从中断 checkpoint 继续不会重复消费或污染 outcome evidence；CUDA 继续暂缓。

**已完成：P3 stochastic/conflicted risk-sensitive execution Gate 已通过。** `TSKV8Adapter.execute_imagined_rollout_step()` 现在把真实 environment 的 after-state、outcome adjudication、ledger uncertainty 和证据数量写入 `WorldCalibrationTrace`；ledger 拒绝不明确结果时 fail-closed，不增加 world learner 的 `online_updates`，并触发当前 rollout 中止/重规划。非终止失败或负奖励也会清空剩余 imagined plan 并要求重规划。`scripts/training/eval_taiji_p3_risk_sensitive_execution.py` 在 seeds `11/23/37` 下验证了 stochastic (`uncertainty=0.4`) 与 conflicted (`uncertainty=1.0`) 两类风险、真实失败、恢复 rollout、trace checkpoint 恢复和 checkpoint 前后不重复消费证据，三 seed cross-seed gate rate=`1.0`；报告/manifest 为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`。定向风险/ledger/probability/P7 回归 `13 passed`，核心 mypy 0、Ruff/Black 通过。该 Gate 证明真实执行边界已能识别并阻断不可靠 outcome，不宣称已完成自动分支搜索或开放世界智能。

**当前唯一下一步：把布尔 `replan_required` 提升为可审计的 outcome-aware recovery branch contract。** 记录触发重规划的 transition evidence key、风险模式、被拒绝分支与剩余 rollout lineage；由 planner 生成并持久化排除失败分支后的候选 recovery rollout，checkpoint 续跑后自动恢复同一 branch context，并用冲突/随机/真实失败混合 episode 验证不会重复选择已拒绝分支；CUDA 继续暂缓。

**已完成：P3 outcome-aware recovery branch contract Gate 已通过。** 新增可 checkpoint 的 `RecoveryBranchState`，保存原 rollout/goal lineage、被拒绝 `WorldAction`、semantic evidence key、风险模式、剩余步数、触发原因和替代 rollout；`TSKV8Adapter.plan_rollouts()` 在恢复上下文中按 action semantic key fail-closed 过滤被拒绝首步，若没有替代分支则拒绝规划。真实执行中的 stochastic/conflicted adjudication rejection 与非终止失败都会建立 branch context，terminal recovery 完成后清理；native checkpoint 往返保留 branch，且不会重复写入 ledger evidence。三 seed 风险执行 Gate 与相关回归全部通过，核心 mypy 0、Ruff/Black 通过。该 Gate 证明 recovery branch 的 ownership、过滤与持久化边界，不宣称已经完成大规模分支搜索或开放世界智能。

**当前唯一下一步：建立多分支 recovery rollout 的真实选择与反事实评估 Gate。** 在同一失败上下文生成至少两个语义不同的替代 rollout，使用 ledger 风险、预测误差和真实后果对候选逐一评估，验证 planner 选择可执行且风险最低的分支；同时保持被拒绝首步不可重入、checkpoint continuation 不重复消费证据，CUDA 继续暂缓。

**已完成：P3 multi-branch recovery rollout Gate 已通过。** 恢复规划现在可同时接收被阻断分支、低风险 deterministic 分支和高风险 unseen 分支；planner 先按 recovery branch 的 action semantic key 排除已拒绝首步，再用 ledger uncertainty、预测 reward/success、预测误差和资源项统一评分，选出可执行的最低风险替代分支。三 seed 真实 environment Gate 均通过：被拒绝分支未重入、deterministic recovery 优于 unseen counterfactual、真实恢复结果成功终止、trace/checkpoint 与 evidence accounting 保持一致。该 Gate 验证分支选择和反事实风险排序，不宣称 recovery candidate 已能从开放世界自动生成。

**当前唯一下一步：把 recovery candidate synthesis 收回 Taiji-owned runtime。** 从当前 world affordance/schema/ledger 状态自动产生语义不同的替代 action 与 imagined rollout，统一做 schema 可执行性、预算和被拒绝分支过滤，再交给现有 risk planner；保留外部候选注入作为受控测试接口，CUDA 继续暂缓。

**已完成：P3 Taiji-owned recovery candidate synthesis Gate 已通过。** `TSKV8Adapter.synthesize_recovery_rollouts()` 从当前 `WorldAffordance`、可用 action-kind/motor capability、horizon 与 resource budget 生成结构化 recovery rollouts，并在生成阶段过滤被拒绝 action semantic key；生成结果继续经过 world learner 投影、risk planner 和 branch lineage checkpoint。三 seed 真实 environment Gate 验证了 `assemble` 被过滤、`idle` deterministic 低风险分支优于 `secure` unseen 反事实分支，最终执行成功且相关 trace/evidence/checkpoint 保持一致。该 Gate 关闭了 recovery 候选由调用方手工拼接的边界，不宣称 affordance 生成已经覆盖开放世界。

**当前唯一下一步：建立 recovery affordance 的真实可执行性与预算闭环。** 将候选生成的 resource cost、motor capability、world schema 可编码性和环境实际拒绝结果统一为可审计 Gate；对不可执行/超预算候选 fail-closed，并验证 checkpoint 续跑不会把被拒绝候选重新注入，CUDA 继续暂缓。

**已完成：P3 recovery affordance executable/budget Gate 已通过。** `synthesize_recovery_rollouts()` 现在校验 action/motor 对齐、唯一 action-kind、motor alphabet 范围和 resource budget；从当前 affordance 生成的 action 先经过 world learner 编码/投影，再进入 recovery branch 过滤与 risk planner。三 seed 真实 Gate 均通过：被拒绝 `assemble` 不重新注入，超预算 `archive` 不进入候选，不在当前 motor capability 中的 affordance 不进入候选，`idle` deterministic 分支仍被实际环境执行并完成 terminal recovery；checkpoint、ledger evidence 与 adjudication trace 保持一致。核心 mypy 0、Ruff/Black 和相关 P7/P3 回归全绿。该 Gate 证明 runtime 可执行性与资源边界，不宣称 affordance 生成已达到开放世界完整性。

**当前唯一下一步：建立真实 environment capability discovery 与 recovery affordance freshness Gate。** 让环境在每个 step 返回的可用 action/motor capability 参与下一轮 recovery synthesis，并用 capability 变化、过期 affordance、schema 新增和 checkpoint continuation 验证旧候选不会越权或复用；CUDA 继续暂缓。

**已完成：P3 environment capability discovery 与 recovery synthesis freshness Gate 已通过。** `EnvironmentOutcome` 现在可携带当前 step 的 `available_actions/action_kinds`，adapter 将其收敛为可 checkpoint 的 `EnvironmentCapability`，并在 `begin_episode()` 清除旧 episode 能力；恢复候选在未显式传入能力时只消费当前快照，显式能力与快照不一致时 fail-closed，同时要求 capability tick 与当前 world tick 对齐。三 seed 真实 environment Gate 验证了：能力边界被发现并持久化、checkpoint continuation 保留能力、step 后能力刷新会约束下一轮候选，`archive` 超预算和不在能力边界的候选均不会生成；报告 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 的 cross-seed gate rate=`1.0`，相关定向回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明当前能力快照的发现、恢复与下一轮 synthesis 约束，不宣称已完成对已持有 pending rollout 的执行时重验证。

**当前唯一下一步：建立 pending recovery rollout 的执行时 capability/affordance freshness Gate。** 为每个 synthesized rollout 保存 capability tick、affordance identity 与 world-schema revision；在 `plan_rollouts()` 和真正执行前重新校验这些 lineage，环境能力、affordance 或 schema 变化时自动失效旧候选并要求重新 synthesis，CUDA 继续暂缓。

**已完成：P3 pending recovery rollout capability/schema freshness Gate 已通过。** 新增 `RecoveryRolloutLineage`，为 Taiji-owned recovery rollout 保存 capability tick、完整 action/action-kind 快照、affordance identity 和 world-schema revision；`plan_rollouts()` 会过滤或拒绝过期候选，真正执行前再次 fail-closed，过期候选不会触发环境动作。候选 synthesis 先统一完成本批 schema 注册，再为整批 rollout 记录同一最终 revision，避免生成未知 action kind 时把同批候选误判为过期；planned rollout、lineage 和 capability 均通过 native checkpoint 恢复。三 seed Gate 验证 planning-time rejection、execution-time rejection、lineage checkpoint continuation 与 terminal recovery 全部为真，报告 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 的 cross-seed gate rate=`1.0`；定向回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明 pending rollout 的边界版本控制，不宣称同一 affordance ID 下参数内容变化已具备内容指纹校验。

**当前唯一下一步：建立 affordance content identity 与 schema-bound action validation Gate。** 为 affordance 生成稳定的内容指纹（action kind、actor/target、规范化参数和 grounding lineage），并将其写入 recovery lineage；在 planning/执行前校验当前 affordance 内容及 action semantic key，任何同 ID 内容替换、action 参数漂移或 schema 编码不一致都必须失效旧候选并重新 synthesis，CUDA 继续暂缓。

**已完成：P3 affordance content identity 与 schema-bound action validation Gate 已通过。** `WorldAffordance.content_identity` 基于 action kind、actor/target、规范化参数和 grounding lineage 生成稳定指纹；`RecoveryRolloutLineage` 同时保存该指纹与第一步 action semantic key。`plan_rollouts()` 和真实执行前都会校验 capability 快照、affordance 内容、schema revision 及 action symbol 映射，同 ID 内容替换、action 参数漂移、schema/action semantic key 不一致均 fail-closed，且不会触发环境动作。三 seed 风险执行 Gate 的 lineage 记录、checkpoint continuation、stale planning、stale execution 全部为真，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明恢复候选的内容身份和执行边界可审计，不宣称已对所有非 recovery planner 候选建立同等 lineage 约束。

**当前唯一下一步：将 freshness contract 扩展到多步 recovery rollout 的逐步 affordance revalidation。** 每次真实 step 成功后重新绑定下一步的 affordance/content identity、capability snapshot 与 schema revision；若下一步候选不再存在或环境边界变化，自动截断旧 suffix、保留 recovery branch 并要求重新 synthesis，而不是只在下一次执行入口才发现失效，CUDA 继续暂缓。

**已完成：P3 multi-step recovery suffix revalidation Gate 已通过。** 非终止成功 step 后，adapter 不再原样保留旧 suffix，而是以 post-action world tick 重新预测 suffix，并重绑下一步的 capability snapshot、affordance content identity、action semantic key 与 schema revision；若任一边界已失效，则当场截断 suffix、保留 recovery branch 并要求重新 synthesis。三 seed 真实两步 environment Gate 验证了第一步成功后的 suffix rebind、第二步 terminal recovery、两次真实 evidence 写入及 checkpoint lineage continuation，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明多步恢复不会盲目复用旧预测 suffix，不宣称未来任意 action 序列都能在当前 capability 未知时安全预绑定。

**当前唯一下一步：建立 recovery branch 的动态再规划与预算累积 Gate。** 将已消费 step 的 resource cost、剩余预算、失败/拒绝次数和 branch lineage 作为持久状态；每次 suffix 失效后由 Taiji-owned synthesis 在剩余预算内重新生成候选，禁止通过 checkpoint 或重复 rebind 绕过累计预算，CUDA 继续暂缓。

**已完成：P3 recovery branch dynamic replanning 与 cumulative budget Gate 已通过。** `RecoveryBranchState` 现在持久化总预算、已消费资源、failure/rejection counters，并提供剩余预算；恢复 synthesis 首次绑定 branch budget，之后只按剩余预算过滤候选。真实执行的 recovery step 会累加实际 candidate resource cost，环境失败与 ledger rejection 分开记账，checkpoint continuation 保持这些累计状态，不能通过重绑定或重新 synthesis 复原预算。三 seed Gate 验证了 checkpoint budget preservation、成功 step 消费 `0.2`、剩余 `0.8` 时 `0.9` 的 secure 候选被阻断、rejection/failure accounting 与 terminal recovery 全部通过，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明 recovery branch 的资源与风险记账闭环，不宣称已实现跨多个并行 branch 的全局资源仲裁。

**当前唯一下一步：建立并行 recovery branch 的全局预算与公平选择 Gate。** 当同一失败上下文产生多个候选 branch 时，维护 branch-owned 与 episode-global 两级预算，按风险/进度/资源做可审计仲裁；验证多个 checkpoint continuation、branch 淘汰和失败重试不会重复消费全局资源，CUDA 继续暂缓。

**已完成：P3 branch-owned + episode-global recovery budget Gate 已通过。** 新增可 checkpoint 的 `RecoveryBudgetState` 作为 episode-global ledger，与 `RecoveryBranchState` 的 branch budget/consumption 分层；每个 recovery step 使用唯一 action identity 幂等扣费，checkpoint 恢复和重复 replay 不会重复消费；失败和 ledger rejection counters 分开累计。Taiji-owned synthesis 同时受 branch 剩余预算和 global 剩余预算约束，三 seed 真实 Gate 验证了 parallel candidate 中高成本分支在累计消耗后被阻断、global ledger checkpoint preservation、duplicate consumption blocking、failure/rejection accounting 和 terminal recovery 全部通过，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明两级资源账本与幂等扣费，不宣称多个并行 branch 已作为独立持久对象同时运行。

**当前唯一下一步：建立持久化 recovery branch portfolio 与公平仲裁 Gate。** 将同一失败上下文的多个候选从一次性 tuple 提升为带 branch identity、状态（active/selected/pruned/expired）、风险/进度/资源审计的 portfolio；checkpoint 续跑后恢复全部候选状态，按全局预算公平选择并可淘汰 branch，禁止被淘汰候选重新注入，CUDA 继续暂缓。

**已完成：P3 持久化 recovery branch portfolio 与公平仲裁 Gate 已通过。** 新增 `RecoveryPortfolio`，为同一失败上下文的候选保存唯一 generation/branch identity、完整 imagined rollout、`active/selected/pruned/expired` 状态、revision 和 retired branch 集合；Taiji-owned synthesis 每轮生成唯一候选 ID，重新合成会退休上一轮 ID，防止旧候选伪装成新候选回流。adapter 在 planning 前只向 risk planner 暴露当前 active/selected 分支，选择后持久化唯一 selected 状态，显式淘汰分支不会重新注入；多步 suffix rebind 会更新 portfolio 中对应候选。普通 checkpoint 与 native checkpoint 都恢复 portfolio 全量候选、状态和 lineage。三 seed 真实 Gate 的 `portfolio_selection_audited`、`portfolio_pruned_not_reintroduced`、`checkpoint_portfolio_preserved` 全部为真，cross-seed gate rate=`1.0`；报告/manifest 为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`，相关回归 `4 passed`，核心 mypy=`0`、Black 全绿。该 Gate 证明并行候选的状态 ownership、选择、淘汰和 checkpoint 恢复，不宣称跨 episode 的长期 portfolio archive 或大规模 branch scheduler 已完成。

**当前唯一下一步：建立跨 episode recovery portfolio archive 与 branch liveness Gate。** 在不污染新 episode transient 的前提下，持久保留可复用的候选 lineage/结果摘要，定义 branch 的 completed/abandoned/expired 生命周期与容量淘汰；验证 episode 切换、checkpoint continuation 和长期预算边界不会复活已终止或已淘汰 branch，CUDA 继续暂缓。

**已完成：P3 跨 episode recovery portfolio archive 与 branch liveness Gate 已通过。** 新增有容量上限的 `RecoveryPortfolioArchive` 和不可执行的 `RecoveryArchiveEntry`：archive 只保存 source episode、portfolio/rollout identity、action lineage、capability/affordance 摘要、resource cost、outcome 与 `completed/abandoned/expired` 生命周期，不把候选重新暴露为可执行对象。terminal 成功 branch 记录为 `completed`，episode 切换清除当前 portfolio/capability 但保留 archive；archive 采用配置化容量并淘汰最旧条目，planner 对 archive 中的 rollout ID fail-closed，不能跨 episode 复活旧 branch。普通/native checkpoint 同时恢复 archive 和生成序号，避免同 episode 重用 branch ID；当前仍执行的 selected suffix 在新一轮 synthesis 时保留到 portfolio，避免归档时丢失最终 outcome。三 seed 真实 Gate 的 archive lifecycle、checkpoint、capacity eviction、archived-branch rejection 和 episode transient isolation 全部为真，cross-seed gate rate=`1.0`；报告/manifest 为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`，相关回归 `4 passed`，核心 mypy=`0`、Black 全绿。该 Gate 证明跨 episode 的 recovery 生命周期和不可复活边界，不宣称 archive 已形成可泛化的长期策略学习器。

**当前唯一下一步：建立 recovery outcome 到长期策略记忆的受控写入 Gate。** 仅允许 completed/有足够证据的 branch 摘要进入 Taiji 的长期 procedural/semantic memory，禁止 abandoned/expired 或冲突 outcome 污染策略记忆；验证跨 episode replay、证据阈值、checkpoint continuation 与错误策略撤销，CUDA 继续暂缓。

**已完成：P3 recovery outcome 到长期策略记忆的受控写入 Gate 已通过。** 新增 `RecoveryStrategyLedger` 与 `RecoveryStrategyApproval`：terminal completed branch 只有在 `evidence_count` 达到配置阈值后，才按对应 episodic memory record 准入 recovery consolidation；abandoned/expired、低证据和非 terminal entry 都 fail-closed。adapter 新增 `consolidate_recovery_memory()`，只向已挂载的 semantic/procedural learner 提供 active approvals；普通/native checkpoint 恢复 archive、准入 ledger 与两个长期 learner 的 consolidation 状态，`revoke_recovery_strategy()` 使该 branch 后续 replay/consolidation 失效。P3 三 seed 真实 Gate 的 low-evidence blocking、strategy admission、两类 memory consolidation、checkpoint preservation 和 revocation 全部为真，cross-seed gate rate=`1.0`；相关回归 `23 passed`，核心 mypy=`0`、Black 全绿。必须保留的边界是：当前撤销阻止未来写入/回放，但尚未对已经写入的历史权重执行反向擦除。

**当前唯一下一步：建立可回滚的 recovery strategy consolidation Gate。** 为 procedural/semantic consolidation 保存可重建的 approved-record provenance 与版本快照；撤销策略后从未撤销记录重建长期读出并验证 checkpoint/重放不再携带已撤销 branch 的权重影响，CUDA 继续暂缓。

**已完成：P3 可回滚 recovery strategy consolidation Gate 已通过。** `RecoveryStrategyLedger` 保存 approved memory record provenance、证据计数、结果和 revoked branch 集合；撤销策略后，adapter 会从 episodic records 中排除对应 memory IDs，重建 procedural/semantic readers，而不是只切换一个布尔开关，并保存 consolidation 参数、rebuild 次数和 ledger 状态到普通/native checkpoint。三 seed 真实 Gate 验证了 low-evidence 阻断、approved-only consolidation、revocation 后 reader rebuild 以及 rebuild checkpoint continuation，cross-seed gate rate=`1.0`；相关回归 `23 passed`，核心 mypy=`0`、Black 全绿。当前边界是：重建保证撤销 record 不再参与后续读出生成，但尚未做大规模策略冲突下的精确影响分解或多版本并行 memory merge。

**当前唯一下一步：建立多策略冲突下的 recovery memory 竞争与撤销传播 Gate。** 在多个 completed strategy 同时准入时，按证据、结果一致性和资源预算进行 memory competition；撤销一个策略后验证相关 alias/sequence/semantic 影响传播到所有下游读出，不误伤仍有效策略，CUDA 继续暂缓。

**已完成：P3 多策略 recovery memory 竞争与撤销传播 Gate 已通过。** `RecoveryArchiveEntry`/`RecoveryStrategyApproval` 现在携带结果一致性与资源代价；`RecoveryStrategyLedger` 通过配置化的 evidence/consistency/resource 权重进行确定性排序，并按 `recovery_strategy_memory_budget` 选择不超预算且不重复绑定同一 episodic record 的策略。adapter 的 recovery consolidation 只读取 selected strategy records；撤销后重建会排除 revoked 与未选中的 recovery records，同时保留仍被选中的 survivor，semantic/procedural 两类长期读者和 native checkpoint 均验证了传播边界。新增 ledger 单测 `2 passed`，P3 风险执行 Gate 在 seeds `11/23/37` 全部通过，cross-seed gate rate=`1.0`；新增指标 `strategy_competition_selected`、`strategy_competition_checkpoint_preserved`、`strategy_revocation_preserves_survivor` 全部为真。为避免验证脚本对每个场景重复拟合，按 seed 缓存只读 baseline 并对每个场景 deep-copy，生产运行时逻辑未改变；核心 mypy=`0`，报告/manifest 已更新为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`，CUDA 继续暂缓。该 Gate 证明策略级竞争、预算裁剪和撤销后的幸存者保护，不宣称 sequence/concept/alias 等所有下游读者已经具备同等 provenance 反向擦除能力。

**当前唯一下一步：建立跨 reader 的 recovery provenance 与精确影响分解 Gate。** 将 selected strategy 的 provenance 继续贯穿 procedural sequence、concept/semantic alias 和 replay/readout 依赖，记录每个 reader 对策略的实际贡献；撤销单个策略时只重建受影响依赖，保留未受影响的 reader 状态，并验证多版本 checkpoint continuation，CUDA 继续暂缓。

**已完成：P3 跨 reader recovery provenance 与精确影响分解 Gate 已通过。** 新增 `RecoveryReaderDependency`/`RecoveryReaderDependencyGraph`，为 semantic/procedural/sequence/concept 四类 reader 记录 selected strategy rollout 与 episodic memory provenance，并持久化到普通/native checkpoint；adapter 新增可选 `ProceduralSequenceLearner` 挂载，recovery consolidation 统一向四类读者写入 selected records。撤销时按 dependency graph 只重建真正受影响的 reader，排除 revoked/未选 recovery records，保留未受影响依赖，并将 revoked/未选记录从 adapter episodic readout 隐藏。三 seed `11/23/37` 真实 Gate 的 `recovery_reader_dependencies_recorded`、`recovery_reader_checkpoint_preserved`、`recovery_reader_revocation_propagates` 全部为真，cross-seed gate rate=`1.0`；相关定向回归 `4 passed`，核心 mypy=`0`，py_compile 与 Ruff format 通过，报告/manifest 已更新。schema alias 仍是 world schema 身份而非 memory reader，因此不在策略撤销时删除；CUDA 继续暂缓。该 Gate 证明 recovery provenance 已贯穿当前真实挂载的 reader 与 adapter episodic readout，不宣称所有外部 reader/alias 已接入同一依赖图。

**当前唯一下一步：建立 recovery provenance 的 contribution attribution Gate。** 将每个 reader 对多个 selected strategy 的实际增量贡献从“输入依赖”细化为可重放的 per-strategy credit/weight delta，验证撤销任一策略只回滚对应增量，不重建无关 reader；继续覆盖普通/native checkpoint，CUDA 继续暂缓。

**已完成：P3 recovery provenance contribution attribution Gate 已通过。** 新增 `RecoveryReaderContribution`，以 deterministic leave-one-out replay 记录每个 selected strategy 对 semantic/procedural/sequence/concept reader 的 `effect_delta_l2`、归一化 `credit`、replay epochs/learning rate；每类 reader 同时保存 consolidation 前的 baseline checkpoint 及内容 digest。procedural/sequence reader 支持固定 action vocabulary 的 ablation replay，撤销时从保存的 baseline 只重放幸存策略，并重新计算 survivor attribution，不再重训普通历史记录。普通/native checkpoint 均恢复 baseline、贡献账本和 digest。三 seed `11/23/37` 的 contribution recording、checkpoint、selective revocation 全部为真，cross-seed gate rate=`1.0`；定向回归 `5 passed`，核心 mypy=`0`，py_compile、Ruff format 与 diff 检查通过，报告/manifest 已更新。该 Gate 的 credit 是 leave-one-out 边际影响，不宣称多个策略之间的交互影响已经被分解为可加和的 Shapley/线性权重；概念 reader 的 `effect_delta_l2` 是符号状态位移而非神经参数权重，CUDA 继续暂缓。

**当前唯一下一步：建立 interaction-aware recovery attribution Gate。** 针对多个 selected strategy 的非线性交互，增加可重放的 pairwise interaction residual 与顺序无关性校验，明确哪些影响可以安全相加、哪些必须保留为组合贡献；继续覆盖撤销、普通/native checkpoint，CUDA 继续暂缓。
**已完成：CI Python 门禁修复。** GitHub 失败运行 `33037813507`、`33037154061`、`33036706021` 的共同失败点均为 Black，而非测试、Ruff、启动冒烟或 Windows 任务；远端日志明确指出 `scripts/make_social_preview.py` 未格式化。本轮按 CI 固定版本 Black `26.5.1` 的 API 对全仓 463 个 Python 文件复核并修复，同时清除 Ruff 暴露的导入排序、`cache` 规则和嵌套条件问题；不降低 CI 规则、不触碰 CUDA。Ubuntu 等价门禁已通过：Ruff 两道检查、Black 0 个未格式化、mypy `44` 个源文件无错误、版本一致性通过；native `221 passed, 1 skipped`、Seed `72 passed`、全量 `465 passed, 5 skipped`，覆盖率 `40.83%`。本地 Black CLI 在 Windows 会挂起，因此采用同版本格式化 API 完成确定性校验；这属于本机工具异常，不改变仓库 CI。提交后唯一下一步仍是 interaction-aware recovery attribution Gate，CUDA 继续暂缓。

**已复核：CI 修复已在远端全绿。** `f8d54cc` 将 checkpoint digest 改为只依赖 PyTorch 的 byte view，消除 CI 未安装 NumPy 时 Python 3.10 的失败；GitHub Actions run `33067181142` 的 Python 3.10/3.12、Windows、启动冒烟、前端、Docker 全部通过，未放宽任何门禁。

**已复核：interaction-aware recovery attribution 提交已通过完整远端 CI。** 提交 `313d4cf` 的 GitHub Actions run `33069906564` 共 7 个 job 全部成功：启动冒烟（legacy/no-legacy）、Python 3.10、Python 3.12、Windows、前端构建和 Docker 构建均实际执行并通过；没有因上游失败而静默跳过下游门禁。

**已完成：P3 interaction-aware recovery attribution Gate 已通过。** 新增 `RecoveryReaderInteraction` 与 checkpoint 格式，为 semantic/procedural/sequence/concept reader 对每一对 selected strategy 记录同一 baseline 下的单体效果、pair effect、可加和基线、带符号交互 delta、非负 residual，以及 A→B/B→A 的 order delta 和 order-invariant 结果；交互账本随 reader dependency 进入普通/native checkpoint，撤销时仅保留幸存策略仍然成立的 pair。adapter 使用真实 reader checkpoint 重放，不引入 Transformer 或 CUDA 依赖；报告/manifest 新增 pairwise interaction 与 order-invariance controls。三 seed 风险执行 Gate 的 interaction recording、checkpoint continuation、撤销裁剪全部为真；定向回归 `6 passed`，完整 `tests/taiji_native` 为 `222 passed, 1 skipped`，核心 mypy `0`、Ruff/Black 全绿。这里的 residual 是 reader 状态的确定性 L2-like 组合效应审计，不冒充 Shapley，也不把交互测量自动当作新的权重；CUDA 继续暂缓。

**已完成：P3 interaction-aware recovery selection Gate 已通过。** `RecoveryStrategyLedger` 的 canonical selection 现在读取 checkpoint 化的 reader interaction audit：在 residual/order delta 均位于配置容差内且已完成 pair audit 时，策略保持独立竞争；明显非加和、顺序敏感或缺少 pair evidence 的关系 fail-closed 地并入 connected atomic selection unit，组合按成员最小 competition score/evidence、成员 resource cost 之和参与同一 memory budget，整组只能一起准入或一起被预算拒绝。选择结果、audit、容差与 reader audit-complete 标记均进入普通/native checkpoint；撤销后重新选择和 reader rebuild 使用同一 canonical policy，不会把已撤销策略或未审计关系静默当作可加和独立项。新增 atomic pair、未知 pair fail-closed、checkpoint/revocation selection assertions；P3 evaluator 三 seed cross-seed gate rate=`1.0`，selection/checkpoint/revoke 三项新增指标均为 `true`；定向回归 `8 passed`，native 回归（排除两个已确认的 Windows pytest 临时目录权限错误）`221 passed, 1 skipped`，Ruff、Black、mypy=`0`、compileall 全部通过。提交 `6a8327e` 的 GitHub Actions run `33079659762` 已完成，7 个 job（Python 3.10/3.12、Windows、双 startup smoke、frontend、Docker）全部成功。该 Gate 只证明当前 recovery memory 的交互约束与可回滚选择边界，不宣称已完成三层以上交互的精确联合 replay 或 Shapley 分解；CUDA 继续暂缓。

**已完成：P3 高阶 interaction-group replay Gate 已通过。** 对 pairwise audit graph 中的每个 connected 三策略以上组件，adapter 在同一 baseline 上真实重放完整 group、每个 singleton，并以 canonical/reverse 两种顺序复演；ledger 同时保存 group effect、additive effect、signed pairwise interaction sum、pairwise-predicted effect、高阶 delta/residual 和 order-invariance。高阶 residual、顺序敏感或缺少完整 pair evidence 的 group 会作为 atomic selection unit，整组共享 competition score/resource cost 并在 budget 下原子准入；普通/native checkpoint 恢复 group audit，撤销时整组删除并保留幸存者贡献 attribution。三 seed `11/23/37` 的 higher-order group replay、checkpoint preservation、atomic revocation 全部为真，cross-seed gate rate=`1.0`；定向回归 `10 passed`，native 回归（排除两个已确认的 Windows pytest 临时目录权限错误）`223 passed, 1 skipped`，核心 mypy=`0`、Ruff/Black、compileall 全部通过。该 Gate 证明三阶以上 group 的非线性审计和选择边界，不宣称已完成可组合 group 的增量 replay 或 Shapley 分解；CUDA 继续暂缓。

**已复核：高阶 interaction-group replay 已通过远端 CI。** 提交 `ed1aae1` 的 GitHub Actions run `33089028142` 共 7 个 job 全部成功：Python 3.10/3.12、Windows、双 startup smoke、frontend、Docker 均实际执行并通过；Node.js 20 弃用与既有 frontend lint 提示仍为非阻断注释，不影响本次 Taiji 门禁结论。

**已完成：CI 下游门禁的 `needs: test` 挟持已结构性解除（详见 13.7）。** 上面记载的"`build-frontend`/`docker-build` 在 `test` 连红 7 次期间一直是 skipped、从未运行"是**遮蔽机制本身**，当时只作为纪律（"核对 job 集合是否都真的执行了"）记账，没动依赖图；本轮删除两处 `needs: test`，`yaml.safe_load` 复核 5 个 job 全部 `needs = None`，步骤数 26/10/5/7/8 与改动前一致，证明未误伤任何步骤，该失效模式此后不可能再发生而非"要记得检查"。同轮否证了上一轮自己提出的建议——CI 并不缺别名门禁：`build-frontend` 的 `npx vitest run` 已收集 `hljsAliases.test.js`（本机实测 20 files / 181 passed），其断言与 `check:aliases` 逐字节等价，按收敛原则不新增重复步骤。`concurrency`/`timeout-minutes` 两项欠账刻意留到独立一轮（见 14.14），当前峰值 7 个并发 job 低于公开仓库 20 上限，不构成阻塞。提交 `9dab2e5`。

**当前唯一下一步：建立可组合 interaction-group 的增量 replay Gate。** 在两个已审计 group 合并、拆分或新增策略时，只重放受影响的 group 与 pairwise 边，验证高阶 residual、顺序不变性、预算原子性、checkpoint continuation 和局部撤销与全量重放一致；未受影响 group 必须保持 digest/attribution 不变，CUDA 继续暂缓。

**已完成：客户端白屏真因已根治，并补上了让它逃过门禁的那层盲区（详见 13.8）。** 上面 13.3.2 记的"白屏已修"是推理结论、修的是另一层缺陷；用户二次上报同一现象后改用真实观测（`QTWEBENGINE_REMOTE_DEBUGGING=9222` + 裸 CDP），实测真因是 `FileUploadQueue.vue` 把 emoji 字符串喂给 `<component :is>`，Blink 校验标签名时抛 `InvalidCharacterError` 摧毁整棵 `router-view` 子树，故点进知识库后所有路由都白屏。修法为 prop 改 `[Object, Function]` + lucide 默认组件 + `asComponent()` 归一化。181 个用例全绿却放过它是因为 jsdom 不校验标签名，已把 Blink 同级校验补进 `setupFiles`（`blinkDom.js`），回退修复可当场变红，套件 181 → 185。同轮另修三项客户端反馈（去「项目文件」文字、托盘通知改用 `self.tray.icon()`、删除与顶栏重复的图标保存按钮），并根治了两条基础设施缺陷：子进程在主进程被强杀后独活占用 8000/8765，改用内核级 Windows Job Object（`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` + 三处 `Popen` 后 `adopt_child()`）而非再加一层 Python `atexit`，**打包模式实测强杀 `Seed.exe` 后 `SeedBackend.exe` 同步消亡、两个端口全部释放**；`scripts/release.py` 的 NSIS 判定自相矛盾（非致命跳过却硬性要求安装包）改为事实回传，`--check-only` 全绿，13.3.6 那条"必须加 `--skip-nsis`"的记忆式绕过随之作废。另清除 `desktop/__init__.py` 的双重导入陷阱（曾使 Job 句柄出现两份副本），并把 `.codex/` 补进 `.gitignore`（含两份活跃 worktree 副本，会污染仓库级统计）。两条持久纪律已登记为 14.15 / 14.16。

**遗留欠账（前端路由级冒烟门禁，尚未落地）。** 本轮白屏能连着两轮逃过 185 个用例，是因为"整棵 `router-view` 被销毁"这一失败态没有任何自动化断言——`blinkDom.js` 只堵住了已知的 `createElement` 这一种触发方式，换个渠道（异步组件解析失败、`defineAsyncComponent` 无 `errorComponent`、子组件 setup 抛异常）同样会白屏而门禁全绿。待做：把本轮那套裸 CDP 脚本从一次性探针固化为受版本管理的门禁——headless 起前端 preview，逐个路由断言"容器内容长度 > 0 且 `window.onerror` / `console.error` 零命中"，任一路由为空即 `exit 1`；先在本机跑通并证明能变红（回退 `FileUploadQueue.vue` 应立即失败），再接入 `build-frontend` job 与 `scripts/release.py` 的必经路径。不新增第二套 E2E 框架，复用现有 vitest/preview 与已验证的 CDP 通道。此项与下面 Taiji 内核线并行排队，当前唯一下一步以本节末尾为准。

**已完成：P3 可组合 interaction-group 增量 replay Gate 已通过。** recovery consolidation 现在优先从 reader dependency 保存的稳定 baseline 重建，新增策略不会再次叠加已训练记录；pairwise audit 保存 replay action-kind fingerprint，group audit 保存 singleton effect、replay digest 和 attribution digest。增量路径只复用 baseline、成员、动作集合、参数和内部 pair audit 全部一致的 pair/group；新增策略、group 合并、group 拆分、局部撤销或审计变化只重放受影响边/组件，未受影响 group 的 digest 与 attribution 保持原值。三 seed 风险执行 Gate 的增量 replay 统计均为 pairwise `8 replay / 4 reuse`、group `4 replay / 0 reuse`，重复 consolidation 不发生 double replay；组合回归同时证明未受影响 group 稳定、变化 group 与全量重放相等，以及 merge/split replay 数量正确。相关回归 `10 passed`，native 回归（排除两个已确认的 Windows pytest 临时目录锁权限错误）`223 passed, 1 skipped`，核心 mypy `0`、Ruff/Black 全绿，CUDA 继续暂缓。该 Gate 证明 group 组合变化下的局部重放与 provenance 稳定性，不宣称 group 内成员 credit 已完成守恒分解。**

**当前唯一下一步：建立 interaction-group 的可验证 credit decomposition Gate。** 在不把高阶 residual 粗略平均给成员的前提下，为 group 建立基于可重放子集的成员增量 credit、交互 credit 和 residual 归属，验证 credit 守恒、顺序敏感时 fail-closed、局部撤销只移除相关归属，以及普通/native checkpoint continuation 与全量重算一致；CUDA 继续暂缓。

**已完成：P3 interaction-group credit decomposition Gate 已通过。** `RecoveryReaderInteractionGroup` 现在持久化成员单体子集增量、按稳定 pair 顺序排列的带符号交互 credit、独立归属的高阶 residual 和守恒误差；group effect 必须满足 `member increments + pair interaction credits + explicit residual` 的确定性守恒，不再把高阶 residual 平均摊给成员。完整归因随普通/native checkpoint 保存，增量 group 复用要求 credit decomposition 完整，旧格式或缺失子集证据自动重放；顺序敏感、守恒不安全或未完成归因的 group 在 selection 中 fail-closed 为原子单元。三 seed 真实 Gate 的 credit decomposition、非平均 residual、普通/native checkpoint continuation、局部撤销和变化 group 与全量重放一致性全部通过，cross-seed gate rate=`1.0`；相关回归 `12 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明当前 interaction-group 的可审计 credit 守恒和安全边界，不宣称已实现 Shapley 或任意规模 group 的指数级全子集分解；CUDA 继续暂缓。

**当前唯一下一步：建立 interaction-group credit 的跨 reader 一致性与漂移 Gate。** 对 semantic/procedural/sequence/concept 四类 reader 比较同一策略组的 credit 结构、reader 状态漂移和 checkpoint 版本变化；当某一 reader 的 credit 结构变化时只失效该 reader 的 group attribution，保留其他 reader 与未受影响 group，CUDA 继续暂缓。

**已完成：P3 interaction-group credit 跨 reader 一致性与漂移 Gate 已通过。** 新增 `RecoveryReaderCreditConsistency` 与 `RecoveryReaderDependencyGraph.credit_consistency`，对同一策略组在 semantic/procedural/sequence/concept 四类 reader 的成员/交互/residual 分解建立 reader-independent 结构 digest，并将不同 reader 的状态尺度归一为 signed credit profile；原始 reader group replay digest 与 baseline checkpoint digest 作为状态漂移和 checkpoint 版本证据保存，不要求不同 reader 的原始数值相等。adapter 在每次 audit 后比较 coverage、结构、归一化 credit L1 漂移和 checkpoint/state digest 完整性；若单个 reader 的 group 结构或 credit profile 变化，仅将该 reader 的 group attribution fail-closed，未变化 reader 与其他 group 保持原对象和 replay 边界。普通/native checkpoint 均恢复一致性记录；真实三 seed 风险执行 Gate 的跨 reader consistency、checkpoint preservation、semantic-only drift isolation 全部为真，定向回归 `13 passed`，native 全量 `228 passed, 1 skipped`，另有 2 个既有 Windows pytest 临时目录锁权限 setup error 未进入测试体；Ruff、Black、mypy 全绿，CUDA 继续暂缓。该 Gate 证明的是当前四类 reader 的 group attribution 可比性、版本证据与局部失效边界，不宣称 reader 输出已经共享同一语义空间，也不宣称 credit 已达到 Shapley 或跨模态因果真值。

**当前唯一下一步：建立跨 reader credit consistency 的多组、跨 checkpoint 增量回滚 Gate。** 在多个 interaction group 同时存在时，为每个 group 保存独立 audit revision；验证 group 新增/合并/拆分、单 reader 漂移、checkpoint continuation 与局部回滚只更新受影响 audit，未受影响 group 的 structure/profile/state digest 保持稳定，CUDA 继续暂缓。

**已完成：P3 跨 reader credit consistency 多组、跨 checkpoint 增量回滚 Gate 已通过。** `RecoveryReaderCreditConsistency` 为每个 group 增加独立正整数 `audit_revision`；revision 只由该 group 的 reader 集合、结构 digest、normalized signed credit profile、base checkpoint digest 或 replay state digest 变化推进，不跟随 dependency graph 的全局 revision。真实 evaluator 覆盖两组基线、group 新增、A+B 合并、拆分恢复、semantic reader 单独漂移、native payload continuation 和局部回滚：未受影响 group revision/profile/state digest 保持稳定，受影响 group 依次从 `1 -> 2 -> 3`，单 reader 仍只被局部 fail-closed。定向回归 `15 passed`，三 seed cross-seed gate rate=`1.0`；native 回归 `229 passed, 1 skipped`，另有两个既有 Windows pytest 临时目录锁权限 setup error 未进入测试体；Ruff、Black、核心 mypy=`0`，CUDA 继续暂缓。

**当前唯一下一步：建立 cross-reader audit revision 的有限历史、回滚目标校验与容量淘汰 Gate。** 在不让旧 audit 重新成为可执行 attribution 的前提下，为 group 保存可验证的前序 revision 摘要；checkpoint 恢复后只允许回滚到存在且结构兼容的目标 revision，超过容量的历史不可复活，CUDA 继续暂缓。

**已完成：P3 cross-reader audit revision 有限历史、回滚目标校验与容量淘汰 Gate 已通过。** 新增 digest-only 的 `RecoveryReaderCreditAuditRevision`，每个 group 只保存 rollout 集合、reader/structure/profile/base/state digest 与完整性摘要，不保存 raw credit profile 或 `reader_attribution_safe`，因此历史记录不能重新变成可执行 attribution。`RecoveryReaderDependencyGraph` 现在按 group 保存有限 revision history，checkpoint/native payload 可往返恢复；回滚校验要求目标 revision 仍在容量窗口内，且 reader 集合、结构 digest、profile digest 和 base checkpoint digest 与当前 audit 兼容，缺失、篡改或结构漂移目标均 fail-closed。容量由 `recovery_strategy_cross_reader_credit_revision_history_limit` 配置，adapter 初始化、reset、restore 和每次 audit 持久化均使用同一上限；撤销策略时同步裁剪历史，合并/拆分留下的旧摘要仍不可执行。P3 evaluator 新增历史完整性、checkpoint continuation、目标 revision 校验和容量淘汰三项 Gate，三 seed 均为 `true`、cross-seed gate rate=`1.0`；定向回归 `15 passed`，native 回归 `229 passed, 1 skipped`，另有两个既有 Windows pytest 临时目录锁权限 setup error 未进入测试体；Ruff、Black、Taiji Mypy=`0`。本 Gate 只证明“可验证的有限审计历史与回滚 allowlist”，不宣称摘要本身能恢复神经状态或执行真实 rollback；CUDA 继续暂缓。

**已完成：P6 native-readable 产品语言表层 Gate。** 根因已确认：Seed 聊天虽然调用 Taiji 的 byte prediction，但直接把 raw bytes 当作答案，且默认 `structured-stub` 只能做无损结构序列化，不能形成可读语言。现在 `SeedRuntime.chat` 将 prediction 和本地会话上下文封装为 Taiji-owned `ExpressionPlan`，经过无外部依赖的 `native-readable` 语言表层；有效的 `surface_text/answer/native_prediction` 候选会被保留，不可读字节会转成诚实的可读状态文本。`structured-stub` 保留为显式 debug codec；Seed 配置升级为 v2，旧的未版本化 structured 默认会迁移到 native，显式 v2 structured 仍保持可用；native organ 已纳入 registry、checkpoint restore 和 `/api` final event 的 `language_backend` 可观测性。产品聊天默认不把用户历史静默转发给外部 decoder；Qwen/LoRA 仍是显式 provider 的表达器候选，不因此宣称 Taiji 已具备开放域语言智能。定向语言/provider 回归 `17 passed`，产品聊天冒烟 `4 passed`。

**已完成：P6 Taiji-owned `ExpressionPlan` 到真实语言表达的训练/holdout admission Gate。** 新增 `LanguageRealizationGate`，以
`LanguageTrainingCorpus` 为唯一监督边界，逐例验证 train/holdout 不串集、UTF-8/可读文本、必需语义词完整覆盖、无结构化泄漏、无
fallback，并要求 rollback reference 与保存后 checkpoint loader 的输出逐例一致。Qwen LoRA trainer 在写出 adapter/tokenizer 后重新加载
保存目录，再把 checkpoint continuation 纳入最终 Gate；真实本机 CPU 复核为 4 epochs/16 steps、270336 个外部参数，train/holdout
质量均为 `1.0`、rollback 与 continuation 均为 `true`。Seed 新增 `chat_enabled` 显式开关，且强制 `guarded` 模式；只有训练 realization
Gate 与 safety Gate 都通过才允许外部 decoder 进入产品聊天，旧报告或缺失证据 fail-closed，默认仍为本地 `native-readable`。相关定向回归
`20 passed`，核心 mypy=`0`、Ruff/Black 全绿；CUDA 继续暂缓。该 Gate 证明可审计的表达准入，不宣称开放域语言智能。

**已完成：P6 语言 provider artifact 内容寻址与首轮 chat canary Gate。** `LanguageProviderArtifact` 现在为文件或目录内容生成路径无关的
SHA-256 digest，并以 role 列表和稳定 manifest digest 绑定 base model、LoRA、训练语料、训练报告和安全报告；目录摘要只依赖相对 POSIX
路径、文件大小和字节，不依赖绝对路径、mtime 或遍历顺序。guarded product chat 在加载前严格要求五类内容摘要、manifest digest、固定
canary 合同和未过期 `expires_at`，逐项重新计算并拒绝缺失、替换、manifest 漂移和过期 artifact；旧 artifact/checkpoint 仍可读取，但没有
内容寻址证据时不能进入 product chat。provider 挂载后，`LanguageProviderCanaryGate` 对实际 language organ 执行两条固定语义表达，要求 UTF-8
可读、`数据库/正常` 与 `接口/恢复` 完整覆盖、无结构化泄漏且不触发 validated fallback；失败统一回退到 `native-readable`，并区分
`chat_artifact_missing`、`chat_artifact_drift`、`chat_artifact_expired`、`chat_canary_failed`。训练侧 artifact loader smoke 也已输出内容摘要和
canary 结果。相关定向回归 `23 passed`，Ruff、Black、核心 Mypy=`0`；本机 Seed/Taiji 全量测试的测试体未见本次回归，但仍受既有 Windows
worktree/pytest 临时目录 ACL setup/cleanup error 影响，未计为全量 Gate 通过；CUDA 继续暂缓。该 Gate 证明的是 provider 资产完整性和首次真实
表达准入，不宣称开放域语言智能或消除外部 decoder。

**已完成：P6 provider artifact 多版本 registry 与原子轮换 Gate。** `LanguageProviderArtifactRegistry` 只保存经过内容寻址的 immutable
manifest，按 artifact ID 去重，显式维护版本 allowlist、active/previous 指针和 monotonic revision，并通过 native checkpoint 保存与恢复；
未 allowlist 的版本、manifest 冲突、未知回退目标和 registry 指针漂移均 fail-closed。Seed 新增 `rotate_language_provider` 与
`SeedRuntime.rotate_language_provider`：候选 provider 在脱离线上 language organ 的 staging adapter 中加载，依次通过内容摘要、训练/安全报告和
首轮 chat canary 后，才以一个 `commit_language_provider_state` 操作同时发布 organ、backend registry、artifact 和新 registry snapshot；候选
失败时旧 provider、旧 runtime 和 active/previous 关系保持不变。定向语言/provider 回归 `25 passed`，Ruff、Black、核心 Mypy=`0`；提交后 CI
已复核全绿（Python 3.10/3.12、Windows、前端、Docker、启动冒烟），CUDA 继续暂缓。

**已完成：P6 provider runtime health watchdog 与自动回退 Gate。** 在 active artifact 已通过首轮 canary 的发布后运行时，增加请求级健康探针
`LanguageProviderHealthProbe`、连续失败阈值、有限冷却窗口和 previous-version 自动回退：`observe_language_provider` 把每次真实发射折叠进
`LanguageProviderHealthState`（可读表层、结构化泄漏、validated fallback 三项判据，异常/不可解码记失败），达阈值后 `auto_rollback_language_provider`
随 `now` 判定 nominal/冷却/回退；有 distinct previous 时回退到 previous 版本（`provider_health_rollback_previous`），隔离劣化版本并移出 allowlist，
冷却期内保持现状，无 previous 则落到 `native-readable` 且 `chat_enabled=False`（`provider_health_rollback_native`）。健康状态随 native
checkpoint 保存与恢复，重启后续接；探针与回退共用原子轮换路径（收敛而非叠加），且任何误报只能保持现状或回到 `native-readable`，不静默加载未
allowlist 的 artifact。Taiji/adapter/config/seed 四层实现，seed 层活体验证 9 项全绿，api 层 `SeedRuntime.chat` 请求级回退实测通过；定向回归
`18+96 passed`，Ruff 全绿、核心 Mypy=`0`；CUDA 继续暂缓。该 Gate 证明发布后劣化可被请求级吊销，不宣称开放域语言智能。

**2026-08-28 watchdog 收尾漏洞回扫（暂停推进期间）：** 按用户指示不向后推进、只回头核对前序 watchdog 推进里的真实漏洞并修复。共修 6 处——
A. **状态键集不一致**：`SeedRuntime._native_status()` 之前手写 9 键 dict，与 `LanguageProviderStatus.to_dict()` 的 14 键形状不符，原生/受管模式 status API 键集漂移；改为复用状态对象自身投影，单一事实来源。
B. **回退后配置残留**：回退到 native/structured 后 `_provider_config` 仍指向被降级版本；新增 `_sync_provider_config()` 在回退提交后清空/重锚配置，杜绝残留降级配置被后续观察误用。
C. **名义探针健康位不实时**：未触发回退的名义探针只更新 adapter 健康记录、不叠加进 status API；新增 `_overlay_health()` 让 status 实时反映健康负载而不翻转角色语义。
D. **重启复活隔离版本**：watchdog 隔离的版本会随重启被 config 重建而跳过 `require_allowed` 复活，违反复苏→劣化→回退死循环承诺；新增 `_registry_revokes_candidate` + `activate_language_provider` 拒活隔离版本，镜像/保留被隔离的持久 registry，覆盖 `provider_health_quarantined`。
E. **核心 mypy 漏报**：`taiji/language_organ.py` 对混合值类型 `metrics` dict 做 `>= 1` 排序（对 int|float|bool|str 联合不合法）遗漏 2 错；改为用局部标量比较。
F. **`_chat_organ` 类型**：`api/seed_runtime.py` 三处把 `LanguageOrgan` 赋给推断为 `NativeReadableTextLanguageOrgan` 的变量；显式标注为 `LanguageOrgan` 协议。
全量门禁复验：核心 mypy（`seed taiji`，`--follow-imports=silent`）`0`、Ruff 全绿、Black 无改动、全量 `498 passed / 5 skipped`。定位用诊断脚本见 `scripts/archive/diagnostics/diag_provider_health_audit.py`。

**当前唯一下一步（已收敛回主线）：** 回到 §16.1 的 Workbench Closure W0–W7。Taiji 已构造世界/计划/`ActionIntent`/`ToolCall`/`Outcome` 认知与
效应器合同，Seed 产品却仍缺一个 Taiji-native 的执行平面把这些合同接到 IDE、文件、终端、诊断和 MCP；watchdog、CUDA/fused kernel、新视觉打磨
等末端优化一律冻结，直到真实工作台纵切片（W0 起步：选定最小真实工具并打通认知→效应器→结果闭环）通过。
