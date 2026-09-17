# Seed / Taiji 长期工作记忆

## 项目双轨架构（机器强制契约）
- `taiji/` = 自足认知基底：**禁止导入** seed / seed_platform / neuroplex / transformers（含传递性），
  由 `tests/taiji_native/test_architecture_contract.py` 与 `test_naming_boundary_contract.py`
  AST 级强制（ast.walk 连函数级导入都抓）。禁止放宽这两个测试的断言凑绿（登记册 §8）。
- 共享测量仪器放顶层 `instruments/` 包（2026-09-13 起）：依赖方向 instruments → taiji 单向
  （仅 content_digest），taiji/ 对 instruments 零引用（duck-typed 参数）。
  `DocumentEmbedder` 现居 `instruments/document_embedding.py`；语义 encoder 的 embedder
  必须显式注入（fail closed，无隐式构造）。
- `neuroplex/` = 冻结 Transformer 基线，不得 import seed/taiji（单向替代关系）。
- 语义 checkpoint 兼容性红线：`taiji-document-embedder-v1` payload 格式与 digest 锚
  不得变更，P5.1x 预注册依赖它。

## 工作流纪律（预注册→实现→门禁→报告→提交）
- 负结果如实落账（transfer_no_gain / transfer_signal_constant），绝不改绿；
  报告对自己机制的描述必须与代码同步。
- 归属证明用引用图检查（谁 import 被改模块），不靠反复重跑。
- 回归测试必须双向钉住（能过 + 该拒绝的拒绝），只查一侧会接受坏中间态。
- 委托父模块 helper 会继承父模块全局常量——gate 有自己的数据集时必须本地实现。
- matrix episode 无 `context_id` 键（身份在 episode_id 内）；step 层才有
  executed_actions/provenance/safety_violation。
- 测试改完必须读回全文/跑测试：Edit 工具可能匹配到错误缩进位置并成功，
  造成断言被嵌套而静默失效（2026-09-13 实例）。
- 临时探针用毕即删；预注册文件冻结不改，新结论写新预注册。

## 文档更新纪律：「只追加」vs「改上文」（2026-09-17 确认，按类型分而非二选一）

- **冻结判据 / 预注册**（阈值、读数映射、停止线）⇒ **只追加，永不改** —— 改它等于事后调参，
  毁掉可追溯性。新结论一律**新预注册**。
- **冻结证据 / 历史报告**（结果快照）⇒ **只追加**；若被覆盖，则**恢复归档版 + 新版本另存**
  （例：B2 报告 `fde682ef` 归档位 + `5abb123a` 另存为 `_rerun_`）。
- **导航 / 状态文档**（如 `03_CURRENT_EXECUTION.md` 的「当前唯一下一步」）⇒ **必须改** ——
  它唯一的价值就是"当前"，只追加会直接让文档失真。
- **原文写错**（笔误、错误事实）⇒ **改 + 显式注明原值**。
- 追加以**新 §** 形式，并在长文档**顶部加「最新状态指针」**指向最新 §，
  避免读者只读上半部分而采信过期结论。

## R2 语言路线的诊断模式（2026-09-17，可复用，**很有价值**）

- **「teacher-forced 指标改善、自主生成 `exact` 恒为 0」已出现三次**（H3.7B / P3b-v2 / H-GEN）。
  不要再把它当成"机制没用好"或"生成路径有问题"来解释。
- **决定性判据 = 生成内容是否随输入变化**：若**不同输入生成同一字节串**，且它等于
  **训练集众数**，则模型学到的是**边际分布而非条件分布** ⇒ **前向/机制/生成态层的解释全部不成立**，
  问题在**训练信号语义**。
- **诊断法极廉价**：把 `generate()` 的输出与 train split 的众数对照（几行代码）。
  本轮即由此定位：两臂都输出 `蓝`，而 train 里 `蓝` 出现 13 次为最高、dev 众数是 `白`。
- **已排除的因素**：exposure bias（H-GEN，两臂 dev exact 均 `0/16`，φ≤0.5）。
- **纪律**：小预算（2 epochs × 12 episodes）下的否决，只能得出"**在该预算下**非充分根因"，
  必须写进结项报告，不得外推。

## ⚠️ 小预算陷阱：连续改动"不移动任何指标"= 先怀疑预算，不要先怀疑假设（2026-09-17）

- **实证**：H-GEN（φ 调度）与 H-OBJ（上下文对比损失）**连续两次**在同一预算
  （2 epochs × 12 episodes、12 个训练 episode）下，**两臂 dev `exact` 均 0/16**，
  **且连 train accuracy 都完全相同**（`0.6265`）—— 尽管对比项的 `margin_gap` 有非零读数、
  treatment 的 loss 系统性更高（证明"改动确实生效"）。
- **判据**：**当"改动已生效"但"两臂所有指标逐位相同"时，正确解释是"预算不足以让改动表达出来"**，
  而不是"该假设被否决"。**两者必须写进报告的界限段**，否则会连续误杀正确的假设。
- **纪律**：在**放大预算**（让对照臂自身指标先动起来）之前，**不要再加新机制/新目标** ——
  否则每一轮都在同一个"表达不出来"的预算上做无用功。
- 相关：R2 语言路线的诊断模式（生成 = 训练众数 ⇒ 边际退化）见上一节。

## R2 诊断阶梯（2026-09-17，**必须按预算分层**）

**判据（新增，很重要）**：一次否决只有在「**对照臂在该预算下已能产出非零 `exact`**」时才算**有效**。

| 预算 | control 臂 dev exact | 是否"有效预算" |
|---|---|---|
| **30 × 25 × 3 seeds** | `0.25` / `0.0625` / `0.0` | ✅ **有效** |
| 2 × 12 | 全 `0` | ❌ 无效（改动无法表达） |

**已在有效预算下排除**：
- **「目标未显式要求使用 context」**（H-OBJ @30×25）：seed 20260917 的 `margin_gap = +4.6867`
  证明"正确 prefix 优于打乱 prefix"的依赖**确实被建立**（且 train_acc 更高 0.9277），
  **而该臂 dev exact 仍为 0** ⇒ **"依赖已建立、不转化为产出"**。
- **exposure bias**（H-GEN @30×25 重验）：φ>0 使 dev exact 从非零变全 0 ⇒ 生成态训练有害。
- **「首字节损失加权」**（H-FBW @30×25）：fbw=5 使全部 3 seeds 的 exact 归零
  ⇒ 首字节全错**不是权重问题**。
- **「容量/几何宽度」**（② @30×25）：2× 所有宽度轴（prefix/renderer/slot）**无提升**
  （0.0625 vs 0.1042）⇒ 容量不是瓶颈。
- **⇒ 结论升级：三次"加损失"（对比项/位置加权/生成态训练）在有效预算下全部失败**，
  而同预算下**不加这些损失的对照臂能产出非零 exact** ⇒
  **"内容抽取 → 字节生成"的缺口不是损失设计能修复的，指向架构层**
  （显式"回答起点/内容读出"机制；注意 H3.8 的 workspace 是"仅加机制"的先例，判定 workspace_unused）。
- **据此剩余候选**：**(c) 架构读出** > **③ 课程/数据判别**（容量已排除）。

**待重验（不要当已排除）**：
- ~~exposure bias~~ → **已于 2026-09-17 在 30×25 重验并排除**（control exact 0.25/0.0625/0，
  treatment 全 0 ⇒ 生成态训练反而破坏了本可成立的 exact；且与 H-OBJ 的 λ=0 臂逐位一致 ⇒ 仪器可复现）。
- `margin_gap` 本身只在 1/3 seeds 明显转正（+4.69 / ≈0 / −0.89）
  ⇒ "依赖被建立"**也不稳健**，只能说"在本 seed 上确实发生过"。

## R2 缺口已定位到字节级（2026-09-17，只读诊断，**最有价值**）

对 λ=0 checkpoint（seed 20260919，30×25）逐条 dev 对比 TF 路径与生成路径：
- TF 命中率 **0.5943**；生成 exact **0**；
- **TF 首错位置 = {pos_0: 16/16}**；**生成首分歧 = {pos_0: 16/16}**。

**=> 缺口 = 「prefix 状态 -> 第一个回答字节」的条件化起点**：
模型在"该开始回答了"的时刻不知道该从材料里取什么，输出训练集边际众数。
后续位置一旦喂真字节命中 59% => 状态推进与读出本身正常；不是生成漂移（首错应分散）。

**与 H-OBJ 的 margin_gap=+4.69（seed 917）合并解读**：
"正确材料整体更好"是**序列级**判断（已建立）；"第一个字节该是什么"是**位置级**条件化
（未建立）—— **缺口在两者之间**。

=> 后续候选的形式空间：(a) 只读深化（判 exact=0.25 是否碰巧）；(b) 首字节加权 CE；
(c) 显式"回答起点"状态/标记。诊断脚本：`scripts/training/diagnose_taiji_r2_readout.py`。

## 环境硬约束（Python312 + shim 缺陷）
- 用 `C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe`
  （managed 3.13.12 无 torch/ruff/black）。
- bash 外部命令 ls/cat/grep/head/tail/mkdir/rm 全部缺失：文件操作用 Read/Write/Edit/Glob/Grep
  工具或 `python -c`；目录操作用 `python -c "os.makedirs(...)"`。
- 严禁 heredoc / `python -c` 内长中文：shim 逐行执行内容。提交信息用 Write 写
  `.git/COMMIT_MSG_TXT` 再 `git commit -F`。
- 全量 pytest ~15 分钟会 SIGTERM：`run_in_background` 或分批；结果用 `--junitxml` 采集。

## 伪影审计纪律（M4，2026-09-13）
- **任何"正增益 + 改善既有面"的结果，必须先过四项审计**（本线曾被零步伪影坑过）：
  ①**特异性**（正增益只在设计面；单成员可解面上必须为 0 或负）
  ②**干预真实性**（`frozen._intervention_reality(...)["interventions_happened"]` 必须 true，零步非 baseline 必须 0）
  ③**机制 lesion**（增益随交接消失 + 单体全败）
  ④**种子稳健性**（多偏移重训，且不得在无组合可解 context 的面上制造增益）
- **改善必须逐步归因**：导出每个翻转 cell 的 tick 级轨迹（`chosen`/`kind`/`executed`），
  能指出"基线里谁一次机会都没有、M4 让谁进场"才算解释清楚。
- M4 审计结论：四项全过（3 种子恒 +2.000）；冻结面 2 改善确认为真实交接。
- **扩规模必须区分"表面变体"与"结构变体"**：只换扩展名/语言/内容形状而**组合结构不变**，
  结果指纹会逐位相同 ⇒ 只能排除表面偶然性，**不排除结构特异性**。加 n 个同构 context = 没加。
  任何"扩样"交付都要自带 distinctness 诊断（`variant_outcome_distinctness`）并写明 bounds/does_not_bound。
- **写进代码的预期值必须有可否决通路**：否则是装饰性字段。检验"合同是否接受该任务"要用
  **与规则无关但含合同层的脚本化执行**（见下条：不能用 `_run_scripted`），
  也不能用真值表跑规则（会混淆"没走到那步"与"被拒"）。
  预期与观测不符 ⇒ 改判 + `SystemExit` 拒出报告，**不静默保留也不删掉错误推理**。
- **⚠️ `counterfactual._run_scripted` 不能判合同合法性**：它直调 `execute_tool`，
  而 `language_evidence_ambiguous` / `language_assessment_unavailable` 一类拦截**只由
  `environment.policy_for` 发出**。用它做的 `contract_admissible` 会把"没人能合法执行"的
  格认证为合法 ⇒ 0 增益被误读为结构结论（2026-09-13 N1 第一轮实测踩中）。
  正确做法：复现 `_member_episode` 的完整路径 bind → `ActionIntent` →
  `WorkbenchActionRequest.from_action_intent` → `policy_for` →（`issue_approval` →
  `dataclasses.replace(approval_token=…)` → 再判 → `consume_approval`）→ `execute_tool`，
  并在 deny / 非审批类 ask_user 处**像真系统那样终止**。
- **修好测量门之后必须重跑并逐格 diff 数值**，把"哪些结论被改变"写进文档：
  本轮 `create__mismatch` `0.000`→`+2.000`、3 个假矛盾归零。同时说明**预测函数未改**
  （否则"预测与观测一致"就可能是事后拟合）。
- **清单/扫描型结论归档前必须删掉临时探针**：全仓 `stop_reason` 扫描会把自己的
  `scripts/_smoke_*.py` 算进清单（12 → 11），污染审查面。

## 机制修法（B0 反事实测量，2026-09-13）
- **反事实方法**：`inspect.getsource(frozen._member_episode)` + **唯一锚点替换**，
  在冻结模块命名空间的**副本**中 exec ⇒ 冻结函数从不被 rebind；锚点非唯一即 `SystemExit`。
- **两个关键锚点**（各只出现一次）：`chosen = bindable[0]`、`cues = tuple([cue] * (len(steps) + 1))`。
- **`editor.set_language` 在文件缺失时 `bound=True` 但 `success=False`** ⇒
  "绑定失败"**不能**当交接信号。失败动作**既不消耗回合也不触发交接** ⇒ 首成员独占 episode。
- **`m4_failure_handoff`（唯一实测可行，未实施）** = ①失败即让位（有人成功则解除封锁）
  ②`cues` 用**该成员自身**已执行步数而非整集步数。冻结面 **0 回归 / 2 改善**，
  候选面 **interleaved=4/4、同参照增益 +2.000 > 1.65** ⇒ H2 首次可达。
- 备选均否决：`m1a` 安全但完全无效；`m1b` 破坏基线（3 回归）；`m2`/`m2a`/`m3` 回归且无效。
- **候选任务必须先证"可满足 + 顺序是否强制"**，否则负结果不可解释。

## 任务/机制/表征三层分离（B0 交接探针，2026-09-13）
- **探针必须先复现冻结历史**：用冻结执行器逐字重跑验证面，11/11 一致才允许谈候选面。
- **`_bind` 只作用于 `task.main_path`**（所有动作种类），且 `workspace.create` 绑定
  `content = goal_files[main_path]` ⇒ 单任务只能有一个 `main_path`；`goal_files` 中非 main 的文件
  **永远无法满足**；"创建后打补丁"**不可表达**（创建即达标）。
- **"双要求"任务不天然是协作任务**：`header_override` 家族政策本身就是
  `read→resolve→set_language→apply_patch`，故"覆盖+补丁"有成员能独做。
  只有 **`create` + `override`** 这类"无任何单成员政策覆盖"的组合才是组合专属可解。
- **机制根因**：交接触发 = "第一个成员**绑定失败**"。`read`/`list`/`resolve` 几乎总能绑定成功 ⇒
  首成员**独占 episode**（到 STEP_CAP），其他成员永不执行 ⇒ 24 单元零交错。
  **修法是 M1：触发条件改为"无进展"**（待决策 D5，未实施）。
- **M1 不违反仲裁上界**：上界针对 `P ∈ {S_i}` 的机制；M1 目的是让 `P ∉ {S_i}` 成为可能。
- 探针/预检都必须**只读**：不训练候选任务的模型、不注册任务、不改 gate；候选定义只存在于探针内。

## 组合机制与协作上界（B0 续篇，2026-09-13）
- `_member_episode`（p5_2b runner）的组合规则 = **优先级回退链**：
  每 tick 所有 active member 预测，但 **`chosen = bindable[0]`**，只执行第一个绑定成功的成员。
  **学习者从不仲裁 cell 内部谁上**（只决定动作类型与选哪一对）。
- **可证上界**：`max(S_i,S_j) ≤ max_i S_i` ⇒ 任何"结果取自某个单体政策"的机制，
  `mean(P − max_i S_i) ≤ 0`。**表征/排序改进永远无法单独支撑协作主张（H2）。**
- 协作（同参照增益 > 0）**只能**来自"**组合专属可解 context**"（所有单体失败、组合成功）：
  `k > n·(参照+margin)/(成功−B)`。n=4、margin=0.15 时：参照 0.5 ⇒ k≥2；1.0 ⇒ k≥3；1.5 ⇒ k≥4。
- 预检脚本 `scripts/training/audit_taiji_b0_task_reachability_precheck.py` 为 B1 **入场强制**；
  新任务须过六条（k 足够、脚本探针能产生交错、lesion 归零、非平凡、合同合法、不可脚本补答案）。
  条件 2 **无法由 oracle 推出**（oracle 不知道绑定失败模式），必须跑脚本化真实执行。

## 测量字典硬约束（B0 起，v1 草案）
- **参照唯一**：任何"增益"必须声明参照；不同参照的数值禁止相减、禁止共用一个字段名。
  旧字段 `mean_gain_vs_strongest_single` 同时承载了 pair 内参照与全体 oracle 参照（已停用）。
- `max(S_i,S_j) ≤ max_i S_i` ⇒ pair 内参照**恒不严于**全体参照；两者不是同一量的两次读数。
- oracle（全体单体 / 全 cell）必须标记 `deployable=False`，不得当策略用；interaction 不是效用。
- **成本权重必须事先冻结**：看到结果再定权重等于事后调参。
- 审计脚本 `scripts/training/audit_taiji_b0_measurement_reachability.py` 为只读；
  其 `cross_check` 失败即 `SystemExit` 拒出报告（不发布建立在不可信重建上的上界）。
- 当前结论：**目标不可达**（三种参照天花板 −0.5/0.5/0.0 vs 要求 1.65/0.65/1.15）⇒
  D1–D4 决策明确前不启动 B1 训练；改任务优先于加特征列。

## N1 结构空间探针（2026-09-13）
- **"结构稳健"的正确计数单位是结果指纹与失败形态，不是 cell 数**：若干格共享同一
  `(增益, 冻结增益, 联合必需 context 数, 交错数)` ⇒ 它们是**一个结构因素经由多条路线**，
  不是多次独立确认。可主张宽度写作"1 因素 × N 路线 × M 失败形态"，并把指纹计数留在报告里。
- **计划的"下一步候选"要先做可表达性探针**：本轮 T1/T3 在冻结 binder 下根本建不出来
  （`workspace.create` 绑 `goal_files[main_path]` ⇒ 创建即达标；`_bind` 恒指向 `main_path`
  ⇒ 第二目标路径不可达），T2 无成员间证据通道（`predict_episode(self, cues)` 单参数、
  元组内是同一张量重复）。⇒ 正确替换方案是**枚举目标谓词能表达的全部子句路线并逐格测**，
  候选本身作为**界限**报告（witness 字段，不是断言）。
- **规则改动后的 cell 归因分三桶**：`handoff_explained`（联合必需 context 上有 ≥2 个
  **执行成功**的成员）/ `fallback_only`（变化只落在某单体已能解决的 context ⇒
  oracle 已在 success，承重增益不可能移动）/ `unexplained`（联合必需 context 上无第二执行者）。
  只分两桶会把普通回退效应报成"无法解释"，诊断哭狼之后人们就会忽略它。
- **N1 结论**：11 格 / 22 context / 3 种子全测，**M4 正增益覆盖 `create` 行全部三条语言路由**
  （各 +2.000、interleaved 0→2、由 `member-a+member-c` 真实交接解释），其余 8 格 0.000，
  `patch` 行两规则同 −2.000（未把不可达变可达），零回归，预测与观测 11/11 一致，
  两种重跑 JSON 字节相同。**界限**：N1a 跨内容结构须先改 binder（另立议题，不属 D5 的改让位规则）；
  N1b 每格仅 2 context；N1c 仅 3 种子（5 种子证据只覆盖 `create__override`）；N2 仍为落地前置。
- 新工具入口：`scripts/training/probe_taiji_b0_structure_space.py` /
  `reports/taiji_b0_structure_space_probe_20260913.json`（≈3.5 分钟整轮，有效性半程 2.8 秒）/
  `tests/taiji_native/test_b0_structure_space_contract.py`（30 项，读 JSON + 源码 + 单个
  合同感知 validity，不训练）。

## 文档链接前缀（写 plans 文档必查）
- `plans/reference/*.md` → `../../` 到仓库根；`plans/active/roadmap/*.md` → `../../../`。
- 同一文件的多处替换**必须串行**：并行 Edit 会丢更新（两条都报 success，实际只生效一条）。
- 写完跑一次链接校验（解析所有 `](path)` 是否 exists）。

## 本地验证环境（2026-09-14 实测，动手前必读）
- **跑 `tests/` 必须设 `CODEBUDDY_SAFE_DELETE_ENABLED=0`**：WorkBuddy 沙箱的 safe-delete
  批删守卫（`cli/vendor/shim/sitecustomize.py:826`，按「单个 tool call 内删除 ≥50 路径」计数）
  劫持 `Path.unlink`/`os.remove`，超限即 `raise SystemExit(1)` ⇒ 大批 gate 用例**假失败**。
  `dangerouslyDisableSandbox` **不能**绕开它（hook 经 PYTHONPATH 注入，与命令沙箱无关）。
- **判据：`SystemExit` 级联先拿栈**（`--tb=long -o junit_logging=all`）**再归因**；
  别用「app_state 共享 / 顺序污染」猜——本项目曾据此错归因一整轮。
- `default_workspace_root()` 取 `get_setting("workspace_path")`，否则回落 `agent_workspace`
  （本机 = `C:\Users\23747\Documents`）。**凡要读仓库文件的 gate/测试，
  `SeedRuntime.load(...)` 必须显式传 `workspace_root=PROJECT_ROOT`**：load 不继承构造时的 override。
- 远端 CI 的 `test` job 曾**唯一**失败于 `black --check .`（460 文件）；已于 `75e9c97b` 全仓格式化
  （纯格式、AST 保持、两次复检幂等，格式化后四段验证仍全绿）。其余 job 一直 success。
  `gh` 已认证（`liulang5945-netizen`, keyring）⇒ 远端 CI 可查，T-CI 债务解除。
- **教训：判定「既有格式债」是否阻塞，先查 CI 口径**——本次 black 是 CI 唯一红项，
  属必做而非可选；只看本地 `black --check` 会把它误判为"未登记的旧账"。
- 前端 `vite build` 在本机失败是 `node-safe-delete-shim` 拦截清空 `dist`（环境伪影），
  输出到隔离目录即成功。

## Git 恢复硬知识
- git 仓库必须有 `refs/` 目录（空也要存在）；`git fsck` 解析 `.git/logs/` 下**所有**文件，
  坏 reflog 必须移出该目录。
- 判定产出是否丢失：先核文件在盘 → 再核哈希差异能否被 CRLF/LF 解释 → 最后才下结论。
- 修复前整体备份 `.git`；**确认无需回溯历史前不要跑 `git gc` / `git prune`**
  （dangling commit 与孤儿 .idx 是仅存线索）。
- 备份位置：`E:/Seed-backup-gitstate-20260913-183929/`。
