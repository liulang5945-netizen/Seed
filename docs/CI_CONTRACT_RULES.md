# CI 契约规则（CI CONTRACT RULES）

> 本文档回答一个问题：**怎样让一次 push 不把 CI 推红**。
> 每条规则后面括注催生它的真实事故——它们都发生过，且都花了 2.5 小时级的
> CI 周期与本机返工才定位。CI 绿的真实含义是「**可测面全绿**」；
> deselect 是登记债（每条带重启条件），不是豁免。

## 0. 推送前最小义务

1. **改动面相关测试本机必须跑过**。跑不了（训练占用 CPU）就缩小改动面。
2. **新测试必须在 LF worktree（runner 同构判据）验证过**，或明确写出它
   依赖本机产物的哪一条。判据做法：
   `git worktree add --detach <tmp> HEAD` → 删
   `E:/Seed/.git/worktrees/<name>/index` → `git -c core.autocrlf=false reset --hard`
   （本地全局 autocrlf=true 会把 worktree 也落成 CRLF，必须强制 LF）。
3. `ruff check .` + `black --check .` + B/SIM 棘轮（基线 0）过。
4. R2 训练运行期间，本机一切 pytest/python 消耗的宿主 CPU 秒**计入训练账本**
   （psutil host 口径）——先查
   `reports/*aggregate_cpu_ledger.latest.json` 的预留余量，一次跑全，不反复小跑。

## 1. 新测试的平台纪律

1. **禁止导入期绑定 Windows-only API**。runner 模块在 import 时构造
   `_WindowsDurabilityApi` 之类平台绑定的话，消费它的测试必须
   `sys.platform != "win32": pytest.skip(..., allow_module_level=True)`。
   （事故：run 40066946185 Ubuntu 腿收集即炸 ×2——v4 快速续跑契约、
   worker namespace 集成。）
2. **引用仓库文件必须用仓库相对路径**。封存记录里的绝对路径键
   （`E:\Seed\...`）在任何非本机上都 fail-closed。
   （事故：n02_c4、plan142_face 等 12 条 deselect 的根因。）
3. **路径分隔符不得参与摘要与计数**（`\` vs `/` 在 Linux/Windows 不同）。
   （事故：n5_17_j、n5_20、n4_05。）
4. **Windows 行为契约测试**（断言 CR 产生等）必须 `skipif(sys.platform != "win32")`。
   （事故：n6_10/12。）
5. **平台敏感的阈值/浮点比较必须带容差**：admission 类判定用
   `math.isclose`（先例：`taiji/interaction_group_online.py` 的
   `ADMISSION_FLOAT_TOLERANCE`，owner 2026-10-25 裁定方案乙）。
   （事故：族 A 8 条——relation 两测同提交三次两红一绿、structural-gain
   门族两轮绿后翻红。）

## 2. 新消费面/新文档的登记义务

1. **新文件消费 stop_reason**（写 payload、断言 reason、聚合 reason）：
   必须当天登记进
   `scripts/training/audit_taiji_b0_n2_stop_reason_disposition.py` 的
   `EXPECTED_CONSUMERS`（已处置）或 `PENDING_FRONTIER_CONSUMERS`（R2 前沿），
   并同步 `test_b0_n2_stop_reason_disposition_contract.py` 的钉数与冻结报告。
   （事故：run 40166946185 b0_n2 红——worker 测试未登记。）
2. **新建 `plans/reference/PLAN-*.md`**：判据段必须有含「判据/出口/验收」的
   二级标题（过措辞门），或显式加进
   `test_prereg_wording_corpus_ratchet_contract.py` 的 `FROZEN_INVISIBLE`
   并在注释里留理由。
   （事故：run 40024125106 预注册棘轮红；run 40166946185 又红——3 份新
   R2 收口文档未处理。）
3. **文档引用 `reports/*.json`**：引用已入库读数时确保该文件在 git 里；
   「计划产出」引用按 n5_38 census 的 planned_output 口径书写。

## 3. 哈希封印纪律

1. 分辨口径：`sha256(path) == 常量` 是**当前值口径**（改文件必须重冻结）；
   `seal_ast.verify_path_pin` 是**历史口径**（旧值留在 git 历史即有效，勿多改）。
   **`.py` 的 AST 哈希钉没有历史回退**——改 .py 后必须把新旧 `ast_hash` 对
   并入重冻结映射。（事故：N1-144 manifest 两枚失败、N1-121 链。）
2. manifest 重冻结只动**活跃钉**（字面量前方 450 字节内出现目标文件
   basename 的路径键钉）；扁平溯源字段（`exclusion_sources.*_sha256`）
   与被字节钉住的结案文档不动。（事故：首轮粗替换引发 104 枚钉上钉级联。）
3. 封存记录引用产物时确认产物已入库；未入库产物（checkpoints、在飞
   jsonl）的哈希钉在任何 runner 上都无法核验。
4. **遗留证据的绝对路径字段**（本规则设立前写入的 runner_path、
   boot_audit_path、boot_accounting_sources 等）：校验方接
   `seal_ast.resolve_recorded_path` 跨机归位＋`seal_ast.verify_path_pin`
   历史取证＋`seal_ast.rendering_tolerant_match` 渲染容忍（先例：R2
   source_revision 链，2026-10-25 owner 裁定）；**不要**重写已入库证据
   文件本身（哈希链会级联）。新证据一律仓库相对路径（§1.2）。
5. **训练治理面文件**（run_taiji_r2_*、taiji_r2_dialogue_continuation_
   extension.py 等）默认冻结；R2 结案后的owner 授权例外修改须在代码
   注释里写明裁定出处与语义转变（部署面活校验→来源出处核验）。

## 4. deselect 清单纪律

1. 清单在 `ci.yml` **两处 env（test 与 test-windows）必须逐条一致**；
   改动后必须跑 `scripts/training/verify_ci_deselect_list.py` 全命中。
2. **摘除**一条须「本机 + CI 双绿」证据；**新增**一条须写明族属与重启条件。
3. 判别「本机绿/CI 红」用 LF worktree，不用本机（本机 E:\Seed 的存在会
   骗过路径映射与产物判定）。
4. 训练治理面（`run_taiji_r2_*`、`taiji_r2_resume_ledger_anchor.py`、
   `taiji_r2_dialogue_continuation_extension.py` 及其契约测试）不改、
   不代提交其证据（reports/ 在飞读数归收口流程）。

## 5. 并行会话纪律

1. 提交只圈自己的文件（`git add <具体路径>`），他人未提交的在飞改动
   （taiji-harness、roadmap 等）不捎带。
   **共享 index 危险（2026-10-25 实际发生）**：并行会话可能已把它的文件
   `git add` 进共享暂存区——`git commit` 会把整个 index 一起带走（事故：
   e3d0ef588 把我的 b0 收口与他人的 plan173 7144 行卷进同一提交，提交
   信息还归属了对方）。防御：commit 前必须 `git diff --cached --stat`
   复核暂存区，或用 `git commit -- <自己的路径>` 只提交指定路径。
2. 推送前 `git log origin/main..HEAD` 过一眼——并行会话可能已推进 main；
   自己的修复不覆盖别人的漂移，别人的漂移按第 2 节登记义务收口或留言交回。
3. 并行会话停机前：worktree/临时目录清干净（每次残留 ≈2–3G）。

## 6. 违规的真实成本

一次 CI 红的修复周期 = 2.5 小时级 runner 等待 + 定位 + 返工 + 再一轮验证。
2026-10-23～25 的修绿轮共消耗 **12 个 CI run、约 30 小时墙钟**，其中约
三分之一的红来自「新测试/新文档未按第 1、2 节登记义务执行」。这些义务
单条成本分钟级——按本文件执行就是把小时级返工换成分钟级预检。
