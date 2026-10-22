# 当前增量工程计划

## 结论
训练期间仅补齐07独立评测器；不更改运行中训练代码及冻结计划。

## 产物与依赖
架构合同 → 独立评测器和契约测试 → 独立审查 → 定向验证 → 计划同步与提交。训练三终点完整且恢复通过后才允许真实生成。

## 遗留风险
训练耗时可能跨会话；本交付完成不等于训练或N主线收官。

## 给下游
保持78次新增生成和39个共同prompt分母，沿用06已保存两端点，失败留证且禁止静默重试。资源预算必须在执行前明确。

## R2 recovery v2 W5 修订计划（2026-10-21）

本增量补充保留以上旧R2-07计划作为历史；旧“78次新增/复用06两端输出”不适用于客户端更新后从zero恢复的v2三臂。后续实现与验收以 `artifacts/04-v2-evaluation-architecture.md` 为准，不改训练合同、runner或checkpoint。

|工作包|负责人|输入|输出|完成标准|依赖|
|---|---|---|---|---|---|
|W5a v2独立评测实现|ai-integration|v2架构补充、固定06生成/restore接口|独立评测runner与synthetic契约测试|严格校验parent+3个v2 assistant_sft终点；restore前write-once计账；所有restore成功前不选prompt；测试证明156分母与fail-closed、不重试|架构补充；可与W4三臂训练并行|
|W5b 真实一次性执行|执行岗|三臂final result/readiness/checkpoint及恢复回执、W5a审查通过|同面generation/result/attempt及CPU账本|四端点×39 family共156次新生成；CPU≤14,400秒且wall≤14,400秒，任一先到即停；0 reference/exact-match评分，can_promote=false|W4所有终点和独立恢复通过；W5a通过|
|W6 人工审阅与结案|评审/文档岗|完整v2评测结果及生成|固定12-family四端点审阅、计划/报告同步与提交|可读性、贴题、历史连贯、重复逐项记录；仅据证据决定下一轮方向，不自动晋级|W5b完整且结果来源核验通过|

关键依赖为W4→W5b→W6；W5a的实现、合成测试、代码审查可在W4训练期间并行。真实selector访问39-family面必须等待全部四个端点（parent baseline、full_turn_1m、warm_start、fresh_same_capacity）fresh-process恢复成功且执行账本完整。预算/资源门之后、首个恢复worker之前先write-once落盘attempt及CPU账本；逐个restore前后fsync记账。任何restore失败/强杀/账本不明均消费attempt并拒绝重跑，prompt仍零读取。

验证边界：使用synthetic prompts和虚拟result/receipt验证，不读取dev/reference、不运行真实模型或训练；source/result/readiness/checkpoint SHA漂移、固定分母不符、路径越界、资源或双预算门失败均应fail closed。评测为局部描述性研究，不构成语言能力晋级或N1-166准入。

## Phase 7 首轮审查回流（2026-10-22）

首轮独立代码审查结论为打回（P0=0、P1=6、P2=3）；W5a尚未通过，不得进入真实评测。P1-06平台合同已由架构岗补充裁决并提交：仅支持Windows本地固定NTFS，以同目录exclusive-temp写入、文件刷新后经`MoveFileExW(MOVEFILE_WRITE_THROUGH)`无覆盖发布attempt和initial ledger；UNC/网络、ReFS、可移动或未验证文件系统必须fail closed。AI集成岗据此修复六项P1（worker只能由有效attempt/restore事件授权、子进程模块启动、restore与generation预算deadline、prompt selector前复核来源、支持平台的durable one-shot文件发布）并处理三项P2（完整成本报告口径、reserve算术、selector阶段意图记录），增加对应的合成失败注入。返工结束后重跑定向测试并由code-reviewer复审。三臂W4训练继续，不访问dev/reference，不启动真实restore、selector或generation。

## Phase 2 W5a返工设计澄清（2026-10-22）

本节将R2V2-08..10与04-v2架构新增合同落实为Phase 6可直接执行的返工顺序；旧W5a职责与Phase 7打回结论保留，不视为返工已完成。

1. **durable one-shot入口：**仅支持Windows本地固定卷NTFS，实际报告目录必须预先存在；在同一目录创建exclusive temp，写后`FlushFileBuffers`，再用`MoveFileExW(MOVEFILE_WRITE_THROUGH)`无覆盖发布attempt与initial ledger。文件系统资格探测、attempt/ledger任一发布或内容核验失败时，在启动worker、restore或selector前拒绝。UNC/网络、ReFS、可移动介质及未验证文件系统fail closed。
2. **单父进程与长驻worker：**父进程拥有attempt、CPU/wall账本和全部授权；一个受监督长驻worker顺序完成四端点fresh-process restore并驻留模型。worker不持有selector、dev路径或预取/重试权限。每个restore前先fsync唯一授权事件，返回后由父进程核验并durable记录完整restore回执；四端点回执全部持久化后，才允许父进程调用一次prompt selector。
3. **逐条授权与有限期限：**selector完成后，父进程对每条生成先fsync `query_started`，再通过IPC只授权一个(endpoint,prompt SHA)工作项；worker完成后等待下一条授权。restore和generation各有有限deadline与有界采样间隔；deadline或预算门触发时停止派发，父进程在有限软/硬终止宽限内终止整棵worker进程树并等待退出，记录CPU/wall尾账与失败事件。终止/退出不可证明时不得继续或运行selector。
4. **原双预算全额记账：**父进程与整棵worker树的启动、模型/语料加载、restore、生成、IPC、超时/取消、进程树终止、ledger及result最终化成本，全部计入原14,400 process-CPU秒和14,400 wall秒双门；不新增预算。每项授权前扣除剩余query、取消、终止与finalization reserve。已触门或无法证明余量时不再授权；保留已写attempt、query和失败/unknown-tail证据，消费attempt且不可重试/续跑。
5. **回P7门禁：**P0必须为0、六项P1和三项P2全部关闭；相应synthetic故障测试均通过后才退回Phase 7独立复审。至少覆盖：worker无父attempt/伪造endpoint授权拒绝；模块方式启动与安全导入；restore/generation hang、CPU忙循环、worker后代及整树强制终止；selector前来源漂移保持selector调用为0；NTFS资格/flush/rename/第二marker失败时restore启动为0；最终CPU/wall包含finalization；reserve算术正确；selector意图事件先行fsync。只做synthetic/本地NTFS验证，不读取dev/reference、不运行真实模型、restore、selector或generation。

W5b仍以Phase 7复审通过及三臂W4固定终点、result/readiness/checkpoint与fresh-process恢复齐备为前置；任一未满足均不启动真实评测。该返工设计不改变Phase 8主训练指针、冻结训练合同、156次分母或`can_promote=false`边界。

## Phase 6 IPC与worker启动硬化补充（2026-10-22）

Phase 7第二轮复审指出的P1-07/P2-04已落实到实现与合成测试：IPC JSON帧硬限制为2 KiB，generation只传已持久prompt face的路径、index与SHA；父进程通过独立写线程发送并在同一CPU/wall deadline循环中监督，超时先终止Job整树并在有限宽限内确认写线程退出。worker以`CREATE_SUSPENDED`创建，只有Job Object分配和成员验证通过后才恢复执行；启动失败不进入授权或selector。旧复审结论仍保留，Phase 7未因此自动通过。

定向IPC/Job测试4/4通过，完整评估器合同40/40通过，Ruff、py_compile、模块`--help`及diff检查通过。完整测试首次发现一条测试仍断言旧ledger字段`worker_startup_cpu_seconds_before_job_assignment`；根因是实现按新启动顺序改为`worker_startup_cpu_seconds_charged_total`，只同步断言字段后精确复测1/1通过，再次全量40/40通过。未访问真实dev/reference、未restore模型、未调用selector或`--execute`。

第二轮审查仍判不放行（P0=0、P1=1、P2=1），原定回流预算2/2已用尽。按用户持续推进至收官的明确授权，新增一次独立Phase 7复审作为本次新补丁的后续门禁；结果未出前，W5a仍标记未解决，W5b不得开始。R2-06/07的最后durable进度、attempt=`training`且无final已同步至活动路线图；因外部进程状态无法从沙箱确认，不重启或覆盖任何训练状态。
