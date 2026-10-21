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
