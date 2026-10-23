# 内存守卫与 fast-ledger 恢复更新

本文已转为历史快照；当前操作、状态及替代入口见[训练系统精简](15-training-system-simplification.md)和[03当前执行](../../../../active/roadmap/03_CURRENT_EXECUTION.md)。

更新：2026-10-23 11:54 Asia/Shanghai。本文是执行快照；当前progress、receipt、账本摘要和资源采样优先。

## 最新训练状态

full_turn_1m与fresh_same_capacity已停止，warm_start为当前唯一运行臂：

| 臂 | episodes | target-byte updates | 累计 CPU 秒 | 结果 |
|---|---:|---:|---:|---|
| full_turn_1m | 17,019 | 8,192,724 | 1,436,869.033 | resumed后首步前因单臂/总额预算投影不足而持久停止 |
| warm_start | 24,975 | 12,028,674 | 1,437,226.873 | fast-ledger恢复后当前唯一运行臂，11:54仍有episode_completed事件 |
| fresh_same_capacity | 25,104 | 12,096,251 | 1,383,166.236 | 10:13因可用内存低于8GiB持久停止 |

warm/fresh先前的内存停止事件均已结算reservation。full的fast-ledger启动写入resumed和资源通过事件，随后因当前单臂/总额投影不足而停止，reservation及progress已由runner停止事件结算。旧partial、父谱系、最新receipt与checkpoint都保留。warm没有final result或assistant-SFT终点。

截至warm启动后约11:50，累计charges加warm的252秒reservation约4,257,514.141秒；4,300,000秒上限余42,485.859秒。启动时全量投影136,656.835秒，full / warm / fresh分别约80,820.936 / 30,003.162 / 25,832.736秒。用户指示先运行一个臂后，fast-ledger按selected-arm scope放行warm：启动时30,003秒投影小于当时113,750秒单臂余量及103,047秒总余量。启动尾账已计入当前账本；warm继续受原单臂/总额硬上限约束。

warm在11:48资源预检通过：可用内存10.058GiB，CPU双采样32.9%/25.0%；fast-ledger manifest绑定序号60,865账本锚点，warm在11:54已推进至24,975 episode并继续运行。global三臂投影为false；本次只按warm单臂30,003秒投影放行，其他两臂保持deferred。

## 账本争用处置

上次三条恢复命令并发时，共享累计账本锁导致full-turn启动安全退出。warm/fresh随后分别恢复并推进约300集与300集后，因内存守卫停止。根据这两项实测，恢复命令改为单臂启动；只有实际资源稳定时才恢复其他臂的并发。

直接重启旧target-update入口会再次完整读取约60,000条历史事件，并在共享账本锁内校验；这次行为与运行中的训练事务发生争用。已新增[fast-ledger恢复入口](../../../../../scripts/training/run_taiji_r2_fast_ledger_resume.py)和[尾账锚点校验器](../../../../../scripts/training/taiji_r2_resume_ledger_anchor.py)：从已完整核验的序号59,476、SHA `542c61dd…739279` 重放后续事件；每个新增reservation与结算、父谱系、哈希链、单臂/总额上限都要通过。训练阶段继续使用checksummed摘要快速读；终点manifest仍执行全历史账本审计。

只读现场重放已验证到序号60,857，SHA `f1fe175c…73144f0`，charges、无reservation状态与summary一致。现有v1.3训练源、批准预算、模型训练输入和原target-update入口均未修改。实际launch校验器已验证旧三臂共6份manifest及26项来源/审计文件；warm新manifest已固定适配器源码和锚点，最终完整source/ledger复核仍留到训练终点。

## 检查结果与边界

37项聚焦测试通过，覆盖锚点后的reservation/结算/停机事件重放、重复reservation与哈希/谱系错误拒绝、fast-resume manifest边界，以及原endpoint gate。ruff通过。fast-ledger已实际恢复warm_start并观察到episode_completed事件；单元测试、锚点校验和已启动都不等同最终训练完成。

终点result/readiness、唯一completion事件和fresh-process恢复仍待齐；extension评测适配与最终独立源码审查仍待完成。W5b、人工样例、dev/reference、selector和晋级继续冻结。

## 当前唯一下一步

让warm_start单臂继续到30,096预训练episodes与525 SFT episodes；先核实最终checkpoint、receipt与fresh-process restore，再决定是否扩大训练臂范围。
