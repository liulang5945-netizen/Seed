# 内存守卫与 fast-ledger 恢复更新

更新：2026-10-23 11:07 Asia/Shanghai。本文是执行快照；当前progress、receipt、账本摘要和资源采样优先。

## 最新训练状态

三臂均已停止，最后耐久进度与停止原因如下：

| 臂 | episodes | target-byte updates | 累计 CPU 秒 | 结果 |
|---|---:|---:|---:|---|
| full_turn_1m | 17,019 | 8,192,724 | 1,381,259.298 | 仍停在客户端中断点；fast-ledger恢复尚未启动 |
| warm_start | 24,915 | 11,986,178 | 1,386,250.248 | 10:12因可用内存低于8GiB持久停止 |
| fresh_same_capacity | 25,104 | 12,096,251 | 1,383,166.236 | 10:13因可用内存低于8GiB持久停止 |

两条停止事件均已结算 reservation，账本当前无活动 reservation。旧partial、父谱系、最新receipt与checkpoint都保留。没有final result或assistant-SFT终点。

累计charges为4,150,675.782 CPU秒；4,300,000秒上限下余149,324.218秒。target-update投影为136,656.835秒，full / warm / fresh分别约80,820.936 / 30,003.162 / 25,832.736秒，各臂预算及总额门通过。剩余余量约12,667秒，继续按实时账本重算。

10:12和10:13的实际运行守卫都因可用内存低于8GiB拒绝了下一work unit。10:48单次采样为8.188GiB；11:07三次只读采样为7.663/7.859/7.839GiB，目前不能证明满足内存门。CPU最近单次采样45.9%，仍须按训练入口要求双采样。训练不在运行。

## 账本争用处置

上次三条恢复命令并发时，共享累计账本锁导致full-turn启动安全退出。warm/fresh随后分别恢复并推进约300集与300集后，因内存守卫停止。根据这两项实测，恢复命令改为单臂启动；只有实际资源稳定时才恢复其他臂的并发。

直接重启旧target-update入口会再次完整读取约60,000条历史事件，并在共享账本锁内校验；这次行为与运行中的训练事务发生争用。已新增[fast-ledger恢复入口](../../../../../scripts/training/run_taiji_r2_fast_ledger_resume.py)和[尾账锚点校验器](../../../../../scripts/training/taiji_r2_resume_ledger_anchor.py)：从已完整核验的序号59,476、SHA `542c61dd…739279` 重放后续事件；每个新增reservation与结算、父谱系、哈希链、单臂/总额上限都要通过。训练阶段继续使用checksummed摘要快速读；终点manifest仍执行全历史账本审计。

只读现场重放已验证到序号60,857，SHA `f1fe175c…73144f0`，charges、无reservation状态与summary一致。现有v1.3训练源、批准预算、模型训练输入和原target-update入口均未修改。实际launch校验器也验证了旧三臂共6份manifest及26项来源/审计文件。

## 检查结果与边界

35项聚焦测试通过，覆盖锚点后的reservation/结算/停机事件重放、重复reservation与哈希/谱系错误拒绝、fast-resume manifest边界，以及原endpoint gate。ruff通过。fast-ledger入口尚未用于训练；不会把单测、锚点校验或预算预测写成训练恢复完成。

终点result/readiness、唯一completion事件和fresh-process恢复仍待齐；extension评测适配与最终独立源码审查仍待完成。W5b、人工样例、dev/reference、selector和晋级继续冻结。

## 当前唯一下一步

等可用内存稳定高于8GiB并通过CPU双采样后，使用fast-ledger入口单臂恢复`full_turn_1m`；观察到新的耐久resume/训练事件且资源守卫持续通过后，再恢复warm与fresh。
