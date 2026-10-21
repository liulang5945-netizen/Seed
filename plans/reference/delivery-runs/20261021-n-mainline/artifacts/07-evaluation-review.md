# Phase 7 独立代码审查：R2 recovery v2 paired evaluator

日期：2026-10-22。审查范围为v2评测新脚本、synthetic契约测试、02/03/04-v2/06岗位产物与交付黑板。旧R2-07的78新调用/78旧输出复用口径不作为本次审查标准。未访问dev/reference正文，未启动模型、restore worker、selector或`--execute`。

## 结论

**打回，不放行Phase 7合入门。** P0=0，P1=6，P2=3。实现岗报告定向测试12/12通过、Ruff/py_compile/CLI模块`--help`通过；本审查未重复执行测试，根任务正在用项目内临时目录独立复核。结构上的fail-closed路径较完整，但restore-worker入口、子进程启动、执行时限、prompt前来源门与durable attempt语义仍有阻断问题。

## 问题清单

|级别|位置|问题与影响|修改方向|
|---|---|---|---|
|P1|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:750-787,1619-1627`|`--restore-worker --endpoint`是独立可调用的公开CLI分支。worker只跑metadata preflight并加载训练语料/模型checkpoint，没有校验父进程attempt nonce、未闭合`restore_started`事件或剩余CPU/wall预算。用户可绕过write-once attempt与14,400秒总预算，反复启动四端点fresh restore，实际CPU不入账。|worker必须证明由当前父attempt授权，且只消费唯一未闭合的endpoint restore额度；或改用不能被无账本独立触发的内部启动方式。增加测试证明无父attempt/伪造endpoint请求在读取语料或checkpoint前拒绝。|
|P1|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:35-37,804-807`|父进程以脚本文件路径启动worker，但此脚本在导入`base`前没有把仓库root放入`sys.path`。根任务验证报告直接`python <script> --help`因`sys.path`缺少repo root失败；同一启动形式用于restore worker，第一次恢复会在`from scripts.training...`处失败并消费one-shot attempt。基础训练脚本显式在包导入前插入root（`run_taiji_r2_daily_dialogue_full_turn_pretraining.py:25-27`），本脚本没有。|用`python -m scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2 ...`从repo root启动，或在任何包导入前可靠设置repo root；加安全的子进程启动/导入烟测。不要用真实checkpoint做该测试。|
|P1|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:804-821,1233-1254,1338`|restore子进程只由`process.poll()`轮询，没有wall deadline、CPU deadline或budget回调；预算只在启动前预留120秒，worker挂起/超慢时父进程持续等待，不能在14,400秒上限前终止并记账。恢复支出可能越过冻结硬门。|把剩余预算/deadline传给worker监视循环，达到单次或全局CPU/wall上限时杀进程、记录保守尾账和失败attempt；用可控的sleep/hang fake worker测试超时和账本闭合。|
|P1|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:1374-1400,1472`|生成调用在父进程同步执行；前置守卫只预留固定60秒，`generator(...)`返回后才检查预算，调用过程中没有deadline或中止机制。若单次512-byte decode超过60秒，预算可实际越过14,400 CPU/wall硬上限。|对单次generation建立可验证的最长执行界或可取消隔离执行；否则不得把60秒经验余量宣称为硬预算保证。测试超慢调用不会越过总预算而继续执行。|
|P1|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:1346-1357,1483-1485`|所有restore后父进程载入四个trainer/corpus并只核验checkpoint SHA和模型digest，随后直接调用prompt selector；source/plan/result/readiness哈希只在最初metadata preflight及各worker内核验。主进程直到156次生成后才执行`after_preflight()`。来源在最后worker预检后、selector前发生漂移时，保留prompt已被访问而后续才报错，违反prompt读取前source gate。|在`prompt_selector()`前重新核验全部绑定source、result/readiness与checkpoint，并与已写attempt的输入绑定逐字比较；漂移时selector调用数必须仍为0。增加“最后restore后改source”故障注入。|
|P1|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:520-543`|write-once attempt与初始ledger通过`open("x")`新建后只对文件句柄`os.fsync()`，没有父目录条目durability sync。04-v2架构明确将目录fsync失败列为拒绝条件；当前仓库没有可复用的跨平台directory-sync helper，根任务在Windows上实测`os.open(dir, O_RDONLY)+os.fsync(dirfd)`得到`PermissionError 13`。因此新建attempt/ledger目录项的崩溃持久性未被实现或证明；若attempt目录项丢失，存在重放one-shot评测的风险。追加到已有JSONL文件的fsync单独不受此目录项问题影响。|提供经过验证的Windows目录项durability实现，或由architecture-reviewer明确支持的平台/降级合同并在不支持平台fail closed；attempt与初始ledger任一durable marker无法确认时不得启动restore。加入平台能力测试。|
|P2|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:1521-1555,1566-1579`|`cpu_seconds_total`/`wall_seconds_total`在result payload写入前取值；final ledger append和result序列化/写盘耗时未进入字段名所称的total。虽保留了300秒finalization余量，结果仍低报本次完整process/wall成本。|将字段明确命名为finalization-start计数，或增加可审计的最终上界/后置completion receipt，避免total被误当成完整执行成本。|
|P2|`plans/reference/delivery-runs/20261021-n-mainline/artifacts/06-implementation-ai.md:45`；实现常量位于`evaluate_taiji_r2_daily_dialogue_recovery_v2.py:69-76,697-707`|岗位产物称整轮初始reserve为10,140秒；按实际常量应为4×120 + 156×60 + 60 + 300 = **10,200秒**。代码`_whole_run_reserve()`计算值正确，岗位说明算术不符。|订正文档计算并保留“余量非实测成本”的说明。|
|P2|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:1357-1360`|prompt selector调用前没有单独耐久的`prompt_selection_started`事件；如果selector已经读取39条但进程在face JSON落盘前被强杀，attempt存在但没有face SHA或可辨识的selector阶段记录。无法审计保留面是否已被访问（虽同attempt仍会拒绝重跑）。|selector调用前fsync阶段意图，返回后先尽可能耐久记录face身份/SHA；强杀尾账明确标为prompt-selection状态unknown。|

## 合并门槛核对

- P0：0。
- P1：6，均未修复，不能合入或进入真实v2评测。
- P2：3，建议在复审前一并处理；至少修正文档reserve算术。
- Phase 6报告的12项synthetic测试、定向Ruff、py_compile及CLI模块帮助为已报告证据；本审查未运行它们，也未运行全仓CI。
- 没有访问dev/reference、模型payload、真实selector或真实评测输出；旧R2-07评测和训练恢复review只作为历史材料。

## 遗留风险

目前新测试覆盖了合成元数据、端点/156分母、失败不重试等路径，但没有覆盖公开worker绕账本、实际子进程导入、worker hang、generation overshoot、restore后source漂移至selector之间、Windows attempt目录耐久等本次P1。因此“12/12通过”不代表上述真实执行门已闭合。远端CI仍待发布后验证。

## 给下游的注意事项

按P1逐项修复后，只对新增synthetic模块做定向复检；任何涉及Windows目录durability的合同取舍先由architecture-reviewer裁决。三臂训练仍可独立运行，真实39面prompt访问必须继续冻结，直至Phase 7通过、三臂final和fresh-process restore齐全且一次性attempt/CPU账本通过审查。

## 回流判定

- 结论：打回。
- 回流目标：`ai-integration`（P1-06若Windows目录durability无法实现，须先退回`architecture-reviewer`明确平台合同）。
- 返工原因：P1-01至P1-06使one-shot attempt/预算、restore执行、来源门或跨平台durability无法按冻结合同证明。
- 复检范围：六项P1全部修复；新增对应negative/failure injection tests；定向12项及新增覆盖全通过；不运行真实模型/selector/--execute。
