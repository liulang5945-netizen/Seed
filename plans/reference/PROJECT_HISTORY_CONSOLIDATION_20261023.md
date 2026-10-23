# 全项目历史整理与删除结案

日期：2026-10-23。用户明确要求整理整个项目历史并“该删就删”；本轮对已跟踪文件按目录/类型盘点，历史层做依赖和Git恢复审计，运行残留做精确白名单核验。Git基线为`d12a8679a`；没有重写提交历史。

## 1. 实际处置

- 已删除18个确认可再生目标：2,254文件、361,584,658字节（361.58 MB），包括14组10月21日已结束pytest输出、旧mypy缓存、Harness旧coverage/pytest缓存及coverage数据库。
- 删除6份整篇重复计划快照，转由Git保存全文；当前入口引用已迁移到本记录。原始结论、旧审批与失败保存在现行过渡矩阵和既有原始报告中。
- 08保留上游同步手册、752条登记标题和650原完整处方；删除重复操作流水正文。报告引用仪器只扫描当前工作树摘要/原件，不再声称覆盖已转Git的历史全文。
- reference/README和archive/README改为主题/恢复索引；docs地图指向当前10/17及本记录，不再把10月14日旧计划称“本轮”。

实际路径、前后字节和精确恢复身份见[机器清单](../manifests/project_history_cleanup_20261023.json)。这里的空间数是删除文件的逻辑字节，不把此前已删除的冷归档原件重复记入本轮，也不据并发磁盘变化声称物理释放量。

## 2. 当前只需要的入口

[03运行队首](../active/roadmap/03_CURRENT_EXECUTION.md)；[10详细持续计划](../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)；[09机制设计](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)；[07验收](../active/roadmap/07_MINI_MODEL_DELIVERY.md)；[17计划衔接](delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md)。历史按reference主题索引、08标题与Git定位，不重新发布旧队首。

## 3. 明确保留的内容

- 8份与训练/评价冻结身份相关的Markdown按17的SHA继续保护；运行中源码、CPU账本、checkpoint/语料与未提交改动不清理。
- `.seed_test_tmp`含`r2_boot_revision_bootstrap.py`与`r2-source-overlay-20261022-wm6f4nbq`，保留整个根，只清已核实的旧pytest子目录。
- 旧321KB worktree.patch仍可能承载独有未合入内容，本轮保留；历史源码/测试/报告没有按日期批量删除。审计的1,030份历史文件没有完全相同Git blob的重复组，不虚构大规模源码去重成果。
- `outputs/cold_archive_20261014/`四份ZIP合计42,047,015字节，现查整体SHA、CRC及20个逐文件SHA均有效；原件早已删除，本轮保留恢复包。当前用户数据、凭据、会话、`.git`、`.workbuddy`及第三方依赖/发布会话格式保持原范围。
- 当前技术债05含尚未结项的责任与证据，保留；上游Harness源码/文档/发布格式不视为本项目过程垃圾。

## 4. Git历史恢复

6份被删快照在`d12a8679a`的原归档路径都可恢复，其原活动文档还可从`c174b3066`恢复。机器清单记录每份Git blob、SHA及路径；17保留源到归档映射。恢复示例只读取一个文件，不还原整棵旧仓库：

```python
from pathlib import Path
import subprocess
commit = "d12a8679a"
path = "plans/archive/history/CONTINUOUS_TRANSITION_PLAN_INDEX_BEFORE_20261023.md"
raw = subprocess.check_output(["git", "show", f"{commit}:{path}"])
# Inspect raw or write only to a deliberately selected recovery location.
```

Git blob按LF保存。17中PLAN_INDEX与09的原SHA对应LF；03、board及两份02的原SHA对应当时CRLF工作树，比较它们的原SHA时须显式转换行尾。六份被删归档的删除前工作树字节均为CRLF，Git版本为LF；机器清单分别记录Git blob SHA256与删除前工作树SHA，按`worktree_line_endings`转换后可精确匹配，不能直接用Git LF字节比工作树CRLF的SHA。08和两个旧索引的完整原文同样在`d12a8679a:<原路径>`读取；已从工作树删去的文字不当作现在还在位。

## 5. 后续留存与验证

新工作包写结论、来源、失败和当前必要接口，不再将每次操作对话复制到多个索引。重复流水经蒸馏/迁链/Git恢复核验后可删除；当前合同、唯一原始证据和仍在消费的接口保留。清理工具默认不整删用户数据、源码overlay或依赖根，也不为dry-run遍历这些保护树。

最终验证：33项受影响契约通过（清理保护、凭据卫生、计划身份、marker审计、报告引用与650逐字处方）；29份活动/交接文档链接0缺失；18个运行目标与6份快照删除状态核实，6份Git恢复按各自行尾全部精确匹配；752个marker顺序不变、650原完整行一致；8份冻结SHA一致。历史审阅与运行保护审阅通过，复审指出的CRLF恢复说明已补全；这不是整仓CI或模型能力通过。训练会话62394在清理前已核验存活，最终复核返回预算上限保护退出；17:48:48停止事件66,699已结算、无reservation，耐久partial为27,584 episodes / 13,304,596 updates。当前事实已同步03；本轮删除不涉及训练源码、语料、账本或checkpoint，不能用沙箱看不到进程推断无活进程。
