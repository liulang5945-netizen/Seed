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

## 6. 2026-10-26活动计划重编

本次按用户要求将03的当前动作、09的机制交汇与10的分阶段排工重新对齐，移除活动正文中已结案的队首、旧审批时态与重复操作流水；冻结合同、原始成功/失败结果和未消费面继续保留原位。没有复制整篇历史快照，也没有删除原实验、源码、报告、数据或checkpoint。

重编前全文固定在提交`e3be4e8b0ab7214c3212534567930839ebc46f36`。以下Git blob均已逐份读取并校验，SHA256对应Git保存的LF字节，CRLF数均为0；不与Windows工作树的潜在CRLF字节混比。

| 原路径（均在`plans/active/roadmap/`） | Git blob | LF SHA256 | 字节／行 |
|---|---|---|---|
| `03_CURRENT_EXECUTION.md` | `f15b751f220cdcf50a913dc9760631e795fc5763` | `85f321be4f142aaf776b3ce19a35724db9ae441df45627222480a8d2758d9fd4` | 33,449／149 |
| `09_NEXT_MAINLINE_PLAN.md` | `e1dcf3f3e50cfc35c7953e64b0a6c5ddf69bb49f` | `60291eb4511e24183ddca5dd8d3c4d84bacd54e6846f2ea13d3443a3d46b08c4` | 17,324／126 |
| `10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md` | `bba1325809a16477d48fbff91e78233c98513bef` | `eac35391dfde1fd817a182a883941301ca50946e797c737e3f57e2212539ab1a` | 38,680／197 |

单文件恢复沿用§4的只读方式，将`commit`替换为上述完整提交、`path`替换为表中完整原路径即可；命令形式为`git show e3be4e8b0ab7214c3212534567930839ebc46f36:plans/active/roadmap/03_CURRENT_EXECUTION.md`。恢复原文只用于追溯，不恢复旧额度、题面资格、审批或队首的效力。

| 原活动正文中的历史内容 | 保留位置与解释边界 |
|---|---|
| 原CC-1、恢复v1工程失败及恢复v2完整能力负结果 | [CC-1失败结案](PLAN-R2-DAILY-DIALOGUE-CONTINUATION-CC1_execution_closeout_20261025.md)、[恢复v1失败](PLAN-R2-EVALUATION-RECOVERY-V1_execution_closeout_20261025.md)、[恢复v2结案与R1–R6对账](PLAN-R2-EVALUATION-RECOVERY-V2_execution_closeout_20261025.md)、[固定48行判读](PLAN-R2-EVALUATION-RECOVERY-V2_human_review_20261025.md)；旧失败和已消费资格不因后继完成而抹除，完整结案不等于L2通过。 |
| 原CC-1工程交接、训练来源与冻结合同 | [02工程交接](delivery-runs/20261021-n-mainline/artifacts/02-plan.md)、[技术摘要](delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md)、[17保护清单](delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md#4-受保护冻结文件)；原合同与来源SHA保持原位，旧“当前”仅解释其记录时点。 |
| N1-167至N1-174C诊断与候选系列 | [167结案](PLAN-N1-167_execution_closeout_20261025.md)、[168结案](PLAN-N1-168_execution_closeout_20261025.md)、[169结案](PLAN-N1-169_execution_closeout_20261025.md)、[170结案](PLAN-N1-170_execution_closeout_20261025.md)、[171结案](PLAN-N1-171_execution_closeout_20261025.md)、[172结案](PLAN-N1-172_execution_closeout_20261025.md)、[173结案](PLAN-N1-173_execution_closeout_20261025.md)、[174C结案](PLAN-N1-174C_execution_closeout_20261025.md)；具体读数沿原件，架构/数据/目标等解释按[前序原因复盘](PLAN-N1-175_prior_results_cause_review_20261026.md)收窄，不改写原失败。 |
| N175/N176合同、评价交接与架构核实 | [N175合同](SPEC-N1-175_data_and_completion_contract_20261026.md)、[N175评价交接](PLAN-N1-175_evaluation_handoff_20261026.md)、[N176合同](SPEC-N1-176_F0_F1_pilot_20261026.md)、[架构主张审计](PLAN-N1-176_architecture_claims_audit_20261026.md)；本次只整理活动计划路由，原题面、预算、阈值、源码绑定和消费状态不在此变更。 |

旧历史队首退出活动调度；未完成的合同责任由03发布当前动作、10维护具体依赖与有限出口、09保留机制及42条来源身份。不能把“转入历史索引”解释为完成尚未完成的评价、撤销失败或放宽冻结规则。

同次黑板收束：`plans/reference/delivery-runs/20261021-n-mainline/board.md`旧全文亦固定在上述提交，Git blob `6267461235d298fb8dde7400921dd8ebbb0e32a2`，LF SHA256 `9443a599182760f3247f49f9f9d034a73f40cbfa554b3f1623bb16ffacdc71dc`，26694字节/114行。黑板只保留阶段、决策与风险摘要，旧逐轮快照按此恢复，不复制一份新历史正文。

## 7. 2026-10-27 已结案R2中间检查点清理

按用户要求清理已结案R2恢复v2的中间训练快照：逐个核对原回执SHA后删除2,869个`.pt`，释放28,754,240,519字节（约26.78 GiB）。保留45个模型文件，包括各阶段首尾、latest及非流水直接引用的检查点和六个正式终点；39个带回执的保留点与六个正式终点均完成哈希核对，12个latest指针未变。全部原始回执和历史流水保留，具体路径、哈希及结果见[清理清单](../../reports/taiji_r2_historical_checkpoint_prune_20261027.json)。

历史流水仍引用已清理的中间模型文件；这些文件不能再在本机逐字节复核或从中间步恢复，回执和最终端点仍可用于相应的历史追溯。上文旧轮次的“没有删除checkpoint”仅描述当时事实，不适用于本次明确授权的清理。N177及其父checkpoint不在清理范围。

## 8. 2026-10-30 项目级整理收束

以`f43b51b344ad067a092e8df03895c9faba8d7f1e`为核对基线，按Git清单盘点源码、API/runtime、测试、脚本、报告、计划、客户端及根目录；依实际消费者和独有价值清理，而非把旧N源码一概视作废弃。机器动作清单见[project_consolidation_20261030.json](../../reports/project_consolidation_20261030.json)，归档与逐字恢复入口见[历史索引](../archive/history/n_mainline_20261030/README.md)。

实际归档11文件：根目录旧CI交付的7份Markdown与1份未完成patch、2张UI观察截图、1份旧N路线交接。已删除26文件与空旧前端更新目录，共108,750,241逻辑字节（约103.71MiB）：旧black/mypy缓存、已摘要两份JUnit探针XML及已收官归因交接。缓存无tracked项且超过1小时未更新；归档原件逐文件核对哈希，旧交接只调整相对链接并增加历史标识。已删除tracked交接的提交/路径/blob与LF哈希已核实，可从Git恢复。

旧CI目录的取证、未应用补丁和未决事项继续留存，归档不代表当前CI问题已解决；两份JUnit的12通过/4跳过仅是历史探针，不代表当前HEAD全绿。截图不代表模型能力。正在推进的R5b报告、数据/checkpoint、含源码或模型的.seed_test_tmp、运行时/CI直接读取的旧代码/合同、最近更新的ruff缓存均保留。.zcode仅是本机插件启用配置，保留原位并纳入本机配置忽略/目录台账，不隐藏其他未知根目录。

本轮没有改变模型、训练/评价源码或默认入口，不运行模型、训练、恢复或消费评价数据。后续执行仍按新N主线08的合同一致性与工程回执推进；全项目收束不另立研究队首。

跨日收尾（2026-10-31）：移除仅含空.tmp子目录的.screenshots旧根。目录/凭据守卫使用本轮专用basetemp完成14/14，单主线审计通过，现行层链接0坏链；未运行远端CI。初次系统临时目录错误与空根守卫失败均保留在机器清单验证说明，不写成全程无失败。

暂存后补验：目录/凭据守卫与单主线合同共16项通过；归档补丁的空白上下文前缀为unified diff语法，原样保留，其余本轮路径diff检查通过。归档工作树哈希与Git规范化字节哈希分别记录。
