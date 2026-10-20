# PLAN-N1-159A 执行结案

**日期：2026-10-20**  
**状态：完整续接；seed129 face全部endpoint均完成**  
**计划：** [PLAN-N1-159A](PLAN-N1-159A_per_endpoint_summary_correction_20261020.md)  
**seed129面：** [题面清单](../manifests/n1_plan159_v31_candidate_parent_seed129_20261020.json)

## 完整性与资源

首轮已保存的seed11 candidate 480条只读并入；N1-159A新增其余五个endpoint各480条，共2,400条新查询，合并覆盖seed129面完整2,880条。六个checkpoint独立恢复6/6，source/face/checkpoint哈希稳定；progress只记录2,400条续接查询，逐行有效。未重复seed11 candidate，也未重新生成题面。

N1-159A新增阶段累计CPU273.03125/3,600秒，关联项目累计CPU10,229.453/18,000秒；0训练、0参数更新、0 dev/holdout/confirmation，`can_promote=false`。完整逐题结果和进度绑定见[结果报告](../../reports/taiji_n1_plan159a_candidate_parent_result_20261020.json)。

## seed129 fresh面描述性读数

三个candidate endpoint的总命中为seed11 451/480、seed12 470/480、seed13 463/480；对应N1-150父模型406/480、466/480、452/480。合计candidate 1,384/1,440、父模型1,324/1,440。

unknown正确弃答按8/16/32行，candidate为93/120、103/120、118/120，父模型为66/120、88/120、101/120。逐题配对中unknown candidate-only/parent-only为8行35/1、16行20/2、32行14/0；已知类合计candidate 1,070/1,080、父模型1,069/1,080。该读数在seed129复制了seed128面上“unknown优势主要来自candidate、known整体近乎持平”的趋势，但两张synthetic face仍不足以构成现实能力确认或晋级证据。

## 后续唯一队首

依据用户授予的执行计划批准权，已批准[PLAN-N1-160](PLAN-N1-160_seed130_candidate_parent_fresh_replication_20261020.md)：以全新seed130面再做一次candidate对同seed父模型的独立复制，样本、endpoint、分母及判读边界保持不变；CPU≤3,600秒，关联累计≤18,000秒，不训练、不访问dev/holdout/confirmation、不晋级。
