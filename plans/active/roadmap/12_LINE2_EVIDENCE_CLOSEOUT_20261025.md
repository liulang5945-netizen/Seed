# 线 2（证据入库）收口计划——R2 结案后的 deselect 清退路径

状态：**分析与清单，待 owner 裁定入库范围**（证据治理是 owner 决定，agent 不代提交）。
背景：R2 已按 CLOSED_FAILURE 结案（CC-1／恢复 v1／恢复 v2 三件，均为工程故障
结案而非模型负结果；R6 结案文档已落盘入库）。账本冻结：三臂合计
4,513,257 CPU 秒、无活跃预留——训练停止，deselect 清单里约 60 条
「缺未入库产物」的测试进入可清退窗口。

## 现状盘点（2026-10-25 实测）

- **批 A｜`reports/` 未跟踪 R2 证据**：209 个文件、合计 407KB、**全部 LF**
  （R2 时代文件在 `.gitattributes` 之后产生，入库零归一风险，哈希钉不受影响）。
  覆盖：extension_v1 全套 attempt/result/readiness/claim、budget_amendment
  v1.4/v1.5、endpoint_manifest、target_update/resume/fast_ledger 各轮读数。
  入库即解锁：n5_38 两条、r2 族（evaluation_adapter×3、fast_ledger_resume
  v2/manifest、warmstart、extension_contract 6 条）等约 15 条 deselect。
- **批 B｜`.gitignore:157`（`*.jsonl`）屏蔽的在飞进度记录**：N1 链的
  progress/attempt jsonl（如 plan160a）对 git 不可见，plan157/160/161 三条
  消费它们。清退需对该批**逐个 `git add -f`**（或为证据 jsonl 加否定规则）。
  reports/ 下被 ignore 的文件共 11.4 万个（多为训练 exhaust），只挑测试
  消费的少数几个，不批量解禁。
- **批 C｜`data/checkpoints` 的 `.pt` 产物**（plan142/143/144 登记表、
  n5_15、reader_transfer、n5_16 等消费）：89G 量级，**git 直存不可行**。
  选项：①维持 deselect 至有 LFS/制品库；②测试改为「产物在位才跑」的
  skipif 形态（本地仍全量守护，CI 显式跳过并把重启条件改成制品库就绪）。

## 建议的裁定问题

1. 批 A 209 个文件是否全量入库（含 paired_eval 的 36 个 claim 分片）？
2. 批 B 涉及的 jsonl 清单（入库时按 deselect 测试的引用反查，逐个 add -f）；
3. 批 C 选①还是②。

## 执行纪律（裁定后）

每批入库＝一次独立提交；入库后跑 `verify_ci_deselect_list.py` 自证、
按 deselect 逐条摘除并盯 CI 双腿绿；批 B/C 涉及的哈希钉如报失配，
按 N1-121 方法论（活跃钉口径）重冻结，不扩大面。
