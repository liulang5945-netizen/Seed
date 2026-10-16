# 77 全fresh三种子三臂固定训练（2026-10-16）

执行76已冻结正式配方，详见 [76](PLAN-N1-76_fresh_paired_admission_20261016.md)。无范围/原门/预算变更。

seed11/12/13×position/terminal/cut，分别从74原零档加载，完整参数62056、空Adam/历史。各24576步、36864QA，共221184步/331776QA；原43流6144步重复4次，0/6144/12288/18432/24576各576已知评分，九模型25920行。单线程串行CPU≤9000秒含前置/18独立恢复/72副本步/144合成QA/压缩与守卫。不得按充分性提前停控制，不循环追加。

position/terminal全部41 Adam步数等于完成数；cut writer4无Adam且逐位不变、其余37步数相同；每步新session，完整bundle/RNG恢复两步签名。zero/final各一次独立恢复；partial不得作为完整或自动续跑。每模型无损gzip6mtime0分片包含24576loss/位移与2880原始行，主报告双层SHA/字节数/身份/精确计数全核验；缺片/改片拒绝。

产物为76声明的reports/taiji_n1_fresh_paired_train_20261016*.json[.gz]及data/checkpoints/n1_fresh_paired_train_20261016/，执行器scripts/training/run_taiji_n1_fresh_paired_training.py、对应artifact-free合同；不改冻结74/75和模型。错误/预算触顶保存全部已得证据与partial，整批停止，不自动重启。

训练0dev/v3holdout/v4/v5查询，仅封存字节守卫。全部九模型/18恢复完整、三个position终点原四门通过后，另版78才一次288项v5查询；14原优势门不变，每seed≥16/32、每类≥2/8，position分别领先terminal/cut各≥12/96。否则不查询，保留失败并按训练证据诊断。保持/C4/三器官/N系列其他出口不替代。

## 实施前置完成

8项artifact-free合同通过：三种臂分别零档独立进程两步签名恢复、三臂synthetic final完整恢复/切步计数拒绝、cut writer逐位不变/无Adam、partial无损分片/缺计数拒绝、固定四周期与已存在批次拒绝。31来源与九74零档只读守卫验证，v5未查询；两根Ruff与改动Black通过。为避免Windows checkout改变封存SHA，v5精确路径固定eol=lf，九gzip命名族声明binary；未重写任何旧档。

## 首组已得证据（不代表整批完成）

77 seed11三臂完整封存，各24576更新/2880已知行/两独立恢复；position原186/改名184、terminal123/105、cut192/189，position/cut配对均96/96，cut四writer无Adam且不变。三片完整计数与原门核验通过；同会话73963已自动进入seed12 position。唯一下一步完成其余六模型并统一核验，78仍未绑定最终SHA、v5零查询，全项目出口债保持。

| seed11臂 | 原题/192 | 原pair/96 | 改名/192 | 改名pair/96 | 分片SHA256 |
| --- | --- | --- | --- | --- | --- |
| position | 186 | 96 | 184 | 96 | 04cb12e90e01e4bd649a8c3b3bacf8bdd633c29d98f50b69ca009267732253af |
| terminal | 123 | 31 | 105 | 26 | ead9b27391ce3da64a00679fce85145278871e0e3574ae7917a3c5a79122120e |
| cut | 192 | 96 | 189 | 96 | 0d741c004b33c84ec4fc26f40db89928cf554bc49c1301668b15233fa39939ef |
