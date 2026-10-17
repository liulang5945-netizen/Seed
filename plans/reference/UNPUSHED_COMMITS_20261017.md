# 未推送提交清单 · 2026-10-17（现取，非回忆）

推送在这台机器上被权限分类器拦下（本轮按 owner 裁定又试了一次，仍拦），所以把清单与命令写成一页，
方便在手机端或有终端时一次推完。**这条命令只推已经提交的 22 笔，不含任何新提交。**

- 现取命令：`git rev-list --count origin/main..HEAD` ⇒ **22 条**
- 远端 `origin/main` 目前停在 `69f7b5254`，本机 `HEAD` 在 `d0e6b24a4`
- 推送命令（PowerShell，可直接粘）：

```
cd E:\Seed
git push origin main
```

- 若推送被拒（非快进），**不要** `--force`；先把报错原文发我，我按 `git fetch` ＋ 变基或合并处理。

## 逐笔（新→旧，`git log --oneline origin/main..HEAD` 原样输出）

| SHA | 标题 |
| --- | --- |
| `d0e6b24a4` | research(taiji): run the signed v6 unseen face and adjudicate it at exit (2) |
| `cd4632d26` | research(taiji): land owner signature d=1 on the v6 unseen face and rule DEBT-G152 |
| `d3a2c8031` | research(taiji): census the debt register for status cells that their own row has already superseded |
| `7d357e26d` | docs(taiji): retract a stale debt-table status cell I quoted as current in the audit |
| `b6075562d` | docs(taiji): refresh the completion audit to today's readings, including the C4 and C5 dimension |
| `745567546` | research(taiji): design the C4 resource-versus-quality face and correct an earlier underestimate |
| `bf33580c5` | research(taiji): ship the C4 reader and name the three input-contract fields the rung is missing |
| `3265841b5` | docs(taiji): pre-register the C4 net-gain package, with a per-deliverable face audit |
| `359aa291e` | docs(taiji): correct the stale prescriptions carried in the queue-head lines |
| `cb96934de` | research(taiji): retract the range-versus-gates exclusivity claim, recomputed through the gate function |
| `b413a9c40` | research(taiji): an in-distribution difficulty axis opens the v6 ruler, and exposes a gate exclusion |
| `99998db8c` | research(taiji): pre-flight the v6 lane on real endpoints before asking for a signature |
| `9b110cbfa` | docs(taiji): record that the v6 confirmation executor is pre-written and tested |
| `6e29b6e62` | research(taiji): build the one-shot v6 confirmation executor before there is a face to consume |
| `cad4accff` | research(taiji): close a shape leak in the v6 face and freeze the confirmation query early |
| `a05a9f607` | research(taiji): split the v6 face into halves and gate the range check on code, not discipline |
| `3f1189fc1` | research(taiji): build the v6 face generator and let a draft run falsify my own threshold |
| `939ac76e9` | docs(plans): record the ruling to build a new unseen face and draft its signable spec |
| `31eaf3e8a` | research(taiji): complete the oracle address probe and let the frozen rule name exit 4 |
| `44f1435f2` | research(taiji): write the class adapter and let its own anchor reject the first version |
| `2c9e894f4` | research(taiji): freeze the oracle probe, pass its anchor, and falsify its own reuse plan |
| `a34c61539` | research(taiji): land the owner ruling and pin the generation cap to its protocol key |
