# 交接件 · N1 在 v6 消费之后停靠（owner 裁"先写交接、停一停"，2026-10-17）

owner 在 v6 出值后的弹窗里裁：**先写交接、停一停**（未选甲／乙／丙／丁），推送一事答"无偏好"。本页因此只做三件事：把**不可逆事实**钉住、把**队首**写成可裁的形状、把**接手者今天就能自己跑的命令**逐条附上本轮真跑过的读数。本页不是判读件（判读＝EXP-N1-96），也不是清单页的替代（未推清单＝UNPUSHED_COMMITS_20261017）。

## 0. 落笔时的现取锚点（每条都出过命令，不是回忆）

- `git rev-parse --short HEAD` ⇒ **`48b508d67`**；`git rev-list --count origin/main..HEAD` ⇒ **24 条未推**；远端 `origin/main` 停在 `69f7b5254`。
- 工作树里有**别人**的在飞件：`reports/taiji_n5_07_*.json` 六枚被改、`plans/reference/DSH_*` 两枚未跟踪、`output/**` 若干未跟踪目录（现取 `git status --porcelain` 去掉 `output/` 后 9 行）⇒ 接手者**不要** `git add -A`，也**不要** `git checkout HEAD --` 整树回退。
- 本轮我的提交共四笔：`cd4632d26`（签署＋G152 文句）、`d0e6b24a4`（v6 三步＋判读＋仪器修口）、`4cbfd9a1c`＋`48b508d67`（未推清单页与它的自指更正）。

## 1. 不可逆事实：两张未见面都已消费

| 面 | 状态 | 证据字节 |
| --- | --- | --- |
| v5 | 已消费、判否（`retry_allowed=false`） | `plans/manifests/n1_fresh_confirmation_v5_20261016.json` ＋ EXP-N1-90 |
| v6 | 已消费（`consumed_face="v6-confirm-20"`、`retry_allowed=false`），落预注册**出口 ⑵** | `plans/manifests/n1_fresh_confirmation_v6_20261017.json`（面 SHA `7dddf00f3f2e24381728855b218e67bc8160fd6badeb1a7a933937d999766d49`，与 attempt 件里钉住的值同串）、`reports/taiji_n1_transition_factorial_confirmation_v6_attempt.json`（`consumed_on_write=true`）与同名正式件 |

⇒ **N1 侧在"不新建面"的前提下没有任何可发射的一次性查询。** 任何复用只能标"已知面上的机制探针"，不得冠"未见"。执行器现在连 `--readiness` 都会响亮拒绝（本轮实测 rc=2，`REFUSE: confirmation report already exists`），这是设计而非故障。

## 2. 队首＝owner 四候选（每条写清"要他给什么"，我不自择）

- **甲（上限最高，只需一句话放行普查）**：攻"位置臂从不弃权"那一格。已知证据：`unknown` 类 5 题 × 3 seed，position 输出 `unknown` 的次数 **0/15**，terminal 4/15、cut 2/15（同题集同长度轮转）。第一步是**零算力普查**（只读已出版的 180 行＋run89 的 288 行），普查成立才谈训练配方或产品码。要 owner 给的：允许我做普查＝一句话；真要动权重或改默认＝另一次批文。
- **乙（要签署）**：造 v7 面并再花一次一次性消费。今天已知的是 d=1 的尺**能开线**（7/9 ≥ 6），所以 v7 买到的仍是"上限卡在哪一格"的第二次读数；若想让四门有机会变绿，必须同时重述判据（那是另一件要签的事，我不自择）。
- **丙（要批文）**：N1 按两表面结项，转 C4 净增益 B 档——一次跑同时补齐判读器点名的三个字段（`usable_endpoints`、与增益同量纲的 `noise_floor_delta_quality_abs`、保持集）。这正是 owner 上一轮裁"先停下"的那格。
- **丁（零算力，我已排好）**：N1 挂着，做保持集／C4 口径／三器官那几格＋清债表 27 行过期状态列。

判级原文与逐格读数都在 [EXP-N1-96](EXP-N1-96_v6_confirmation_adjudication_20261017.md) §3／§4；构造与准入线在 [PLAN-N1-93](PLAN-N1-93_v6_unseen_face_construction_20261017.md)（§1 已签署＝草稿形状＋`name_neighbor_distance=1`）；四门线值在 [PLAN-N1-94](PLAN-N1-94_v6_confirmation_query_20261017.md)（**该页被确认报告按 `plan_sha256` 钉住，任何改动都会作废那次消费的可复现引用 ⇒ 不许就地加戳**）。

## 3. 接手者今天就能自办的三条（命令均本轮真跑，rc 与读数照抄）

1. **复算四门（零查询、零装载）**：从已出版件的 `rows` 直接调执行器自己的判门函数，别手算——手算这条族已欠过三次债（DEBT-G153）。

   ```
   python -c "
   import json,sys;sys.path.insert(0,'.')
   from scripts.training import eval_taiji_n1_transition_factorial_confirmation_v6 as v6
   e=json.loads(v6.REPORT.read_text(encoding='utf-8'));g=v6.evaluate_gates(e['rows'])
   print(json.dumps({k:v['passed'] for k,v in g['gates'].items()},ensure_ascii=True),'all_four',g['all_four_gates'])
   "
   ```
   本轮实测 rc=0 ⇒ `{"each_position_seed": false, "each_position_kind": false, "advantage_terminal": true, "advantage_cut": false} all_four False`（件里值：12/12/1 对线 10；`advantage_terminal` 17/60；`advantage_cut` 0/60；CPU 151.7344 秒、180 条查询）。
2. **复跑牺牲半范围检验（可反复，它不承载主张）**：`python scripts/training/verify_taiji_n1_v6_range_check.py --readiness` ⇒ 本轮 rc=0（九枚端点齐、`confirm_queried=false`、上限 180 秒）。正式支 `--execute-approved` 会因写靶已存在而拒（一次性纪律），要重测就换 `--out-report` 到仓外并自己确认不覆盖在库件。现读结论：过线端点 **7/9**（逐枚 `addressable_errors`＝11 号 5/13/7、12 号 4/13/6、13 号 12/13/13，顺序 position/terminal/cut，门槛 ≥6）。
3. **引用债表某行状态之前先跑普查仪**：`python scripts/training/census_taiji_debt_row_effective_status.py --out-report C:/Users/23747/AppData/Local/Temp/<任意新名>.json` ⇒ 本轮 rc=0，现取 `strict_rows`＝`distinct_ids`＝`rows`＝**155**、`stale_head` **27** 行（历史遗留，属丁档，没人清过）、`reopened_by_stamp` **0**、`unclassified_stamp` 36。别按列取状态：G125 那行我就是这样误引过一次（08 ㊵-1041）。

## 4. 永久不许写的句子（点名收回句，出处是本轮读数）

- **"位置读出臂在未见集上领先"** ⇒ 合取假且 `advantage_cut` 为 0；单门过线只许写成"`advantage_terminal` 这一门 17/60 过线"。
- **"C3 达成／N1 可晋升"** ⇒ `can_promote=false`、`all_four_gates=false`。
- **"换个 seed 或再跑一次确认半"** ⇒ `retry_allowed=false`，两张面都消费了。
- **"弃权类是尺的问题"** ⇒ 同一张面 terminal 4/15、cut 2/15 答得对。
- **"范围门与四门互斥"** ⇒ ㊵-1036 已收回（源自我按每 seed 与 12 比的手算，冻的是 12/60 合计）。
- **"unseen_claim_allowed 这一列证明仪器认了这次消费"** ⇒ 2026-10-17 之前出版的件里它是恒真字面量（DEBT-G155 已修成推导）；要证消费请看 `attempt` 件与 `complete_checks`。

## 5. 门禁基线（本轮现取，接手者拿它当"我有没有新加红"的尺）

`ruff check .` rc=0；`ruff check . --select B,SIM` rc=1／**11 枚**（我名下 0）；`black --check .` rc=1／**18 枚**会重排（我名下 0，本轮新增件已单文件 `black --check` 干净）；`check_plans_markdown_links.py --json` rc=0／450 件／2190 链／活动坏链 **0**；`audit_taiji_debt_table_column_shape.py` rc=0（155 行全五列、`rows_not_expected_cells=0`）；措辞门对 PLAN-N1-93 rc=0（`criterion_lines` 16）、对 EXP-N1-96 rc=0（8）；`tests/taiji_native/test_n1_run93_v6_face_builder_contract.py`＋`test_n1_run94_v6_confirmation_contract.py`＋`test_n1_run93_v6_range_check_contract.py` **28 passed**；`tests/seed/test_project_identity.py`＋台账圈号唯一性测 **10 passed**。全族面（`pytest tests/`）本轮**没跑到**（被权限分类器拦，按规矩不硬重试）⇒ 它的数字是继承的，不是现取的。

## 6. 我这一轮踩到的两次当场红（写给接手者的两条防线）

- 新写一条守卫，**第一次执行必须打在一份已知-good 的真件上**：我把 `unseen_claim` 的总行数写成 20×3＝60（真面 20×3 seed×3 臂＝180），是自己的新测当场判红（7→9 passed 之前先 2 failed）。
- **盘面状态一律不进合同测试**：我在造面器测里写了 `assert not v6.REPORT.exists()`，题面一落盘它就红（1 failed／27 passed）——这是 DEBT-G148 记过的同一族，我仍犯了；改法是 monkeypatch 写靶到 tmp。

## 7. 一页索引

判读 `plans/reference/EXP-N1-96_v6_confirmation_adjudication_20261017.md`｜构造 `PLAN-N1-93…`｜判级 `PLAN-N1-94…`｜预飞 `EXP-N1-95…`｜未推清单 `plans/reference/UNPUSHED_COMMITS_20261017.md`｜台账 `plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md` ㊵-1042／1043／1044｜队首句 `plans/active/roadmap/03_CURRENT_EXECUTION.md` 第 3 行｜N1 阶梯行 `plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md` §2 第 6 项。

## 8. 停靠说明

owner 裁"停一停"，所以本页落盘后我**不起跑任何新格**：不造 v7、不发射查询、不动权重、不跑训练；C4-B 与 N2 甲层继续按上一轮"先停下"停着。下一次开工的**第一个动作**是回读本页 §0 的三个锚点（HEAD／未推数／别人在飞件），因为它们是会过期的一格。
