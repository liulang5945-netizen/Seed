# PLAN-G138 `tests/` 侧写靶判别式的判据（先冻件，实施排在下一格）

冻结时刻：2026-10-14（㊵-812）。这条件把 03 队首⑤ 里扩面那句——原文逐字＝**"属改判据、要单独一轮"**——
落成可判的判别式；本轮**只冻判据、只做前置测量**，不动任何测、也不动 `audit_taiji_sealed_report_overwrite_surface.py`
的既有口径（`37／36／22／21` 一字不改）。判别式的出处不是我的直觉，而是 DEBT-G138 修法① 那条
"从被调工具自己的 argparse 取写靶，不在测的源码里猜旗标语义"。

## §0 现状（全是现取，可复算）

1. 只读探针（仓外一次性脚本，未入库）扫 **717 枚**含 argparse 的 `scripts/**.py`、共 **2637 个旗标**，
   按"该旗标的值在工具自己的 AST 里最终进了哪个调用"分档：**write 762／read 304／both 96／unresolved 1475**
   ⇒ 未决率 **56%**。这条读数否证了"全仓判别式可以便宜做出来"，本件因此把域收窄（§1）。
2. 探针在三个**已知形状**上是准的：`audit_taiji_n5_22_cap0_pairing.py` 的 `--out-report`＝write
   （`out->write_json_report@244`）、同件 `--treated-report`＝unresolved（人读源码它是**读源**：`:212` 把
   `(args.treated_report, args.control_report)` 交给 `:47` 的 `_load`，但"值经 for 循环元组进别的函数"
   这一形探针接不住 ⇒ 见 §1 第 3 条的处置）；
   `audit_taiji_batch_c_arm_parity.py` 的 `--out-report`＝write（`out->write_json_report@333`）。
3. 探针在四个已知形状上**不准或不够**，成因各不相同，实施格必须逐条解决而不是绕开：
   a) `audit_taiji_n5_23_repeat_arm_noise_floor.py` 的 `--out-report` 读成 unresolved——真实写法是三元式
      `out_path = args.out_report if args.out_report.is_absolute() else PROJECT_ROOT / args.out_report`，
      朴素"直接赋值"匹配看不见三元式（＝[[ast-classifier-blind-spots-three-shapes]] 的第六形）。
   b) `eval_taiji_cap0_baseline.py` 的 `--report`／`--checkpoint`／`--worksheet`／`--adjudicate` 全部 unresolved
      ——真实链是 `:1074` 的三元式绑定（`report_path = args.report if args.report.is_absolute() else
      PROJECT_ROOT / args.report`）把值交给**另一个函数的形参 `path`**，写靶落点在该件 `:80` 与 `:809`
      （`report_io.write_json_report(path, ...)`／`write_text_report(path, ...)`），模块级 walk 接不上这条改名线；
      而同件 `:947` 还有一条 `if args.report == DEFAULT_REPORT:` 的**比较式**读法 ⇒ 同一枚旗标在同一个工具里
      既可当写靶又可被读，按旗标名归类必错。
   c) 测侧的旗标字面量本身脏：`test_n1_09_envelope_recipe_face_contract.py:131` 里有
      `raise AssertionError("--only 打空必须拒绝")`、`test_n5_15_pairing_preflight_contract.py:247` 里有
      `raise AssertionError("--help 没有按惯例退出 0")`——**断言消息**被我的探针当成了一枚旗标
      （名字面匹配会把一句断言的话读成一个参数名，同 [[ast-classifier-blind-spots-three-shapes]]）。
   d) 测侧有 22 枚文件同时出现 `REPO`＋`reports`，其写靶形状与读源形状混在同一枚文件里（如
      `test_cap0_baseline_contract.py` 有 6 处写靶调用与 `--baseline`／`--candidate` 这类读源旗标并存），
      ⇒ **按文件归类必错**，只能按"工具＋旗标"这一对归类。

## §1 判别式（本件冻的就是这四条，实施格不许换形）

* 域＝**只包含在当前 tracked 测的调用面里真的出现过的（工具，旗标）对**，逐对出版，不宣称覆盖全仓旗标。
  实施格的第一支契约测＝"域内每一对都能在 `scripts/**.py` 里解析到同名 `add_argument`"，缺一支即红。
* 角色只从**被调工具自己的 AST** 取：该旗标 `dest` 的值在工具内最终进入 `write_text_report`／
  `write_json_report`／`write_text`／`write_bytes`／`open(..., "w")` 的**第一个实参** ⇒ `write`；
  进入 `read_text`／`read_bytes`／`json.loads` 的**被读对象** ⇒ `read`；两条都命中 ⇒ `both`；
  接不上 ⇒ `unresolved`。**不在测的源码里猜旗标语义**（㊵-807 立的修法①）。
* `unresolved` 与 `both` **一律不入写靶桶**，并逐枚点名出版。理由：本桶的用途是"谁可能覆写在库件"，
  把未决读成写靶＝造出假暴露面；读成读源＝漏掉真暴露面 ⇒ 只能单列，且**未决数本身就是读数**。
* 三元式、**经 for 元组传参**、跨函数改名绑定与 `Path(...)` 包裹这四种形状**必须被解析到**
  （§0 第 3 条 a／b 与第 2 条那条就是反例），
  实施后 §0 第 1 条那组 `762／304／96／1475` 必然变化 ⇒ 变化值要现取入台账，不许沿用探针旧数。

## §2 验收（四条各自写明"什么情况下为假"）

* **J-G138-1 反例集必进对桶**：`test_p3b_review_worksheet_contract.py:201` 那枚假阳性（`:200` 有
  `monkeypatch.chdir(tmp_path)`）必须**不入**写靶桶；`test_a30_lf_conditioned_rule_bound.py:195` 的 `--report`
  与 `test_n5_15_pairing_preflight_contract.py:224` 的 `--replay-before`＋`--root` 必须读成 `read`。
  为假的情形＝这三枚有任何一枚进了错桶 ⇒ 判别式按名字匹配，不按工具数据流。
* **J-G138-2 经验复核是仲裁者不是装饰**：㊵-807 现读那 **6 枚启发式候选**各跑一遍，跑前跑后各取一次
  `git status --porcelain -- reports/`；凡"分类为 read 却出现内容真变" ⇒ 该对**降级为 unresolved 并点名**，
   classification 让位于观测（㊵-794 已证明这条经验法有效）。为假的情形＝只出 AST 数、不跑这 6 枚。
* **J-G138-3 既有口径不许被顺带动**：扩面之后 `default_targets_on_tracked_files`／`distinct_files`／
  `exposure_targets`／`exposure_files` 四个数必须**逐位不变**（＝37／36／22／21）；新桶另起一名。
  为假的情形＝把测试侧的数并进原桶，或原桶因"识别面变宽"而悄悄变化。
* **J-G138-4 不许把新桶读成全集**：件里必须同时出版"域内对数""write 数""unresolved＋both 数"三个数，
  引用时只许说"在测的调用面里可判为写靶的 N 对"，**不许**说"tests/ 的写靶已清点完"。
  为假的情形＝件里只有一个总数、或文档把 `test_write_targets` 写成"tests/ 全部写靶"。

## §3 守卫

* **G-G138-1 断言消息不许当旗标名**：测侧取旗标字面量时要求整串匹配 `^--[A-Za-z0-9][A-Za-z0-9_-]*$`，
  §0 第 3 条 c 那两枚（`:131` 与 `:247` 的 `AssertionError` 消息）必须落在门外；负对照＝把该正则放宽成前缀匹配，命中数必须变大（守卫能为假）。
* **G-G138-2 一次性探针不得充当仪器**：判别式必须落成一枚入库仪器（暂名
  `scripts/training/audit_taiji_test_write_target_discriminator.py`）＋契约册，仓外探针只算前置测量；
  台账与债表引用读数时**只许**引入库那枚的出版值。

## §4 落笔即钉的三行

* **预期绿**：J-G138-3（本件只冻不实施，实施格再验）；J-G138-1 的"假阳性不入桶"那一支形状性可当场验。
* **仍红时收回哪句**：J-G138-2 红 ⇒ 收回"tests/ 侧写靶已可机检"，把那一读回落到 ㊵-807 那句"卡判别式"，
  并把 §0 第 3 条的 a／b／c 三形逐条点名回来；J-G138-1 红 ⇒ 判别式退回按名字匹配，扩面整格作废。
* **发表资格前置**：本件全绿之前，#80 那句"覆写普查面扩到 `tests/`"不许出现在任何"已收口"清单里；
  全绿之后发表的也只是"**在测的调用面里**有 N 对写靶可判、M 对未决"，**不是**"tests/ 的覆写风险已清点"。
