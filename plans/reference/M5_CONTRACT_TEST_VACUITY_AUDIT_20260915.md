# 合同测试普查结果：恒真断言与"只读封存报告"的结构性局限（2026-09-15）

日期：2026-09-15（本地 02:32）。方法：对 11 份合同测试逐条问"这条断言在**什么实现下会失败**"，
凡答不出即为候选；每条候选都回到被测函数/被测脚本核对后才入册（普查代理的结论**不直接采信**，
其中"跨文件夹具泄漏"一条已用反序收集实测复现）。

**判据（写死，避免日后争论）**：一条断言只有在"被测代码或产物变化时可能变红"时才算在保护什么。
`x == helper(x)`（`helper` 是 `return x`）、由相邻断言在**所有输入上**逻辑蕴含的断言、
遍历"本该被校验的那个集合"来证明该集合性质的循环、对散文/标识符做子串 grep 的"结构性"断言，
都是**零覆盖**。它们比没有测试更糟：制造信心。

## §1 已修（提交 `06f6396e`，均在 `test_p3b_campaign_contract.py`）

| # | 位置 | 原缺陷 | 修法 |
|---|---|---|---|
| 1 | provenance 测试 | `trainer.ARM_MANIFESTS = {...}` 直接赋值泄漏进另一份测试（module-scoped + sys.modules 缓存），**只因字母序才绿** | `monkeypatch.setattr`；已用两种收集顺序各 42 passed 验证 |
| 2 | 受保护检查点测试 | 遍历 `PROTECTED_OUTPUTS` 自身 ⇒ **对成员减少完全失明** | 加三条成员断言（含 campaign 侧 ⊆ trainer 侧），行为断言钉在显式两件上 |
| 3 | 预算档测试 | `symbols < 559_000_000` 由上两行蕴含（约 4.9e7 必成立），且那个数是**旧子集**尺寸 | 改读两份 arm manifest 的 `emitted_symbols` 实际比较 |
| 4 | `stop_definitions` 测试 | 八个 token 里**五个**由标识符/docstring 满足；删掉整块 payload 仍绿 | 定义升为 `STOP_DEFINITIONS` 常量（记录值逐字不变），断言核结构/数字/margin/四字段 |
| 5 | headline 测试 | 单段 stages ⇒ `latest` 取第一还是最后不可区分 | 两段，断言取后者 |
| 6 | 上一提交我自己加的路径测试 | docstring 超售（只能抓路径漂移，抓不到"评拷贝"） | 改名 `..._agree_on_the_arm_path` 并把两种危害分指各自测试 |

**顺带发现（不属于测试缺陷，属于工具语义）**：N2 清单扫描器是**纯子串匹配**，
把扫描器自己的文件名（路径含那个受审计令牌）写进注释，就会把该文件拉进消费面——
本次真被它抓到并把 `test_current_review_surface_is_complete` 弄红。
处置是按角色指代而不写路径，**不是**把误报登记成第 18 个消费者（该文件确实不读那个字段）。

## §1b 追加（本地 02:35，同一批普查的 A1–A3 与两处缺失测试）

- **A2**：`make_p3b_review_worksheet.py` 现打印**每维待复核题数**（`B：20 题；G：20 题`），
  空维度不再与满表长得一样；测试两种情形都断言该串。
- **A1**：测试改为**正向**断言"评分位为空"（正则取 `**评分（0/1/2）**：` 之后的内容，
  并去掉 `$` —— 非 MULTILINE 下它只会在文末匹配一次），并把 `机检预判` 参数化过
  `precheck_pass / precheck_fail / hard_safety_risk / precheck_skipped` **四个真实取值**。
  原先的禁用词表恰好把这四个全漏掉：那是一支"挡不住预填分数"的不打分测试。
- **A3**：`DEFAULT_OUTPUT` 改为与真实的封存/在跑报告路径**求交为空**，不再靠"文件名里没有
  baseline 这个词"。
- 补齐两处此前只有手工验证、没有测试的守卫：`trainer.main(["--budget-tier","48h"])`
  必须因缺 `--arm` 而退出（默认臂会写在跑臂自己的文件上），以及 `stage_integrity`
  的完整/截断/空题表/缺失四情形。两处都用 `monkeypatch` 改 `PROJECT_ROOT`，
  不再直接赋值——**同一个跨模块缓存泄漏**（§1 第 1 条）不能在这儿重犯。

§2 的 A1–A3 条目**保留不删**（§4 约定），以本节为处置记录。
合同测试 **61 passed**；ruff/black 全仓干净；P3a 工作表已按新表头重新生成。

## §1c 追加（本地 02:56）：判据延伸到"比对型脚本"

同一套判据在本轮咬到了我自己的诊断脚本：对照臂阶段 1 的确定性复评第一次比对用了
`item.get("correct")` / `.get("machine_correct")`，而报告真正的字段是 `score` ⇒ 两键皆缺、
默认 `None`、100 题全列空白，脚本却**退出码 0** 并打印"0 处变化"——一个把"读不到"
渲染成"完全一致"的假绿。抓住它的是逐题矩阵整列 `None`，不是任何断言。

⇒ **判据补一条**：凡"比较两份产物是否一致"的代码（测试、诊断脚本、结项工具），
取字段必须**失败关闭**（`row["score"]`，或显式列出必需键并对缺键报错），
不得用 `.get(key)` 的默认值——默认值会把"缺失"与"相等"合并成同一个状态。
`scores()` 一类辅助函数若被 `.get()` 兜底，其覆盖率为 0 而看起来是 100%。

## §1d 追加（本地 03:15）：两条不属于"测试缺陷"的过程事实

1. **在受审文件里改一个变量名就会把 marker 弄断。** 修 B3 时把局部量 `block` 改名 `branch`，
   `test_current_review_surface_is_complete` 立刻红——清单的 J8 是**按字面行**匹配的
   （`assert GOAL_REASON not in block`）。处置是把名字改回，**不是**改清单里的 marker：
   marker 的作用是"这段被审过的代码还在原位"的漂移提示，自证式地把 marker 跟着改成新写法，
   等于让清单永远追不上代码；而那会让已封存报告的 `judgement_sites` 与源码脱节。
   ⇒ 记录一条约束：**凡出现在 JUDGEMENT_SITES marker 里的标识符都不可重命名**，
   除非同时重跑清单并接受产物换代。本轮不做后者。
2. **等待器在一条臂停机后就监视不了另一条臂。** 实测：`--stages 2` 秒回
   `reason="condition"`，因为判定用的是 `any(stops.values())` —— 对照臂一停，
   "某臂写出停止结论"从此恒成立。已把判定抽成 `decide(reports, target, watch)` 并加
   `--arm`：未选中的臂不得触发 condition/stall，选中一个不存在的臂 ⇒ `arm_missing`
   且退出码 1（宁可报错也不静默等满 6 小时）。两支新测试把**两个方向**都钉住
   （含"另一臂成孤儿不算我这臂的 stall"）。

## §2 未修，按价值排序（后续批次；每条都给了"正确不变量"）

| 优先 | 位置 | 缺陷 | 应断言的东西 |
|---|---|---|---|
| A1 | `test_p3b_review_worksheet_contract.py:98-103` | "表格不打分"的测试**挡不住预填分数**：导出器会逐项打印 `机检预判：<verdict>`，而禁用词表漏掉真实取值 `precheck_pass/precheck_fail/hard_safety_risk`；末行断言的是导出器无条件打印的表头 | 正向断言"评分位为空"（对 `**评分（0/1/2）**：` 之后的内容做正则），并把 `_item` 的 verdict 参数化过那四个真实取值 |
| A2 | 同文件 `:109` | `assert "待复核维度" in text` 对任意输入恒真（表头无条件输出） | 导出器打印**每维待复核题数**，测试断言 0/0 与非 0 两种情形 |
| A3 | 同文件 `:112-116` | 名叫"绝不指向封存报告"，实际只做一次子串禁令，从未与任何封存路径比较 | 与 P3a 基线路径及封存产物文件名集合**实际求交为空** |
| B1 ✅**已修（02:52）** | `test_cap0_baseline_contract.py` 原 `:84-94` | **整支测试空转**：唯一断言藏在 `score != 1 → continue` 之后，而干净基线报告里 `score==1` 的行数为 **0** ⇒ 永不执行，判分器怎么改都不会红 | 普查给的修法（"对污染报告重放那 14 行"）**照做不可行**：受污染首版早于 `verdict_text` / `raw_last_output` 字段，报告里没有可重放的回答文本。改为驱动**当前**判分管线：用冻结清单的提问构造"只回显提问"的回答，断言 `_strip_prompt_echo` 确实去回显、`_score_closed` 拒绝给 1 分，并**统计不剥离时命中几行**（`naive_hits > 0`）——否则哪天构造失效，这条钉又会悄悄变成空跑。同时保留"污染报告里恰有 14 行 score==1"作为种群断言 |
| B2 ✅**已修（02:41）** | `test_b0_structure_space_contract.py` 原 `:457-461` | 11 格算 `expected` 只对 3 格断言（漏 else）⇒ **77 条种子扫描行（7 偏移 × 11 格）里 56 条无约束**。（普查报告写的"33 条"也是错的，实测 77 条。） | 改为逐格 `SWEEP_EXPECTED_GAIN` 全表断言 + "扫描格集合 == 表键集合"（新增/缺失一格都会红）+ `positive` 由符号**推导**而非采信字段。<br>**普查给的修法本身是错的**：它建议 `else: assert gain == 0.0`，而 `patch__*` 三格实测为 **−2.000**（正是路线 B 冻结版声明的反例面）——照它写会把测试跑红才发现。两份封存报告（扩面 / 落地后）逐格一致，故该表可安全钉死 |
| B3 ✅**已修（03:13）** | `test_b0_n2_stop_reason_semantics_contract.py:118-126` | 名叫"源码级钉住"，四条断言全部 grep **同一支仪器里定义的字符串常量**，从不打开真正落地的 gate 源码 | `assert counterfactual.M4_SELECTION in inspect.getsource(frozen_gate._member_episode)`，再把三条性质查在那段真实源码上。<br>落地方式：新增 `_blocked_branch(source)` 从**已发布**源码切出 `if chosen is None:` 到该 `return finish(...)` 的片段，三个标记缺任一即 `assert` 失败（不静默返回空串）。<br>**过程记录**：改完 `test_current_review_surface_is_complete` 立刻变红——清单把 J8 的 marker 钉在字面行 `assert GOAL_REASON not in block`，我把局部量改名 `branch` 就把它断了 ⇒ 变量改回 `block`（marker 只是漂移提示，见 §1d） |
| B4 ⏸**本轮不做** | `test_cap0_legacy_load_contract.py:37-41` | `source_edited` 是探针**初始化的字面 False**，无人重算 ⇒ 断言的是声明不是测量 | 让探针实测（对 `taiji/` 取 `git status --porcelain` 或哈希前后差）后再断言该差为空。**顺延理由**：要改 `probe_taiji_cap0_legacy_load.py`，与 DEBT-I5 同属评测面批次（双臂在跑） |
| B5 ⏸**本轮不做** | `test_cap0_inventory_contract.py:99-103` | 三条"格式支持"断言查的是 runner 自己拼的散文（句子含句子），不碰 `taiji/model.py`；`"2726-2732"` 是会腐烂的行号化石（已登记 DEBT-I5） | `assert {"taiji-native-v8","taiji-native-v9"} <= set(Taiji.LEGACY_CHECKPOINT_FORMATS)`；行号断言换成已被行为测量的那条错误文本。**顺延理由**：与 CAP-0 加载器决策简报 §5 同批（本轮不动 `taiji/`） |
| C1 ⏸**本轮不做** | `test_cap0_baseline_contract.py:458-463` | "退出码合同"断言的是 `return` 的一种拼写；`main()` 从未被调用 ⇒ 判分反向也绿 | 直接调 `main([...])`：基线自身→1，改进候选→0。**顺延理由**：`main()` 一次要 200+ s 全新评测并与在跑双臂争核（实测 273.7 → ~180 符号/秒），且要动评测面 |
| C2 ⏸**本轮不做** | `test_b0_rule_revision_seal_contract.py:177-178,193` / `test_b0_m1_counterfactual_contract.py:106,138,140,142` | 由同一行/同组字段**算术蕴含**的断言（`frozen_attribute_unchanged` 就是 `is not` 的复述；行数等于 `a-(a-b)`；同一对象比自身源码） | 断言置换的 `status` 列表、`M4_SELECTION in getsource(...)`（B3 已按此式落地，可复用）、以及"生成的臂 ≠ 封存文本"。**顺延理由**：需逐字段重推 counterfactual delta 哪些是测得、哪些是算出，与主线无关，留 §2 |
| C3 ✅**已修（03:07）** | `test_p3b_arm_corpus_contract.py:95,99,213-216` | `qualifies(..., "all")` 对任何输入都是 True ⇒ 两行零信息；`密度 > 2×对照密度` 由 `==1.0` 与 `<0.5` 两行蕴含 | 删去两行恒真断言（**理由写进 docstring**，不是静默删除），改为：过滤落点的显式期望表（元数据角色不算说话人）+ **嵌套性**（treatment 接受的每行 control 必接受）+ 一条防退行守卫（样本集若两规则重合则嵌套断言无意义 ⇒ 红）。密度两行换成独立区间 `0.15 < x < 0.30`（实测 0.2356） |
| C4 ✅**已修（03:08）** | `test_p3b_campaign_contract.py:387-388` 一类 | 单段 fixture 的 pass-through 读取（已在 §1 第 5 条处理一处，其余同类待扫） | 一律两段以上，令"取第一个"会红。落地比建议更强：两臂阶段列表**各自乱序**、共同 tick 排在其中一臂的**第二位**、两个对照阶段彼此分数不同 ⇒ 任何"按位置 zip"的实现会算出错位的 delta 而不是恰好相同；另钉"链未核验的阶段仍报差值、但 `chain_ok=False"`。<br>**过程记录**：我第一版期望写错（19M 只有一臂有阶段 ⇒ 差值是 `None` 而非"仍报"），跑测试前自查发现并改掉 |

## §3 结构性局限（一条，但笼罩全表）

`test_cap0_*_contract.py` 全部、以及 `test_b0_n2_stop_reason_semantics_contract.py` /
`test_b0_structure_space_contract.py` 中"读报告"的那一半，**只消费已提交 JSON、从不重跑探针**
（其 docstring 自己写明）。⇒ 它们**不可能因代码改动而红**，只会因产物被编辑而红。
这是有意的（探针每支 9–15 s 全新进程），但后果必须说清：**这批文件里能抓住仪器行为回归的，
只有直接调用函数的那几支**（P3b 三支、`test_b0_n2_stop_reason_disposition_contract.py`、
两个 B0 文件里以 `probe`/`counterfactual` 为 fixture 的测试）。

**若要一次结构性升级而非 20 处点修**：让 CAP-0 那组像 N2 那样**当场重导**
（in-process 复算 `model_reality` / `output_summary`，或断言 `current == report`），
把"读封存"与"复现封存"分成两类明确的测试，而不是同一支测试里混着两种语气。
本项与 DEBT-I5/I6 同属"评测面脚本"改动，须等双臂 campaign 结束后再做。

## §4 本文件的维护约定

修一条就在 §1 追加一行（含提交号），未修的留在 §2 且**不得删条目**；
若某条经复核其实是恒真误报，移到 §1 并写明"误报及理由"，不要静默清除。
