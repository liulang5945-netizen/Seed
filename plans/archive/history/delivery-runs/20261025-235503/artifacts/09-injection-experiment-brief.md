# M5/skipped 注入实验 brief（R-S6a 实战受控验证）

状态：草稿 v1（待 architecture-reviewer 复核；复核通过后由主理人亲自执行）
放行依据：board 决策追记 3（R-P1/R-P2 复核通过，注入实验正式放行）+ 追记 4（实战 ≠ 注入实验，R4 开口维持）
基线修订：本 brief 将基线锚从 40258536618 修订为 **40288176554**（571431884，D-01 修复后干净基线）——
40258536618 的红位混着 pytest-full/pytest-windows failure 与 collection-crlf D-01 两族，非单点；
40288176554 预期红位仅 N1-167 漂浮 timeout（若在），gate-summary 逐行 outcome 形态即注入实验的对照形态。

## 一、目的

证明 `Summarize CI gate outcomes`（id=gate-summary / gate-summary-windows）的判定路径对**单点注入**的
非 success outcome 能否 100% 抓红。40258536618 的实战抓红（pytest-full/pytest-windows outcome=failure
→ 两腿 gate-summary 全部 RED）是附带实证，变量不单一（同 run 混 D-01 探针红）；本实验补受控证据，
failure 与 skipped 两通道各一次。

## 二、实验设计

**E-fail（failure 通道）**：Ubuntu 腿 `coverage-gate` step（ci.yml ~536 行）注入必然失败。
`run` 从单行 `>-` 改为块：

```yaml
        run: |
          false  # M5-inject: forced failure (E-fail), revert after readout
          python scripts/training/verify_core_reasoning_coverage.py --from-report coverage.json --threshold 60.0 --out reports/core_reasoning_coverage_ci.json
```

GitHub Actions run 默认 `bash -e`，首行 `false` ⇒ step outcome=failure，覆盖率逻辑不执行。
其余四门禁（deselect-guard / pytest-full / junit-faces 及 Windows 腿两项）保持 success。

**E-skip（skipped 通道）**：Ubuntu 腿 `deselect-guard` step（~482 行）注入强制跳过：

```yaml
        if: false  # M5-inject: forced skip (E-skip), revert after readout
```

单点性已核：`DSH_CI_DESELECT` 定义在 job 级 env（115 行），不依赖 deselect-guard 的 output，
skip 不引发 pytest-full 第二红。预期 `steps.deselect-guard.outcome == 'skipped'` → gate-summary 判红。

**纪律**：
- 每个注入独立 commit（单文件 pathspec `.github/workflows/ci.yml`）+ 独立 push，读数落账后**立即 revert**
  （revert 也单独 commit + push）。串行执行：E-fail → 读数 → revert → E-skip → 读数 → revert，两注入永不同 run。
- **不碰冻结注释区**：481–492（deselect-guard R-S6a 注释块）、578–585（gate-summary 机制注释①–④）、
  962–964（Windows 腿注释）等冻结区零字节改动；注入只动 run 行/if 行。
- 注入 commit 信息前缀 `M5-inject:`，revert commit 信息前缀 `M5-revert:`，board 逐条登记。
- **范围声明（AR 补强）**：Windows 腿 gate-summary-windows **不单独注入**——其 run 块与 Ubuntu
  gate-summary 的 check()/case 表逐字同构（bc019e517 已核），判定路径由 Ubuntu 两实验覆盖，
  防后续读数争议。

## 三、验收读数（三要素）

每个注入 run 逐项登记：
1. **gate-summary 表逐行**：step / outcome / verdict——E-fail 预期 coverage-gate 行
   `failure / RED` 其余 4 行 `success / GREEN`；E-skip 预期 deselect-guard 行
   `skipped / RED`（本实验的核心新证据：skipped 通道）其余 4 行 GREEN。
2. **汇总步**：`::error::` 文案输出 + exit 1（表打 stdout 与 GITHUB_STEP_SUMMARY 双面）。
3. **job conclusion**：`failure`；coe 未吞红（五门禁 conclusion 可能显示 success，以 outcome 与
   gate 表为准——API quirk 已落账决策 25）；后续 step 零意外跳过（upload 等带 `if: always()` 的除外）。

**对照形态与硬性 gate（AR 补强升格）**：基线 run 40288176554 两腿 gate-summary 逐行 outcome 先行登记
（唯一允许红位 = pytest-windows N1-167 漂浮）。**基线自检是硬性 gate：40288176554 实际结论形态若与
预期不符（尤其探针未回绿 count>0），注入实验中止**——否则对照形态失效、注入红位无法单点归因。
登记格式：基线两腿逐行表 + 探针 count 值，作为注入 run 的 diff 基准；注入 run 红位必须且只能在注入行，
否则实验判失败并如实落账。

## 四、成本与风险披露

- 4 个额外主 CI run（2 注入 + 2 revert），每个 ~2h；放行条款已含此成本。
- 注入 push 连带触发 fast lint / m7-ci（各自 concurrency，不受 gate-summary 管），不污染读数。
- D1（CI 全绿）观察窗口排除 4 个实验 run；最终 revert 后的 run 才是 D1 读数对象。
- 注入期间 main 上存在短窗红 run（~2h/个）：board 已预告，属实验语义红，不触发回归排查。

## 五、执行后

- 读数落 `08-test.md` §16（E-fail / E-skip 各一节）+ board 决策 27；
- §16 须包含「本实验证明的外部契约」清单，首两条：① coe 吞 conclusion 不吞 outcome
  （failure 通道，40258536618 实战 + E-fail 受控双证）；② **coe:true 与 if:false 并存时 outcome
  仍为 skipped、不被 coe 改写**（E-skip 专属，AR 指认的 skipped 通道真正外部契约点）；
- **cancelled/空值两分支的覆盖声明（AR 条件 1/2）**：由本地演练 `gate_summary_drill.sh` 覆盖——
  六场景（全 success / failure / skipped / cancelled / 空值 / 全失败）**6/6 ALL-PASS**，
  主理人 2026-10-26 独立重跑确认（AR 申报核实物一致）。数据点分布：CI 实证 failure + skipped
  （外部上报值假设），本地实证 cancelled + 空值（判定逻辑分支）——四分支全覆盖、零额外 CI 成本。
  cancelled/空值不做 CI 变体的接受理由：判定输入值无法在不改 gate-summary 代码的前提下 CI 单独注入；
  且 job 被 cancel 时 cancel 传播可能让 gate-summary（if: always()）自身来不及执行，CI 上测
  cancelled 本身是噪声实验，本地演练才是该分支的正确验证位；
- R-S6a 从 §8 R4「未交付」转「已交付」（AR 终审后）；
- revert 完成后 main 树内容与 571431884 等价（`git diff 571431884 HEAD` 应为空）。

## 六、AR 复核意见（2026-10-26，终版放行）

v1 通过，附 2 条件 + 2 补强（均文本修订，不改实验设计），修订后即终版：① E-skip 单点性论证完备
（job 级 env / guard 零 output 消费者 / `if: always()` 覆盖三处独立核验）；② cancelled/空值变体同意
不加（条件 1 演练读数登记 + 条件 2 噪声实验论证）；③ 基线自检升格硬性 gate + Windows 腿范围声明。
全文见上文对应小节（已合并），复核原文存 04-architecture.md 附录 B §B.8。

## 六、AR 复核意见（2026-10-26，architecture-reviewer）

**① E-skip 单点性 ✅ 论证完备，无遗漏**。三处事实独立核验：
- `DSH_CI_DESELECT` 确为 **job 级内联 env**（ci.yml:114–193 Ubuntu / :740–753 Windows，非任何 step 的 output），
  deselect-guard skip 不影响 env 可见性；
- `steps.deselect-guard.outputs` **零引用**——guard 无任何 output 消费者，skip 无语义链下游；
- upload/trend/gate-summary 全部 `if: always()`（:462/:537/:547/:562/:574），E-fail 中 coverage-gate 失败后
  照跑，受影响 step 唯一。
另补一个 brief 未写但应登记的点：**E-skip 同时实证「coe:true 与 if:false 并存时 outcome 仍为 skipped、
不被 coe 改写」**——这才是 skipped 通道的真正外部契约点（skipped 的 outcome 字符串值 + check() 的 skipped
分支），建议写进 §16 读数登记的「本实验证明的外部契约」清单。

**② 验收判据 ✅ 同意不加 cancelled/空值 CI 变体，但附两个条件**：
- **条件 1（登记补全）**：check() 四分支的判定逻辑完备性由**既有本地演练** `gate_summary_drill.sh`（6/6，
  含 cancelled 与空值两个变体）覆盖——把该演练读数登记为 08-test.md §16 的组成部分。这样数据点分布为：
  CI 实证 failure + skipped（两个「外部上报值」假设），本地实证 cancelled + 空值（判定逻辑分支），
  四分支全覆盖、零额外 CI 成本。
- **条件 2（接受声明）**：cancelled 的上报字符串（`cancelled` 双 l）与空值（`steps.<不存在id>.outcome` ⇒ 空串）
  作为 Actions 文档级行为接受，理由有二：判定输入值无法在不改变 gate-summary 代码的前提下于 CI 单独注入；
  且 **job 被 cancel 时 cancel 传播可能让 gate-summary（if: always()）自身来不及执行——CI 上测 cancelled
  本身是噪声实验**，本地演练才是该分支的正确验证位。

**③ 两处补强（不改实验设计，改 brief 文本）**：
- **基线自检 gate**：§三「对照形态」应升格为硬性 gate——**40288176554 的实际结论形态若与预期不符
  （尤其探针未回绿，即 count>0），注入实验中止**，否则对照形态失效、注入红位无法单点归因。
  登记格式：基线两腿 gate-summary 逐行表 + 探针 count 值，作为注入 run 的 diff 基准。
- **范围声明**：Windows 腿 gate-summary-windows 不单独注入——其 run 块与 Ubuntu gate-summary 的
  check()/case 表**逐字同构**（bc019e517 已核），判定路径由 Ubuntu 两实验覆盖；请在 §二纪律节写明，
  防后续读数争议。

其余项（冻结区行号、revert 后 `git diff 571431884 HEAD` 为空、D1 观察窗口排除实验 run、注入期短窗红不触发
回归排查）均核验通过。**复核结论：v1 通过（附条件 1/2 + 补强 ③ 两处文本），修订后即为终版，可执行。**
