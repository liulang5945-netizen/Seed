# PLAN-A-29：出厂基座的体积与"从没写过的 75 MB 空仓"（owner 裁定 (i) 的第一步）

日期：2026-09-28。归属：A 支线 → **产品出厂件**（`PLAN-A-28` §6 裁定 (i) 的执行前置）。
状态：**乙档已实现并有守卫（§7）——尺寸 87.9 MB → 12.3 MB，但主-2 的 ≤10 MB 这条线没够到（§8）**；
本件同时保留了**设计预登记**（§2 三条路、§3 撞面清单、§4 判据）供复核。

---

## 0. 为什么会走到这里

`PLAN-A-28` 量到：把已训复制回路装进产品信封，落盘从 **4.14 MB → 87.9 MB**，而**回路本身只 0.18 MB**。
owner 裁 (i)＝**"先压体积，再换出厂基座"**（不接受直接吃 +84 MB）。⇒ 本件回答"那 84 MB 是什么、
怎么压、压它会撞到谁"。

## 1. 实测分项（只读，零训练）

对 `Seed.from_checkpoint(checkpoints/seed_beta.pt).checkpoint()` 递归遍历每一个张量：

| 读数 | 值 |
|---|---|
| 张量字节合计 | **87.4 MB** |
| 其中**整张全零**的字节 | **80.2 MB（91.8%）** |
| 非零张量条数 | 194 |
| 最大两块（各自全零） | `.taiji.kernel.identity_organ.value_keys` **37.75 MB** ＋ `.substrate.…value_keys` **37.75 MB**（v10 镜像存两份） |

⇒ 体积问题的**九成以上是那 75.5 MB 的两份空路由键仓**——`identity_organ.py:127` 在构造时
就把 `value_keys` 按 `[slots × capacity]` **稠密预分配成零**，而本基底从未写过它
（零面普查早已记过同一件事："一张从未被写过的路由键仓，9.4M 个数"）。
`to_payload()`（`identity_organ.py:580`）把**整张稠密缓冲**原样落盘，尽管 payload 里
**已经有 `value_counts`**——即"每槽只前 `count` 行有意义"这件事本来就被存进档了，
存的时候却把 37.75 MB 的零一起背上。

## 2. 三条可选路（按能声称的上限排，不按省事排）

**甲：懒分配（上限最高，因为它同时修运行时）**
`value_keys`／`value_actions` 不在构造时开稠密零仓，**首次写入才按需要的容量开**。
收益：出厂件不再背 75.5 MB；**每次 load 也不再在内存里开这 75.5 MB×2**
（本仓所有取数脚本都被它拖慢/拖大）。
代价与风险：路由器的读路径（`entries = self._value_keys[slot, :count]` 等 10+ 处）都要能处理
"未分配＝空"，且 `capacity` 增长路径（`identity_organ.py:217-236`）要重写 ⇒ **动的机制面最大**。

**乙：按 `value_counts` 截断存盘（改动最小、收益最大的一块）**
`to_payload()` 只存每槽用到的前缀（或整表截到 `max(count)`），并落一个
`value_layout: "truncated"` 之类的标记；`load_payload()` 按 `capacity` 把零补回去。
收益：87.9 MB → 约 **7 MB 量级**（非零字节只有 7.2 MB）。
风险：`content_digest(to_payload(...))` 会变 ⇒ 见 §3 的影响面；**必须**带"旧档（无标记＝稠密）
仍可载"的兼容分支。

**丙：通用零张量省略（任何面都省，但语义最糊）**
在信封层统一"全零张量不落盘，靠形状索引还原"。
收益：80.2 MB 全免；**代价**： digest／"存盘全零≠运行时全零"这类**已有账**会把"没存"与"存了零"
混为一谈——本仓已经踩过一次（记忆里那条 P1 待查），**不推荐**在没有身份形状索引前做。

**建议顺序**：先 **乙**（一次改一处、收益占九成、还原规则明确），再评估 **甲**（是否值得动运行时）。
**丙不在本件范围**。

## 3. 会撞到谁（动手前必须逐条读完，这就是"第二次实现前要先量影响面"）

`to_payload()` 的字节被**内容摘要**消费，以下每条都会因"截断"而变红或需要重新钉：

* `tests/taiji_native/test_m1_identity_organ_canary.py:102-103`：两份 payload 的 `content_digest` **相等**断言；`:262-263`：**不等**断言（受罚 vs 受奖）——截断后只要两边同形仍成立，但"只改了未用区"的改动将不再改变摘要（语义上更对，但要写进注释钉住）；
* `tests/taiji_native/test_foundation_tasks.py:186` `test_persistent_digest_responds_to_identity_organ_mutation`：**持久摘要必须随器官改动而变**——本条是这次改动的主要撞点，截断后"新增一条写入"仍会变（前缀变长），但**要实测**；
* `tests/taiji_native/test_receptor_factorization_contract.py:97-103`："器官拓扑逐位相同、只允许 `parent_checkpoint_digest` 不同"——它比的是 `_digest(payload)`，截断标记必须对两边**一致**才会仍相等；
* `tests/taiji_native/test_cap0_legacy_organ_migration.py:68-77`：器官 payload 的形状校验（塞进非法类型／改 lineage 必须响亮失败）——新增键要进它的校验面；
* **已封存读数**：任何把器官摘要钉进件里的对比（CAP-0 普查／`pinned-digest` 一类的窄口径钉子）——
  按本仓既有教训〔参数面加新张量的三件配套事〕，**"默认不改行为"必须拿真实链路的摘要来证**，
  不能靠推理；
* 产品侧 `Taiji.restore` 的 **lineage 守卫**（`identity organ checkpoint lineage does not match Taiji core`
  那条已知的血缘校验）——还原后的 payload 必须与保存前**等价**，否则带器官的档全部拒载。

⇒ 因此本件的**第一个执行动作不是改码**，而是：**跑一次"器官 payload 摘要的消费面"清点**
（把上面每条实测出改动前后的摘要值，并列成表），有红才决定是"重新钉＋兼容旧档"还是退回甲。

## 4. 判据（先冻）

| # | 判据 | 线 |
|---|---|---|
| 主-1 | 改后 `save → restore → save` 的**逐位相同**（含被截断的 `value_keys` 还原成同形状同零） | 一位不差，否则整个方案作废 |
| 主-2 | 产品信封落盘尺寸 | **≤ 10 MB**（现 87.9 MB；非零字节实测 7.2 MB） |
| 主-3 | 受检基底只读：跑前后 `checkpoints/*.pt` sha 不变 | 硬约束（本仓已两次踩覆盖产品件） |
| 守卫-A | **旧档兼容**：无截断标记的 v10 档（含 `checkpoints/seed_beta.pt` 与 dist 里的产品件）必须照旧载入，读数逐位不变 | 失败即不合并 |
| 守卫-B | 挂电路的产品信封（`PLAN-A-28` §4 那三档）在改后**命中/成句/可解码三列逐位不变** | 否则"压体积"改变了行为＝方案错 |
| 守卫-C | §3 列出的每条摘要消费面：要么仍绿，要么**显式重新钉并把新值写进件里**（不许静默变红） | 逐条列名 |
| 报 | 改后**每次 load 的常驻内存**（甲才测；乙不改变运行时分配） | 只报不判 |

## 5. 明确不做

* 不改产品默认装配、不换 `DEFAULT_CHECKPOINT`（那是裁定 (i) 的**第二步**，还欠 `PLAN-A-28` §8 的
  表层链复读处置与第二次独立取数）；
* 不动信封版本号以外的语义（`taiji-native-v10` → 需要时升 `v11` 并把 v10 进 `LEGACY_CHECKPOINT_FORMATS`，
  这条路径本身要单独批）；
* 不碰 `PLAN-A-28` §4b 那条复读症状（另案，属发射侧/器官计划打架）；
* 不做 §2 的 **丙**（通用零省略）。

## 6. 起跑点（执行史，四条全部已完成，见 §7）

1. ✅ **清点 §3 那张消费面**：`test_m1_identity_organ_canary／test_foundation_tasks／
   test_receptor_factorization_contract／test_cap0_legacy_organ_migration／test_m1_66b_value_router`
   一起跑＝**40 passed／0 failed** ⇒ 那是一条**活着的、当前为绿**的面，
   改后任何一条变红都是真信号（"改前本来就红"这种推脱不成立）；该 40 条即改前**固定对照面**。
2. ✅ 实现 **乙**（截断存盘＋还原补零＋两条兼容分支＋三条响亮拒绝），带 §4 守卫；
3. ✅ 取改后尺寸并核对**回装无损**（260 张量逐位）；⚠️ **守卫-B（`PLAN-A-28` §4 三档读数复跑）尚未取**
   ——那是一趟约 20 分钟的整跑，本件不用"看起来不会有影响"顶替实测，见 §8 第 3 条；
4. ✅ 顺带查出并修掉 `load_payload` 的那条**假兼容路**（见 §7 末段）。

## 7. 落地记录（2026-09-28：乙档已实现，尺寸 −86%，但主-2 这条线我没够到）

**改点**（`taiji/identity_organ.py`，两处，共 66 增 7 删）：`to_payload` 按已有的 `value_counts`
只存前 `used = max(counts)` 行并落 `value_router_used`；`load_payload` 见到标记就按稠密形状补回空槽
（键 0／动作 −1），且带三条**响亮拒绝**（counts 超过存档行数／自称的行数与 counts 不自洽／
形状对不上）。不变量一旦破了（存档区外却有值）⇒ **整表存盘**，宁可大也不丢。

**实测尺寸**（同一台机器、同一受检基底，`torch.save` 后的文件字节）：

| 件 | 改前 | 改后 |
|---|---|---|
| 产品基底重存（v10 信封） | 87.5 MB | **11.9 MB** |
| **带复制回路的产品信封**（`PLAN-A-28` §5 那个数） | 87.9 MB | **12.3 MB** |
| 相对今日出厂件（`seed_beta.pt` 落盘 4.14 MB）的净增 | +83.8 MB | **+8.2 MB** |

**回装无损**：260 个张量 `save → restore → save` **逐位 0 处不符**；还原后
`_value_keys` 运行时形状仍是完整的 `(128, 64, 1152)` 且全零（截断只发生在存档侧，运行时不动）。

**一条必须留下的陷阱记录**：第一版我写的是 `keys[:, :used]`，形状对、`numel()` 对，
但**切片是视图、底层 storage 仍是整张稠密缓冲**，而 `torch.save` 序列化的是 **storage** ⇒
实测 `numel()==0` 的张量仍写出 **37.75 MB**，"截断"一位字节都没省下来。
修法＝`clone()`。守卫第一条直接钉 `untyped_storage().nbytes()`，这条是**先红后绿**（不 clone 时实测红）。

**主-2 这条线我没够到（≤10 MB，实测 12.3 MB），原因如实**：截掉 75.5 MB 的空路由仓之后，
剩下的 11.8 MB 张量里仍有 **4.7 MB 是散落在多个小结构上的全零**（`bank.prototypes` 0.59 MB×2、
`memory.cortical_readout.edge_weight` 0.44 MB×2 …），**没有第二块"结构性空仓"可以再按计数截**；
要再降只能走 §2 的 **丙（通用零省略）**，而丙是本件预注册时判为"不做"的那一档（语义会把"没存"与
"存了零"混起来）。⇒ 这是一次**判据与手段的边界碰撞**，不是一句"再努力一点"：
要么 owner 授权丙（约可再降到 7.4 MB），要么接受 +8.2 MB 这个净增。**本件不自行挪线。**

**顺手修掉的第二条真实缺陷**：`load_payload` 里 `self._value_counts = restored_counts.clone()`
曾写在 `if "value_counts" in payload:` **外面**，而 675-677 行的注释声称"旧档没有路由三件就空载"——
实测那条兼容路直接 `UnboundLocalError: cannot access local variable 'restored_counts'`
⇒ 注释说的路从来没通过。守卫第 5 条是它的**阳性对照**（改前红、改后绿）。

**门禁（跑绿才算落地）**：

| 面 | 命令/范围 | 结果 |
|---|---|---|
| 新守卫 | `test_identity_organ_router_truncation.py`（8 条） | 8 passed |
| 器官摘要消费面（§3 那张） | 5 文件 | **40 passed／0 failed**——**一条都不用重新钉** |
| 广面 | `tests/taiji_native tests/seed` 全量 | **1793 passed／4 failed**（1069 s） |
| 4 条红的归因 | 逐条 | ①②`test_cap0_*_contract::reproduces_the_sealed_one`＝本地 `checkpoints/` 多出他人文件（`PLAN-A-26` §6.5 已登记的既有环境红）；③`test_platform_boundary::source_face_is_the_git_face`＝`taiji-harness/.dsh-sbx2/lock-audit/*.py`（他会话在飞的跟踪文件，遍历看不见）；④`test_project_identity::resolvable_links`＝`08_UPSTREAM_SYNC_PLAYBOOK.md` 里一条指向不存在的 harness 笔记（他人文件的坏链）。**②另做了停用复跑**：把我的 `identity_organ.py` 整档退回 HEAD 复跑⇒**照样红**⇒ 与本次改动无关 |
| 主-3 受检基底只读 | `sha256sum -c`（`seed_beta.pt`／`seed_corpus.pt`／`dist` 内产品件） | 全 **OK**（跑完 1793 条测试后逐位不变） |
| lint | `ruff check`／`black --check`（identity_organ.py＋新测试文件） | 全净（`identity_organ.py` 在 HEAD 上本就是 black 净的，本次未引入脏） |

## 8. 现在欠的两件事（都不该被本件的数字盖掉）

1. **主-2 未达**（12.3 MB vs ≤10 MB）⇒ 要 owner 在"授权丙"与"接受 +8.2 MB 净增"之间选一个；
2. **运行时那 75.5 MB 的常驻分配没动**（乙只管存档侧）——那正是 §2 的 **甲**，
   它同时还能让每次 load 少开两份 37.75 MB 的零表（本仓所有取数脚本都被它拖大）。
3. **守卫-B 未取数**：`PLAN-A-28` §4 那三档（28/104、成句 7、切尾可解码 0.5256）在本次改动后
   **没有复跑**。理论上截断只发生在存档侧、运行时逐位相同 ⇒ 读数不该动，
   但"读数不该动"必须由读数证明（本仓教训：门的绿不许从聚合入口继承，也不从推理继承）。

