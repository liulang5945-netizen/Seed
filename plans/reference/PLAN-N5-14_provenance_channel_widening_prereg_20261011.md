# PLAN-N5-14 情节场来源编码加宽到 `generated` 的通道重映射与验收（DEBT-G104／G100 的张量层半；判据先冻，2026-10-11）

批准：owner 2026-10-11 弹窗裁 **G104＝批：接受既有 `.pt` 检查点拒载**（同批四题里的第三题）。
本件是那条批文的**前置冻结件**：批次 B（动 `taiji/`）之前先把"加宽会连带改什么、什么必须能为假、
哪些历史读数因此不可复算"写成可评审、可机检的形式。

零算力：不加载检查点、不训练、不更新权重。本件落盘时**不动产品码**。

## §0 结的是哪两条，以及为什么不能"顺手加一枚枚举"

* `EpisodicField.PROVENANCE_KINDS` 现读（`taiji/memory.py:65`）＝`("experienced", "imagined", "replayed", "external")`，
  四枚；而 VISION §19.6「防止自证学习」要求未经核验的生成内容**保留 `generated` 来源**、不混进事实与奖励。
* ㊵-682／㊵-683 已经把**记录层**做完：`EpisodicMemoryRecord` 接受 `generated` 标签，且带非零
  `outcome.reward` 在构造期 `ValueError`。⇒ 禁令今天有可判形式，但只在记录层。
* 张量层加宽**不是加一个字符串**：本仓有三处宽度从这枚枚举**导出**（§1），加宽会让既有 `.pt` 的
  权重形状与受体通道图同时对不上（§2）。因此"加宽"是一次**架构迁移**，需要验收式，而不是拼写改动。

## §1 三处派生宽度（现读；每处点名文件与条件，不写行号——行号会随本文件之外的提交漂移）

| # | 位置（取法＝符号＋条件） | 今天的形状 | 加宽后的必然变化 |
|---|---|---|---|
| 1 | `taiji/memory.py` 的 `EpisodicField.__init__` 里 `self.provenance_encoder = SparseSynapses(...)` | 入参两处使用 `len(self.PROVENANCE_KINDS)`（`in_features` 与 `fan_in`／`row_fan_in` 同源） | 三枚宽度一起 +1；`developmental_synapse.py` 的构造期校验要求 `pre_index.shape == (out_features, row_fan_in)` ⇒ 旧权重矩阵形状直接不匹配 |
| 2 | `taiji/memory.py` 的 `self.provenance_readout = self._blank_readout(len(self.PROVENANCE_KINDS), ...)` | 读出层入参同样从枚数导出 | `out_features` 变化；one-hot 编码 `code = torch.zeros(len(self.PROVENANCE_KINDS))`（同文件）也要跟着变宽 |
| 3 | `taiji/organs.py` 的受体通道装配（`_payload_tensors`：`if bool((counts == 0).any()): raise ValueError("receptor payload leaves a context channel unconnected")`，以及 `load_payload` 的 `torch.equal(channel, self.channel)` 比对） | 通道图按读出头累计宽度生成；`counts == 0` 那一支是现成的响亮拒绝 | 新通道必须在通道图里有像；旧 payload 的 `channel` 矩阵与新架构**不相等** ⇒ 走 `load_payload` 即抛 `receptor channel map does not match architecture` |

同族先例（**复用形状，不重抄实现**）：`taiji/identity_organ.py` 里已有一段按
`source.row_fan_in != old_capacity or source.fan_in != old_capacity` 判定的容量迁移处理。

## §2 作废半径：2026-10-11 现取的五面（**逐面点名取法；跨面不相减**）

| 面 | 取法 | 枚数 | 加宽后的实际后果 |
|---|---|---|---|
| 全仓盘上 `.pt` | `find . -name '*.pt' -not -path './.git/*' \| wc -l`（Git Bash coreutils find） | **867** | 数字本身不是代价；见下面四行的分层 |
| 版本控制内 `.pt` | `git ls-files '*.pt' \| wc -l` | **0** | ⇒ 没有任何一枚入库件会因本笔"损坏"；代价落在**本地复算能力与装机可用性**，不是仓库完整性 |
| 默认装载面 | `find checkpoints -name '*.pt'`（13 枚，含 `seed_beta.pt`＝产品默认基座、`seed_beta_with_circuit.pt`、`seed_a31self_with_circuit.pt` 与 N2 母／候选／回滚档；`du -sh checkpoints` ＝154M） | **13** | **这一面才是"装机即坏"**：默认基座拒载 ⇒ 需要一次重建跑（批次 C，机时归 owner） |
| 历史实验件 | `find reports output outputs -name '*.pt'` 分层现读：reports 396／output 217／outputs 182 | **795** | 只在有人按原命令复算历史读数时受影响（含双臂逐张量比较件与 A-30／R2 期档）；全部 git 忽略 |
| 运行时自生物 | `tests` 25／`.seed_test_tmp` 31／`taiji-harness` 2／`artifacts` 1 | **59** | 重跑即再生，不计入作废面 |

**一条必须更正的已发表数字**：DEBT-G104 与债表 G100 行都写过「盘上共 **230** 枚 `.pt`」。
今天按上述任何一面都取不到 230（能命名的面给出 867／0／13／795／59），
而那句报价**没有点名取法** ⇒ 它整条降级为历史读数，不得再当"作废半径"引用；
owner 的批文是按"230 枚"接受的，所以本件把重取后的分层一并交回（§8 的问句），不替那条批文默认扩展。

## §3 判据（冻五条；每条写明"什么情况下它为假"，全部零新算力）

* **J-N5m-1 宽度取自枚举本身**：断言 `provenance_encoder` 的入参与 `provenance_readout` 的入参
  **都等于** `len(EpisodicField.PROVENANCE_KINDS)`（比较对象是同一个表达式，不是常量）。
  为假的情形＝任何一处把宽度写成字面量（写成 `4` 或 `5` 都算坏）——那会让"枚数变了但宽度没跟着变"
  这件事静默通过。
* **J-N5m-2 `generated` 真的进得了编码，且零权时中性**：
  正支＝同一线索下 `provenance="generated"` 与 `provenance="external"` 的编码输出**必须不同**；
  反支（W=0 回退）＝把新通道对应权重全部置零后，编码输出与**加宽前的四通道基线逐位相同**。
  为假的情形＝正支相等（说明新通道没接进去）或反支不相等（说明"零权"并不是中性档，
  那条等价声明是假的）。两支都要跑，缺反支时这条判据是恒真式。
* **J-N5m-3 旧件响亮拒载**：拿一枚四通道 `.pt` 去载，必须抛错且错误消息点名通道数（旧 4 对新 5），
  并**禁止**静默补零误载。
  为假的情形＝载入成功、或抛的是不含通道数的错。
  本条取代债表原 ① 那句「迁移后旧件仍可载」的验收——owner 已裁"接受拒载"，所以那条验收作废，
  取而代之的就是这一条。
* **J-N5m-4 受体通道图覆盖新通道**：新通道在 `readout_receptors` 的通道图里必须有像；
  且当有人构造一份"新通道 `counts == 0`"的 payload 时，`organs.py` 那枚
  `receptor payload leaves a context channel unconnected` 必须**被触发**（正反两支）。
  为假的情形＝漏掉新通道却仍载入成功（那正是该检查存在的理由）。
* **J-N5m-5 默认位不被本笔改动**：本笔**不得**改 `DEFAULT_CHECKPOINT`／`FACTORY_CHECKPOINT` 的指向；
  在批次 C 的重建跑完成之前，任何按默认位装载的仪器必须 `rc=2` 并点名是哪枚件拒载，
  **不许**把"载入失败"降级成"跳过"或"看起来绿"。
  为假的情形＝默认位被悄悄指向一枚新产的件（那等于用改默认换绿），或仪器在拒载时返回 0。

## §4 守卫（三条，各自能为假）

* **G-N5m-1 不重抄生成链**：新代码必须复用现成的 `SparseSynapses`／`_blank_readout`／受体装配路径；
  包里**不得**出现第二份 one-hot 来源编码实现（判定＝按文件搜 `torch.zeros(len(self.PROVENANCE_KINDS))`
  的出现处数，必须仍为 1）。为假的情形＝我为了省事在迁移脚本里另写一份编码。
* **G-N5m-2 不许改测试期望凑绿**：现读三册钉过 `PROVENANCE_KINDS` 的测——
  `tests/taiji_native/test_n2_06_episodic_write_contract.py` 与仪器
  `scripts/training/measure_taiji_n2_06_write_contract_values.py`、
  `scripts/training/probe_taiji_a27_memory_write_port_grid.py`——加宽时**逐册重钉**并在提交信息里
  写明"枚数由 4 变 5，钉子随枚举走"；不许新增宽泛豁免。为假的情形＝某册被改成"两种枚数都过"。
* **G-N5m-3 半径要现取**：§2 的五面计数属 2026-10-11 时刻；批次 B 开工前**重取一次**并把差值写进本件
  （`.pt` 群体在其他会话里会继续增长）。为假的情形＝沿用 §2 的数而不重测。

## §5 实施批次与批准边界

| 批次 | 内容 | 半径 | 谁能做 |
|---|---|---|---|
| A | 本件＋登记（`PLAN_INDEX` 一行、债表 G104／G100 带日期注记、08 台账一条） | 仅 `plans/` | 我可自办（＝本件） |
| B | `taiji/memory.py` 枚举加宽＋`organs.py` 通道图迁移＋§3 五判据的契约册落地 | **产品码** | 批文已到；但开工前需要 §8 的两件（整族跑许可＋重建机时）先定序 |
| C | 基座重建跑＋配对件重录（默认装载面那 13 枚的替代） | 训练跑（会更新权重） | **归 owner 排期** |

批次 B 落在批次 C 之前会造出一个"仓库绿、装机坏"的中间态（默认基座 13 枚全拒载）。
本件对此的立场：**不允许**用改默认位来掩盖它（J-N5m-5 正是为此而冻）。

## §6 判据载体在哪（措辞门要求的一节）

J-N5m-1／2／4 的执行体＝批次 B 新立的契约册（建议名
`tests/taiji_native/test_n5_27_provenance_channel_widening_contract.py`；
`n5_26` 已被债表列形普查册占用，落之前要再 `ls tests/taiji_native` 现取一次）；
J-N5m-3 的执行体＝同一册里"载旧件"的一支，其夹具必须用**真实四通道 payload** 而不是我手搓的形状；
J-N5m-5 的执行体＝`train_seed_corpus.py` 的默认位装载路径＋一支按 `rc` 断言的测。
⇒ 本件没有任何一条判据是手算的；本件落盘时五条判据**全部未出值**（批次 B 未开工），
这属于"判据已冻、尚不可判"，不是"已验收"。

## §7 落笔即钉的三行

* **预期绿**（批次 B 完成后）：J-N5m-1／J-N5m-2 两支／J-N5m-4 两支；J-N5m-5 在批次 C 之前
  应判"拒载被响亮报告"＝成立（它要求的是 fail-closed，不是装载成功）。
* **仍红时要收回哪句**：若 J-N5m-2 的反支不等，则"零权中性"这句收回，并点名是 `fan_in` 还是
  受体尺度（`counts.rsqrt()`）在动；若 J-N5m-3 不抛错，则"旧件不会被误载"这句整条收回，
  因为静默补零会造出一枚看起来正常、实际错位的基座。
* **发表资格前置**：在批次 B **且**批次 C 完成并出读数之前，以下两句仍不许发表——
  「自生成内容进不了情节场编码」「系统知道刚改了哪些参数」（沿用 DEBT-G100 行的口径）。

## §8 交回 owner 的两件（不是新决策，是把已批文的边界说清）

1. **重取后的半径与批文的对应关系**：owner 按「230 枚 `.pt` 拒载」接受，而今天能命名的面是
   867 盘上／**0 入库**／**13 默认装载面**／795 历史实验件／59 运行时自生物。
   实际"坏掉的是装机可用性与历史复算"，不是仓库完整性。要确认这条批文在**分层后的形状**下仍然成立
   （尤其是那 13 枚里的默认基座必须由一次重建跑替代）。
2. **顺序**：批次 B 需要整族回归（`python -m pytest -q tests`）作为门；该长跑在助手侧两次被权限分类器
   按"未在本轮点名"拦下。要么授权我跑，要么由 owner 在终端跑并把汇总行交回登记（与 #51 同一道门）。
