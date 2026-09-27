# SPEC-R2-01：语言地板可训（UTF-8 续字节加权读出重训）——开案预注册

日期：2026-09-27。归属：R2 主线语言能力（M5 退出显式排除项的重开，owner 2026-09-27 选定方向
＝"训练目标加多字节/成句约束"）。本件判据、臂定义、预算与止损**先于代码冻结**。
与 [`PLAN-A-24`](M5_R2_A_BRANCH_PLAN_REV2_20260926.md) §5d 冻结令的关系：F0 判 fail 后
表层判据冻结、A 支线不加机制；本件是**主线侧**的修复案，F0 过线后 A 支线前置阶梯才恢复。

---

## 0. 盘点事实（全部已实测，出处随条）

1. F0 判 fail：无掩码通道上五类任务整句可解码率全 0.0、前 16 字节全合法率全 0.0
   （`reports/taiji_f0_language_floor_20260927.json`）；上下文即时复述/听写/同轮问答命中全 0。
2. **M1 成句率 0.87 的真相**：FROZEN 判读仪器 `diag_taiji_r2_surface_decode.py` 的生成环
   自带 **UTF-8 合法性掩码**（`_utf8_allowed` 把非法字节压 −1）——0.8688（arm A）与
   0.3625（arm C＝纯 seed_beta）都是"仪器扶着手写"量出来的。
3. **arm A 无掩码实测也是 0/32 可解码**（本日直测 `output/taiji_r2_readout_retrain/A/checkpoint.pt`），
   且输出带"的的的的"单字循环病理、每条含 ~18 真汉字——读出重训 16M 符号提高了"认字"，
   没有修"续字节跟上调"。
4. 语料血缘干净：`data/simple_zh/dialogue_extended_clean.jsonl` 与 `data/p3b_all_fresh.jsonl`
   均为正常中文（此前一屏乱码为终端显示伪影，已用 GBK 回编探针排除双重编码）。
5. 学习信号里存在现成钩子：`observe(..., _predictive_update_scale=…)` 直通
   `predictive_readout.learn(learning_rate_scale=…)`（M4.R5 实验控制，非新增参数面）。

## 1. 目标

让基座在**被计分的无掩码通道**上自己守住 UTF-8 序列（进而成句）——
不是换解码掩码（那是已知的 readout 侧捷径，arm A 数据证明它只是掩盖）。

## 2. 判据（冻结，跑完不改）

* **主判据**：F0 重跑（同一仪器 `probe_taiji_f0_language_floor.py --checkpoint <臂产物>`、
  同一冻结过线）——**T1a 或 T1b 的整句可解码率 ≥0.5**。
* **次级（只随表报出，不入判）**：T2/T3/T4 命中率；掩码 wf（M1 风格，作上界参照，
  **与无掩码读数不许互换**）；前缀 8/16/32 可解码率；平均真汉字数。
* **三分支**：
  1. 过线（≥0.5）⇒ F1（问法效应）恢复排队；
  2. 0.1 ≤ 可解码 <0.5 ⇒ 记"方向为正、预算或权重不足"，**允许一次性**续跑至总量 4M 符号后终判；
  3. <0.1 ⇒ 记"续字节加权在该预算内修不动地板"，转其余方向（数据规模/解码掩码产品化/架构）。

## 3. 臂定义（唯一变量＝续字节加权 W；其余与 retrain arm A 逐字相同）

| 臂 | W | 观察调用 |
|---|---|---|
| **D0（对照）** | 1.0 | `observe(sym, learn=True, readout="predictive", learn_motor=False, learn_fabric=False, learn_predictive_context=False, learn_predictive_readout=True, _predictive_update_scale=1.0)` |
| **D（处理）** | 4.0 | 同上，但 `symbol∈[0x80,0xBF]`（UTF-8 续字节，即"正在写字的中间"）时 `_predictive_update_scale=4.0` |

* 语料：`data/simple_zh/dialogue_extended_clean.jsonl`（QA 标记在文本内；123,090 行、≈40MB，
  2M 符号预算只吃其中一小段——按行流式、不循环）。两臂**同一段符号流**（同起点同长度），否则配对作废。
* 预算：各 **2,000,000 符号**（smoke 各 200,000）；checkpoint 每 250,000 符号一份（F0 可做曲线）。
* 为什么 D0 必须存在：没有它，"涨了"无法归因给加权还是重训本身。

## 4. 实现清单与守卫

| 件 | 内容 | 守卫 |
|---|---|---|
| `train_taiji_langfloor.py` | 上述两臂的 runner（沿用 retrain runner 的全部纪律：out-dir 隔离、resume、stop 文件、基底 sha 只读、进度流跨会话可追加） | ①写面指纹：前后差异必须**只在** `predictive_readout`；②**续字节加权步数**真计数必须 >0（防"参数收了没传"重演——B1 首跑教训）；③两臂流对齐断言；④产物 envelope 带 trainer 名/臂名/预算/W，F0 探针可直接 `SeedRuntime.load` |
| F0 探针 | 现成，不改 | guard 四项照旧 |

## 5. 纪律

* 判绿只认 F0 报告自身 guard 全过＋主判据数值；不允许引用任何 masked 读数充当"过线"。
* 训练按**符号数**钉（不按墙钟）；`--max-minutes` 只作保险。
* 判读件不覆写、时间戳防撞、入版本库。

## 6. 预算与签字

两臂合计 4M 符号 ≈ 2×1.5h（CPU），smoke 先行。owner 已选定方向（"训练目标加多字节/成句约束"），
本件按该授权开案；**若三分支走 2（续跑至 4M）即超出本件冻结预算，届时需再确认一次**。
