# M5 / R2 内容绑定：零训练实现门交付报告

日期：2026-09-19。依据：[内容绑定实施合同草案 v1](M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md) §6"允许分步但不跨权"——本轮唯一交付**静态数据/评分器＋计算图/恢复实现门**，未启动统计训练、标定或确认；训练预算审批仍是后续独立决定。当前状态记录见 [03 §5.7](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 交付物清单

| 工作面（合同 §7） | 交付物 | 说明 |
|---|---|---|
| 数据 | `scripts/training/build_taiji_r2_content_binding_data.py`；fixture 三件套 `tests/fixtures/r2_content_binding_v1_{train,calibration,sealed}.jsonl`；[数据报告](../../reports/r2_content_binding_v1/data_contract_v1_20260919.json) | 7 类题组×3 split 物理分文件；语料 digest `377a7391a5b7…6580c7` |
| 评分 | `scripts/training/eval_taiji_r2_content_binding.py`；[静态基线报告](../../reports/r2_content_binding_v1/static_policy_baselines_v1_20260919.json) | 七类成对评分器（后续标定/确认复用同一规则）＋4 条固定策略＋参照键 |
| 模型 | `taiji/sequence_content_workspace.py`（format `taiji-sequence-content-workspace-v1`） | A/B 两臂独立 config/model/trainer 格式；公共编码/生成结构共享，内容模块独立；旧 char-v1 模块零改动 |
| 训练/快照 | `scripts/training/train_taiji_r2_content_binding.py` | 有界 runner：复合 loss 分账、两配方、选择/停止规则、atomic checkpoint；训练进程无 calibration/sealed 读路径（calibration 仅 `--mode calibration` 下的评价函数可达；sealed 全模块无引用） |
| 测试 | `tests/taiji_native/test_content_binding_data_v1_contract.py`(12)＋`test_sequence_content_workspace_v1_contract.py`(13)＋`test_content_binding_runner_contract.py`(10) | 35 项实现门测试全绿；旧 char-v1 合同测试 9 项回归全绿（合计 44 passed） |

## 2. 静态数据核验结果（合同 §3）

- 配额：train 256 组/类×7＝1792 组/3584 题；calibration 32×7＝224/448；sealed 64×7＝448/896。每组恰两成员（a/b），同组只在一个 split。
- 组内反事实按类成立：fact_flip/object_swap/relation_flip/negation_scope/missing_to_filled 两题答案不同，distractor_invariant/unknown_preserved 相同；答案语义逐类符合合同 §3.1（完整值/相同不同/是否/未知）。
- 组级留出键（实体词集、完整值词集、模板族）跨 split 无交集；对象词池与值词池逐 split 离散（字符允许共享）；词面清洁检查（值词不内嵌于值词/对象词/模板文本）通过。
- 值长平衡：6 个含值类每类单字组＝双字组（各半）；否定询问对象、干扰位置精确对半；材料提及对象数分布 1/2/3（如 train：768/2304/512）。
- 长度：最长 prefix 38 字符 ≪ 256；response ≤ 2 字符 ≪ 32；确认集全部预定题在范围内。
- sealed 切片分母：可复制回答 448；未见完整值 448（≥64✓）；含 train 未见字符 340（≥64✓）。
- **独立参照解题器**（解析渲染文本，不读结构化标签）在全部 4928 题与标签一致。
- **固定策略门**（合同 §3.2，train 上 事实/对象/关系 三类成对宏平均 ≤ 0.25）：fixed_unknown / first_value / last_value / second_object_value 四条策略宏平均**全部为 0.0**——布局规则（fact_flip 组内目标位置交替、object_swap 同材料换问、缺失组共享干扰事实）使内容盲策略结构性无法成对得分，非随机压线。参照键成对宏平均 1.0。

## 3. 计算图核验结果（合同 §2）

以 train 词表（113 字符，train 文本构建）计：

| 项 | A（隐式对照） | B（绑定候选） |
|---|---|---|
| 参数量 | 114,483 | 108,963 |
| 每 episode 计算量估算 | 1,213,440 MAC | 2,641,920 MAC |
| 内容模块计算量估算 | 87,552 MAC | 1,516,032 MAC |

按合同 §1 要求披露：B 的内容计算约为 A 的 17 倍、参数反而少约 5.5k——"纯槽位因果/参数量无影响"不可主张。实现门测试断言两臂 profile 均已披露。

逐项核验（35 项测试）：
- **未见字符身份**：字节特征统一加在所有字符上（≤4 UTF-8 字节×8 位＋5 维长度 one-hot，padding 槽与 byte 0 可区分）；两个未见字符投影前可区分；未见输出字符保留真实码位并进入动态候选（无 UNK 回退）。
- **无 oracle**：运行时 API 只收 question/material 字符串；篡改记录的标签字段不改变输出（签名检查＋行为检查）。
- **公共初始化**：23 个共享参数按稳定名称同 seed 初始化，A/B 及创建顺序互换下逐位相等；发射后继 bonus 两臂初始严格为 0。
- **C 消费**：loss 梯度到达两臂内容模块参数（A：head_query/a_content_mlp；B：slot_init/relation_mlp）；对 C 施加扰动同时改变词表路径与 copy 路径的分布（`generate_with_content` 诊断入口已备）。
- **归一化/边界**：有无材料时 mixture 均归一；无材料 copy 关闭且分布有限；重复字符 copy 质量合并；超 256 输入字符/超 32 输出字符为显式范围错误（自由生成超限记 `range_error`，不静默截断）。
- **checkpoint**：保存→按位恢复→续训一步与不中断路径 digest 一致（含独立进程子进程验证）；payload 含参数/train 词表/候选构造版本/配置与数据 digest/优化器/RNG/更新数/选择规则/合同身份；digest 篡改与异图 checkpoint 均被拒绝；保存走 `atomic_save`。

## 4. Runner 与选择/停止规则（合同 §4/§6）

- 两配方 `cal_lr1`（峰值 0.001）/`cal_lr3`（峰值 0.003）；AdamW betas=(0.9,0.999)、eps=1e-8、weight_decay=0；全局梯度范数裁剪 1.0；前 5% 线性 warmup，其后余弦衰减至峰值 10%（调度数学有测试）。
- 微批 4 整组＝8 题；组级洗牌、7 类轮转均衡、组不跨批；同 seed 批序列确定；暴露/重复次数按类记录进 run report。
- 复合 loss 分账：CE（逐题答案长度均值，含 EOS）与 copy NLL（可复制值位置，非复制题为 0）分开记录于 health 统计；非有限 loss/梯度立即停止（`stopped_non_finite`）；墙钟超限记 `incomplete`，不以早期权重冒充终点。
- 选择规则：满足复制≥0.90、未知≥0.90、值有限的候选中取三类成对宏平均最大；并列优先低 lr、再较早更新。500/1000/1500/2000 保存并评价 calibration（评价函数仅 calibration 可达；模块内无 sealed 路径，CLI 无 fixture 参数——均有测试）。
- 隔离：产物目录 `reports/r2_content_binding_v1/<arm>/<seed>/<config>/`；测试产物一律写系统临时目录（DEBT-I7 纪律）；run report 记录 `split_read=train`、`calibration_read`、`sealed_read=false`。

## 5. 实施中在合同边界内做的选择（备案）

1. **固定策略的算法定义**（合同只给名字）：first/last_value 扫描材料中的 train 值词（无值回退"未知"）；second_object_value 取按对象词表解析的第二条语句值（无则第一条，再无则"未知"）。数据布局保证四条策略在三类门控类别成对得分结构性为 0。
2. **未知保持组配对方式**：同对象、不同（问句词干×材料措辞）组合的精确配对；缺题成员不伪造值长度。
3. **字节特征编码**：每字节 8 位按位展开（0/1），padding 用第 5 个长度槽而非零向量，与"padding 独立于 byte 0"一致。
4. **fact_flip 布局**：组内目标事实序位交替（a 先 b 后）＋1 个干扰对象；missing_to_filled 两成员共享干扰事实、仅目标子句变化（缺失子句 vs 补足子句）——这两条是"策略门 0.0"与"内容盲策略不可成对得分"的结构保证，同时满足干扰位置/目标顺序平衡。
5. **A 臂读出头数与槽宽**：A 的 6 路汇聚与 B 的 6 槽使用相同 48 维证据宽度与问题向量（64 维），保证两臂公共输入/生成结构逐参数同源。
6. 模板族按 (类, split) 头标记＋句式离散；calibration/sealed 问句复用 train 词干构造变体＋新头/界标（词干词面部分共享属"字符允许共享"范畴）。

## 6. 未做与未测（显式清单，不冒称）

- **未启动任何统计训练**：calibration（4×2000 更新）、formal（6×2000）、两臂比较、确认集评价全部未运行；runner 的实际训练循环只经 3–4 更新的 wiring 级验证（测试产物，写入临时/隔离目录，已清理）。
- 未读取 sealed 题目内容之外的评价（fixture 文件本身按 D1 惯例入库封存，digest 已登记）；未改旧 D1/D8 fixture、旧合同、默认入口或任何旧 checkpoint。
- 资源预检（设备/线程/精度/内存/每步时延）未做——需先有真实训练运行数据；20000 更新/12 小时/10GiB 预算未批准。
- 合同 §5 的全部能力门（BIND/DELTA/COPY/其他切片/旧 dev 回归/内容消费规模诊断/真实性恢复）为训练后判据，本轮只交付其测量入口（成对评分器、C 消费诊断入口、checkpoint 身份绑定）。

## 7. 下一步（按合同 §6 闸门顺序）

实现门已过 → 待办为**资源预检与训练预算审批**；批准后本草案升 FROZEN 并运行标定（seed 20260920，两臂×2 配方×≤2000 更新）。在此之前不启动任何统计训练。
