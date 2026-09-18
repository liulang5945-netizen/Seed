# M5 R2-D7：字级 token 图族假设包（H-Tok，冻结版）

2026-09-18。状态：**已冻结**（用户在 [D6 结项评审](M5_R2_D6_MATCHED_DEV_REVIEW_MEMO_20260918.md) 阶段裁决「H-Tok token 化包」后提交即冻结）。前置：D3–D6 收敛结论——字节图族（几何/损失/覆盖/驻留/归纳五条线）全部判否；根因判定为**表示粒度**：UTF-8 字节不是词的可寻址单位，一词多决策 + 字节共享（汉字首字节 E4–E9 趋同）使任何字节级链条先验在值域不相交的 dev 上必然稀释断裂。H-Tok 是对该归因的**决定性检验**：把决策单位改为汉字（一词一决策），若成功则粒度是根因、若仍败则问题在寻址/任务族本身（证据链双向闭合）。

## §1 假设 H-Tok（冻结）

> 在与 D1 仪器完全相同的评价口径下（自由生成、逐字精确、值域不相交、反事实对），把内部表示与生成单位从 UTF-8 字节改为**汉字字符 token**（每汉字=一个单位；词=token 序列）后：(a) 回答一个值从「连续 3·k 次独立字节寻址+延续决策」降为「k 次字级寻址」，多字值第二步不再有字节碰撞稀释；(b) D4 的 copy 值监督与 D6 的发射匹配 induction 在字单位上重放，匹配集从「E7 开头的几十行」缩到「该字本身的行」；(c) 未知字（dev 新值字）经 copy 通道的**码位输出空间**产生，无需进入词表——生成空间 = 训练词表 ∪ 材料码位集，与字节图 256 固定全域结构同构、不引入语料泄漏。若 dev 双字值 full_rate（K1）与 M4 翻转对被打开，则「表示粒度」根因成立；token 化线之后 R2 在图族内无更多未检验轴。

## §2 图族规格：`SequenceCharWorkspace`（新模块 `taiji/sequence_char_workspace.py`，独立版本空间）

1. **Token 化**：字符串 ↔ Unicode 码位序列（每字符一 token）。**训练词表** = train(v2) 全部字符 ∪ {PAD 不用}，boundary=0、unk=1、其余按排序映射；**断言 dev/final 独有字符不在表中**（unk 压缩）。
2. **输入侧**：前缀（问题+材料+提示尾部）逐字符查表；不在词表 → unk 共享 embedding。**编码器扫描、workspace 行、renderer 步进全部以 token（字符）为步**（workspace 行数=字符数）。
3. **输出空间** = 词表 ∪ 材料码位集：p = gate·p_vocab(词表+b) + (1−gate)·p_copy(**scatter 到行的码位值**，码位空间 0x10000 稀疏 `index_add`；boundary 在 copy 中零质量，与 v4 同构)。生成贪心 argmax 解码回字符；unk 出现即错误计数。评测 decode → UTF-8 字符串与 gold 精确比较——**评价口径逐字不变**。
4. **机制继承（不新增变量）**：per_position 证据、正弦位置键、4 头读取、问题条件化起点、copy 混合 gate、**D4 值位置 copy NLL λ=1.0**（值 mask 规则同构，改为**字符**单位：fact/sof 全 response、negation 去「不是」两字、unknown/combo 空）、**D6 induction**（前一 token=输入符号的匹配行的**后继行** bonus）、D5 教训条款（canonical 表尾、冻结重放断言）。参数量：词表 ~230 字符 × 宽度 + 与 v8 同构主体（报告精确值）。
5. **版本空间独立**：`SEQUENCE_CHAR_WORKSPACE_VERSION=1`（char 图族 v1；与 byte 图族 v2–v8 不互换，恢复测试各自文件内）。
6. 数据臂：**v2 train**（多字值监督是 H-Tok 的主检验面）；训练微批 8、固定顺序、30 epochs、lr 0.01、seed 20260917（probe）/三 seeds（matched）。

## §3 实现门（零训练）

1. **无泄漏断言**：词表由 train 生成；dev/final 值字（灰/乳/米/白/黑/墨/漆/暗/红…逐一列出）∉词表；材料 copy 空间允许任意码位（这是输出身份通道，非 embedding 先验）。
2. 结构：p_copy 码位散布正确（重复字符行质量求和）、p_copy[boundary]=0、输出分布候选集=vocab∪材料码位、gate 混合形状；H=1 退化与多头清单。
3. 值 mask 与 induction：字符单位规则（「不是」=2 字符、双字值=2 行）；induction 匹配集=**单行级**（dev 新字对：前一 token 匹配材料中该字行→后继行唯一）。
4. 冻结重放不适用（新图族无历史 checkpoint），改为**自洽 preflight**：零步保存→新进程 logits 一致→续训一致（byte 图族同纪律）。
5. 训练集成：train_step(episodes with value_masks) 与 byte 版 API 对称；checkpoint 恢复矩阵（char-v1 自身 + 拒收 byte payload）。

## §4 probe（train-only，一次）

主臂 **T1**（v2 train + char 图 + λ=1 + induction）：
- P1 复制支持行（fact/sof/negation 144）M1≥0.90；P1b **多字值行 72/72**；
- P2 值位置 copy 概率≥0.90；P3 entry_rotation 错位后≤0.50；
- P4 稳定性（Q1 口径：逐对增幅≤15%、e25→30 回撤≤0.10）；
- 描述：induction bias 终值符号、unk 输出率、combo_different 仍 0 的 C/B 信号延续。

## §5 matched dev 与路由（预承诺）

- 评价：v1 dev 98 题、G1–G6 口径不变；**K1=dev 双字值 full_rate≥0.30 为主门**；K1b=first_byte(首字)率相对 A00 0.529 不降；δ 基准仅描述（跨粒度不强推）。
- 路由：
  | 结果 | 路由 |
  |---|---|
  | G1–G6 全过（首次） | H-Tok 支持 → 包评审（集成/扩规模授权另行裁决） |
  | K1 过、G2/K2(M4) 败 | 词复制打开但绑定精度不足 → 组合层新假设（指针对象化）候选升级 |
  | K1 不过、P 系 probe 全绿 | train 可学而 dev 仍零——「表示粒度」根因**判否**，R2 图族内无剩余未检验轴 → 阶段评审（收束/任务族外部对照） |
  | probe 失败 | 回实现门（粒度转换 bug 优先），不放预算 |
- 变体线预承诺：本包后 char 图族不追加 token 粒度变体（子词/BPE/拼音=变体，除非 K1 过后的评审另行开线）。

## §6 边界与预算

不读 final；D1 仪器/fixture/评分器不动（token 化仅在图内部）；`growth_admitted=false`、`can_promote=false`。probe 1 运行 + matched 1 臂×3 seeds+自洽 preflight ≈ 5 次训练（<15 min CPU）；checkpoint 隔离 reports/r2_d7_checkpoints/。
