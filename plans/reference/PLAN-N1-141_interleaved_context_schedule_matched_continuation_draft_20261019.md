# PLAN-N1-141：交错slot匹配续训，检验两行基线保持

**状态：草案，待用户批准。**本文件只提出一次新的train-only配对续训；未获批准前不做readiness、checkpoint写入、训练或计分。

## 依据与问题

已结案的[N1-140](PLAN-N1-140_execution_closeout_20261019.md)完成55,296次更新和7,488条train-only计分，N1-132原训练门5/6通过。candidate增强上下文662/864，高于两行control的411/864；但seed13 candidate原题transition由N1-139的28/48降至19/48。离线逐题归因显示17项错误重合、N1-140另新增12项并修复3项，整体净降9项。

N1-140 runner将每个父项/route的24个transition slot按顺序分配为12个baseline在前，再依序放置六种增强条件各2个。该局部成块次序是可能的时序干扰因素，但现有结果不能证明因果。N1-141只改变candidate slot顺序，直接检验交错是否能同时保住两行transition与增强上下文表现。

## 冻结配方

1. 以N1-132 seed11、12、13的position final checkpoint为父端点；六个endpoint均新建zero candidate/control bundle与fresh optimizer，不从N1-139或N1-140续训。
2. candidate保留N1-140每个transition父项的slot总量：baseline 24/48，六种增强条件各4/48；raw/hard各承担一半。每个父项/route的24个有序slot固定为下列12位序列重复两轮：`baseline, 8/d1, baseline, 8/d2, baseline, 8/d4, baseline, 16/d1, baseline, 16/d2, baseline, 16/d4`。这样每route仍是12 baseline与每类增强2个，且baseline均匀穿插。
3. control保持N1-140的原始两行输入，不加增强上下文。候选与control模型、目标函数、seed、数据、排除清单及更新次数均沿用N1-140。
4. 每endpoint固定9,216次更新，六endpoint共55,296次。所有模型训练完成并独立恢复后，按原协议计分：每endpoint 960条冻结N1-132输入及288条增强train-only输入，共7,488条。保留完整逐题输出、原N1-132门及逐endpoint的context读数。
5. CPU总上限7,200秒，包含readiness、六个zero bundle保存/独立恢复、训练、final保存/独立恢复及计分。checkpoint使用独立write-once目录；任一来源、保存或恢复校验失败即停止，不重试。

## 判据与边界

- N1-132原训练门逐项、逐endpoint原样保留；只有6/6 endpoint全部过门才算本批训练门通过。context计分只作辅助诊断，不替代原门。
- 只访问冻结的v3 train输入；不查dev、holdout、confirmation，不生成新测面，不改变source-exclusion清单，不追加训练，也不自动晋升。
- 即使全部训练门通过，也只说明train-only续训条件达到门槛，不产生未见泛化或能力晋升主张。
- 与N1-140相比，唯一处理变量是candidate中24个transition slot的时序排列；control、输入计数、模型初始化和评分门保持相同。

## 预期读数与裁决

重点比较seed13 candidate原始transition是否恢复，以及seed11/12是否保持N1-140表现；同时保留每种context条件的配对题目读数。若原门仍未全部通过，按协议结案，不再追加训练。任何改变slot配比、种子、模型结构、判据或测面都需另行讨论。

批准前本草案不授权执行。建议批准后按上述固定配方一次运行。
