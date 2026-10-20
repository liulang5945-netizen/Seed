# PLAN-R2-DIALOGUE-03：匹配续训扩展与剩余自然对话生成检查

**状态：已冻结并按用户授予的计划批准权准入。**  
**前序训练结案：** [R2-DIALOGUE-01](PLAN-R2-DAILY-DIALOGUE-01_execution_closeout_20261021.md) · **离线归因：** [R2-DIALOGUE-02](PLAN-R2-DAILY-DIALOGUE-ATTRIBUTION_20261021_execution_closeout.md)

## 为什么继续

R2-DIALOGUE-01在12棵留出树上使candidate end-marker率达到9/12，优于父档1/12和单轮对照5/12，但样例仍是重复字词和片段拼接，没有稳定自然语义。离线归因显示输入历史会改变输出，且candidate/control在同输入下也有4/6不同；当前缺口是语言内容形成而非只缺停止信号。OASST2已有固定许可与隐私筛选的自然对话材料，因此本轮以更完整课程量验证能否学出连贯响应，不扩数据源、不改模型结构。

## 固定续训合同

- 两臂分别从[R2-DIALOGUE-01 retry2 final checkpoint](../../reports/taiji_r2_daily_dialogue_20261021_retry2_result.json)续训：多轮历史candidate SHA-256 `c264eca87f1f9199b488d09afa1288a9f9ce65f63135ed7443b7278bb180428b`；同源单轮control SHA-256 `2b54d88917265a92bae6a5b8578ac15b699ff1f39fe19591b0390860d3a270b6`。
- 数据不变：同一OASST2 train 525 episodes、同一byte-aligned response target、同一trainer/解码配置；每臂固定额外6个epoch（此前2个epoch后，累计8个epoch），candidate仍输入最多3轮历史，control始终去掉历史。没有额外子采样、早停挑点或修改参数后重跑。
- 两个续训parent final bundle先各自fresh-process恢复；写出continuation zero bundle后再次各自fresh-process恢复，通过后才开始6个完整epoch。训练后写final full bundle、独立恢复2/2，验证输入source和前序final SHA未变。
- OASST2 final原有123 episodes中，前批已消费12棵树。本批排除这12个family，按family id每树选一个代表，共使用剩余**70棵树**（104条未消费episode中按固定规则选70个代表；history分层为55无历史、15有历史）。不查询dev。
- 固定自由生成对照：在这70个未消费树上，parent full-history/单轮、retry2两枚旧终点以及两枚8-epoch新终点各生成一次，共**420次生成**。生成器不读目标回复；不计算exact-match、teacher-forced准确率或参考答案分数。人工样例用于直接检查连贯、贴题和重复情况。
- 本批CPU上限30,000秒，含恢复、续训和420次生成。达到上限即停并保留完整partial；不缩短epoch、不重试、不晋级，`can_promote=false`。

## 解释边界

如果8-epoch candidate相对父档/单轮对照仍以无意义重复为主，结论就是当前native H3.7B语言通路未形成日常中文生成；不再在同一语料上自动叠加epoch。若明显改善，也只支持这批OASST2树与当前parent上的局部续训效果，后续仍需新的整树留出和人工样例检查。

## 产物

- runner：`scripts/training/run_taiji_r2_daily_dialogue_continuation.py`
- attempt/progress/result/generation：`reports/taiji_r2_daily_dialogue_20261021_continuation_*`
- checkpoints：`data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation/`
- 中文结案：`plans/reference/PLAN-R2-DAILY-DIALOGUE-CONTINUATION_20261021_execution_closeout.md`
