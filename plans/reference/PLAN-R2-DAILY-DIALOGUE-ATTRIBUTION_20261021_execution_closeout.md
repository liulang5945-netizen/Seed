# R2-DIALOGUE-02 执行结案：自由生成输出离线归因

**日期：2026-10-21**  
**状态：归因完整；0模型加载、0checkpoint加载、0生成、0更新；`can_promote=false`。**  
**方案：** [PLAN-R2-DAILY-DIALOGUE-ATTRIBUTION](PLAN-R2-DAILY-DIALOGUE-ATTRIBUTION_20261021.md)

## 结果

归因器核对retry2的48条自由生成记录、12棵final对话树和已封存结果，SHA与门均通过。父档在6条有历史的输入上，含/不含历史的输出6/6都不同，说明生成状态会响应历史变化；在6条无历史样本上，多轮candidate与单轮control有4/6输出不同，说明续训改变了输出分布。两项都只表示输出发生变化，不表示变得正确或自然。

结合逐条样例，candidate的end-marker边界率9/12，优于parent 1/12和single-turn control 5/12；但candidate仍有重复字词与片段拼接，无法稳定给出连贯、贴题的日常回复。输出结构上的进步尚未转化为对话语义质量，目标能力尚未达到。

## 证据边界与下一步

这份归因没有加载模型，也没有生成新输出或读取参考回复。12条final样本仅供定性查看；parent/context差异和candidate/control差异都不能单独解释语义能力。本批不晋级、不自动追加同配方训练。

下一步转为**固定数据不变的matched continuation**：从retry2 candidate与single-turn control的final full bundles各续训6个epoch，共观察candidate是否能从相同对话材料继续学出连贯响应；保留原未消费的70棵final树作为一次新的自由生成面。细则见[PLAN-R2-DIALOGUE-02续训](PLAN-R2-DAILY-DIALOGUE-CONTINUATION_20261021.md)。

完整归因JSON：[taiji_r2_daily_dialogue_20261021_attribution.json](../../reports/taiji_r2_daily_dialogue_20261021_attribution.json)，SHA-256 `8ae2ee97776d65aaa83cc79b71429f1b119dd671064a645d9d90fceed7cf87d2`。
