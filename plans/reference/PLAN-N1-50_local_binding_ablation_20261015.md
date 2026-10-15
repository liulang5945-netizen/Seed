# PLAN-N1-50 局部问题残差冻结消融合同

按推荐本地主线直接执行授权，49四门失败不追加。只读49 final seed11，复用原192＋pair96＋43固定改名192＋pair96，共576项，每项native和旁路local binding两条件，共1152条gold-prefix teacher。CPU≤60秒（58预留），0更新/自由/dev/holdout，不重复失败或缩分母。

旁路仅把新增bind_question返回设为原question_state的hidden及逐byte question_states，不改旧37问题condition/双hop/相对地址/已训练其他参数；native576项的symbols/正确数/CE1e-6必须逐项复现49终点。两条件target不变，记录teacher全对/CE/可见byte及boundary，第二hop地址差异与逐题得失；不生成自由输出，旧free读数只离线引用。

原生路径离线统计local salience落在问题实体byte的mass及其均匀先验、问题实体/模板byte的最大局部相似度；若实体在观察出现，统计问题实体行的alignment落在同名观察实体byte上的mass与位置均匀先验。absent时对应mass记null，按实际问题实体是否出现分组，不混原unknown的已训练pair。实体span从既有train元数据解析仅用于诊断，不传入step/绑定/输出。固定采集不改原生teacher语义，独立旁路及train-only/1152完整唯一分母先测后冻执行。

完整参数/Adam/RNG/原会话和源码/49源报告/档/48工程/全部manifest前后不变；本次session只清内存。该诊断仅检查通路利用与对齐相关性，attention不等于理解，旁路效果不算正常能力。完成后只推荐一个证据支持的机制修订并连续推进。所有旧门/失败/保持/C4/三器官/CI债不变。
