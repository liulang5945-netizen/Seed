# PLAN-N1-51 有序观察条件问题编码工程合同

按推荐本地主线直接执行授权。50显示local匹配有用且同名位置mass显著高于均匀，但问题池化不充分；保留49已学匹配层，唯一修订为按字节顺序消费匹配上下文，新增独立候选不改48旧类/默认。

输入每问题byte为共享embedding16＋local alignment加权raw writer上下文32＋该byte最大局部余弦1，共49维；新增单层nn.GRU(49,32)，输出全部问题状态及末态。继承49原bind_question先得base状态（包括原局部残差），新32维context_gate参数全零，gate=tanh(parameter)，分别作base逐byte状态＋gate×(新GRU状态−base状态)、base初始hidden＋gate×(新GRU末态−base hidden)。保留双hop/相对地址/正常condition，teacher/free同路。新增GRU7968＋gate32=8000，总47480→55480，无实体解析/gold地址/来源/供体缓存/数据id入推理。匹配和上下文完全从当前问题/观察bytes与Taiji参数产生，保留顺序而非只加池化标量。

固定迁入49 seed11 final全部47480参数、完整会话和Adam moments：原reader部分step12288、原local部分6144；新GRU/gate seed11初始化无moment，新分支RNG明确。zero gate初态必须与49多步logits/地址和free trace逐位等价，源参数及每项旧Adam逐位相同。独立format与显式机制marker，拒绝旧档直接加载或缺新头，不隐式迁移。

本批独立CPU≤60秒，0参数更新/课程/dev/holdout查询，仅合成byte工程输入。先验证新gate初始信用非零、初始GRU零信用符合链式关系；合成非零gate时GRU、匹配投影/温度、question/writer可获信用。cut需阻断writer及observed-feature信用，terminal上下文按本source末态重复，不跨source窗口。tape/session守卫、非ASCII、共享生成路径、临时session清理、参数量与旧/新moment分账、zero保存/新进程恢复全参数/Adam/RNG/来源与多步状态逐位等价、所有源报告/档/manifest/代码不变。

工程成功后唯一下一步52固定6144新更新训练集校准，仍需先另版冻结预算/目标/数据/四门；51本批不训练/不查新未见。旧49失败、四门与33已消费holdout不变；保持/C4/联合三器官/根CI及14历史格式债保留。工程通过不代表能力/默认采用。
