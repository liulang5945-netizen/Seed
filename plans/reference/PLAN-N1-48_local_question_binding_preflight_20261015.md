# PLAN-N1-48 观察条件局部字节匹配问题残差工程合同

按推荐本地主线直接执行授权，47激活干预支持优先修订问题表征，非writer大改。新增独立候选，保留37双hop/相对地址/旧问题融合，44085→47480（新增3395）；不改现有默认或旧类/冻源码。

共享embedding的逐byte五字节中心窗口（两侧用boundary embedding256补齐，各观察独立），共享可学习80→16无bias投影并L2归一；问题×观察余弦相似度经可学习温度exp(log8)、限制1..32后softmax得每问题byte的观察上下文，取raw writer states。以原question states＋每byte最大相似度经33→1线性/softmax作问题池化。共享64→32零初始化线性+tanh残差分别更新逐byte question states和初始hidden（后者取池化观察上下文），然后走原生condition/双hop，正常teacher/free同路。额外1280＋34＋2080＋1=3395参数。无实体解析、词表、gold地址/来源/训练id或供体缓存入推理。

固定迁入41 seed11原44085参数/原会话；新头seed11初始化并明确新分支RNG；完整共享Adam旧moment逐参数映射保留step6144，新头未训练无moment，后续训练新旧step分账，不冒充全新fresh或统一step。工程本批0更新/课程/dev/holdout，独立CPU≤60秒包含新档独立进程恢复，所有工程合成输入无新能力主张。初始化与41在固定合成byte输入上多步logits/地址逐位相同，旧参数/Adam逐位相同，参数量机检。

验证：窗口source边界隔离、恒同窗口投影同值、非ASCII byte可用；零残差初始新融合信用可达，合成非零融合时投影/温度/池化及question/writer信用可达，writer cut信用仍阻断；新独立format拒绝旧档直接加载与缺新头，不做隐式迁移；正常teacher/free相同step与tape/session守卫；zero保存/新进程恢复全参数/Adam/RNG/来源/多步状态逐位等价。旧源档/报告/manifest与代码不变。临时会话只清本次内存，档置data/checkpoints新命名族。

成功后唯一下一步49另版固定6144更新训练集校准，从该零残差迁移初始化、采用43原QA+保真目标与已冻stream，保留43四门；先另冻正式配方与预算再执行，48本身不含训练/留出。即使工程过，也不能宣称名称稳定/未见/默认采用/联合能力。全项目保持/C4/三器官与CI债不变。
