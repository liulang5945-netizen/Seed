# PLAN-N1-47 writer/question激活交叉合同

按推荐本地主线直接执行授权。46四门失败后不追加训练/调权重。固定41及46 final seed11，复用43原192＋pair96及已有一一对应改名校准。每模型288组，每组原始native reference、改名native、改名路径中仅原始writer状态供体、仅原始question状态供体、两供体同时，合2880条gold-prefix teacher。CPU≤60秒、58秒留收尾，0参数更新/自由/dev/holdout，不减分母、不重复失败。

writer供体仅原始观察逐byte writer states（及一致raw key/value）按完全相同位置映射到改名tape；保留接收session/source provenance和原生双hop/相对地址/condition机制。question供体为原始问题编码的全部question_states及初始化relative.hidden；供体不带已读地址/conditioned tape。供体全部来自同模型已知训练等价题，不使用gold来源/地址，不据错误预测选择供体。此为离线激活因果干预，混合路径不是有效能力评估，结果不得计入模型能力。

41原生576控制对46 step0、46原生576控制对46 step6144，symbols/正确数/CE1e-6逐项复现；两个供体同时的576条须复现对应原始reference（逐位logits优先，CE/symbols完整），未复现拒判。原生采集不改变teacher语义，位置/target长度一致，完整唯一2880分母、train-only、独立供体与无状态泄漏先测试后冻工具执行。

记录每题teacher symbols/CE/可见byte及boundary正确数，按模型/原或pair/四原始类别/干预条件/原与改名原生正确组合汇总，并比较第二hop地址差异。参数/完整Adam/RNG/原会话、source档/报告/全部manifest/源码前后不变；本次临时session只清内存。依据结果只选一个下一步机制修订并自动执行。46失败、所有既有门、全项目保持/C4/联合三器官与CI债保留，不默认采用/推远端/删数据。
