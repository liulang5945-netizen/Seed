# N1-72 · 已知面冻结writer交换诊断

71完整但原信用优势门失败：position91/96、cut92/96，v4已消费。先定位最后阶段writer变化是否有用或与读出协同，不能盲目再改结构或拿旧面调参。

## 固定范围

仅52母档、70的三个position与三个cut终点；严格核70/71完整身份及各原档bytes，71结果SHA `eaee44f6eb980d9b0a3632a39a32c3898d6eecae72e869cc5fff33c45160250e`。每seed四条件：position原生、position仅换52母writer、cut原生、cut仅换同seed position终点writer。交换精确writer.*四参数，不能换embedding、读出/相对地址、local/context/relation分支或Adam状态。每条件结束恢复原writer，完整parent bundle/RNG/历史必须逐位不变。

固定70原有576已知train/probe四视图（192原题+96 pair+192已知改名+96改名pair），三seed×四条件＝6912 teacher/free组；0dev/holdout/v4/新变体查询、0训练或副本更新，单线程CPU≤180秒。两种native每seed必须逐项复现70终点576行，否则整批停止，不能把错误归因于writer。仅描述交换增损/逐类/原始bytes、四参数漂移范数与指纹，不设事后能力门，不把oracle/交换结果当正常模型能力。

保留所有6912行及source/code/state守卫，exclusive不重跑；预算/错误保存已得部分并停止。诊断产物：`scripts/training/audit_taiji_n1_frozen_writer_exchange.py`、`tests/taiji_native/test_native_frozen_writer_exchange_contract.py`、`reports/taiji_n1_frozen_writer_exchange_20261016.json`、plans/reference裁决。先提交冻结/实现与纯合成合同，再一次执行。

## 后续纪律

据证据只推荐一个修订方向。若已训练writer已足够，不能为凑优势削弱cut或改旧门；应审查哪个更高层能力缺口真实需要新学习，并另冻新任务/保持测面。若writer交换有害，应区分表征使用和分支协同，不能直接宣称学习无效。全项目出口不缩为当前小试，默认/远端/数据删除权限边界不变。
