# R2 continuation v1.2 来源与预算修订

用户明确批准将本续训窗口累计上限提高为每臂1,500,000、三臂合计4,300,000 process-CPU秒。批准仅提高同一原窗口的ceiling；原三臂actual CPU与未知尾账继续计入，不重置、不重训。

v1.2 write-once amendment链接v1原contract、v1.1 amendment、三臂原sidecar和同一个append-only aggregate ledger。v1 contract与v1.1 amendment保留原字节。新source binding记录经过review的runner、backend、source validator、tests及本报告SHA。budget guards只在v1.2 amendment成功发布后使用新ceiling。

本次唯一允许修订的runtime历史是三臂各自一次`continuation_started`、资源预检通过，并以`throughput_projection_exceeds_remaining_budget`停止；不允许restore或training commit、其他stop原因、pending reservation、terminal result或存活writer。全量账本链在共享OS aggregate lock内回放，拒绝未知事件。先验证前序修订/旧lineage、三臂来源、已prepared parent副本和receipt、旧/new预算，再原子write-through发布v1.2。

按v1.2准备后的进程重新读取三臂累计账本，检查每臂与总额度、资源、速率并发起受限恢复；所有parent restore必须计费且与原latest receipt匹配后才可更新权重。超上限时耐久停止。W5b、dev/reference、selector及能力晋级保持冻结。
