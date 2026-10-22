# target-byte updates吞吐投影修订

真实故障：最近16个episode的平均长度会随corpus位置大幅改变。full-turn在9,856停止时，8.3139 CPU秒/episode被外推到全部20,240个剩余episode，预测171,768秒，超过臂内152,478余量；其实际target update吞吐并未同比恶化。该同一预测还阻断另外两臂。

修订只改变预测单位：预训练总分母14,517,912 target_byte_updates，assistant SFT总分母113,209，episode终点30,096/525均保持。采集同一run/stage内CPU增量除以target update增量，重启/停止切断样本段，不把初始化或未知尾账跳变当成模型吞吐。按最近256个有效区间汇总，以parent历史和extension历史中较高的CPU/update作预测；SFT取pretraining rate×1.2、冻结SFT worst-probe rate及可用实测SFT rate中的最大值。仍乘原1.15 safety factor并加原completion reserve。

实际硬预算、每个work unit reserve、共享aggregate ledger、CPU/内存门、checkpoint保存与独立恢复门均由未修改的v1.3训练实现执行。新manifest绑定resume入口、projection helper、receipt ancestry adapter、projection tests及v1.3来源修订和起始receipt。6项检查覆盖长短message不影响CPU/update、restart费用不污染吞吐、预算不足仍拒绝。旧预测失败保留。

只在已确认旧句柄退出后从原latest恢复：full=9,856，warm=14,912，fresh=15,136。新句柄分别45943/45618/39673。真实启动已出现新episode_completed，接下来监测新增耐久checkpoint直至原终点；不改预算、不删原事件、不运行dev/reference/selector或晋级。
