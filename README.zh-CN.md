# Seed — Taiji 原生认知架构的运行时

[English](README.md) · [项目地图](docs/README.md) · [当前执行](plans/active/roadmap/03_CURRENT_EXECUTION.md)

Seed 托管并对外提供 **Taiji**：探索持续状态、记忆、局部学习和行动的研究架构。仓库包含 Python 研究运行时，以及 Taiji Harness 产品外壳。

## 当前状态

N1 最新诊断在同九枚 checkpoint 上，对5道已消费的v6 `unknown`题测试了观测数和实体名距离。position 臂逐字输出 `unknown`：d=1 为0/15，d=2 为1/15；观测数2或4没有改变命中。run88来源守卫绑定的v3 train流中，48个unknown训练项没有d≤2近邻样本。PLAN-N1-100九模型训练通过冻结的position门，但v7的4观测和v8的8观测sacrifice范围检验均只有4/9端点可用，低于冻结6/9；两个confirm子集都保持未查询。v8范围查询180条、CPU36.671875秒，来源/代码守卫未变、0更新；两次测面恰为相同四个端点过线，因此PLAN-N1-101以`range_not_met`关闭，`can_promote=false`。唯一提案是[PLAN-N1-102](plans/reference/PLAN-N1-102_v12_observation_range_probe_draft_20261017.md)：新建12观测、d=1的测面并保留所有门；这会改变测面，尚未获批、未实施或查询。详见[v8范围判读](plans/reference/EXP-N1-101_v8_range_adjudication_20261017.md)、[v7范围判读](plans/reference/EXP-N1-100_v7_range_adjudication_20261017.md)、[冻结的PLAN-N1-101](plans/reference/PLAN-N1-101_v8_hard_neighbor_range_recovery_20261017.md)与[PLAN-N1-100](plans/reference/PLAN-N1-100_balanced_hard_neighbor_training_and_v7_20261017.md)。

活动研究为 **N 主线**，N1 正在研究学习型读取、实体绑定和状态转移。已有结果属于限定实验；训练、未见评估、保持和消费者行为分别记账，尚不足以证明通用对话、AGI 或自主进化能力。

截至 2026-10-17，run88c 的三个 fresh position 种子已在"末周期学习率衰减"（前 18432 步 0.001、末 6144 步 0.00025）配方下跑完，并由一台零算力复核仪从封存分片独立复算出：三种子全部通过冻结的六训练门；至 18432 步（含该里程碑）为止的 loss、五组更新读数与各 960 行评分逐项复现 run86c 基线；正式 6 次加预检 3 次独立恢复均 equal、110 枚来源守卫未变、CPU 3385.578125 秒在 4800 秒上限内。训练充分不等于能力：候选仍不可晋升、`TRAIN_SHA` 保持 None、v5 确认面仍未消费；判读件同时记下反向读数——同一衰减使 seed13 少 1 题，而被修好的失败原本只出现在三个种子中的一个。两条对照臂批次随后均收口并核验，九枚同协议终端也按已冻绑定对 v5 **只查询了一次**：四门合取**不成立**——`advantage_cut` 假（position 79/96 对 cut **80**/96）、`each_position_kind` 假（seed13 的 lookup 0/8、revision 0/8），`each_position_seed`（31/32/16）与 `advantage_terminal`（对 33/96，差＋46）为真。只读已封存的 288 行还能看到：position 与 cut 在 **79/96** 个未见 (seed,item) 上逐字输出相同，且"只有 position 对"为 **0** ⇒ **位置读出臂在未见证据上没有一次做到 cut 做不到的事**，失分按 seed 与题目类分布而与读取臂无关。该面已消费、不可重跑；另有一格 `sources_unchanged` 是我自己那台绑定仪的已知假阴性（批次存 CRLF 归一摘要、收尾按原始字节复核，149 枚守卫真实改动 0 枚＝DEBT-G147），故这次执行也不声称工程完整。owner 随后在三条预注册出口里裁定**甲**，甲已零查询跑完：一台只读仪把那 288 行逐题归因后出版 `reports/taiji_n1_confirmation_error_attribution_run89.json`，读数是 **`selection_side=96/96`、`compounding_only=0`**（每一行自由生成失败处，教师强制也没选中目标序列），错形只有 `first_symbol_wrong=69` 与 `prefix_then_diverge=27` 两类非零，**`transition` 类 72 行零错**，而三臂同错的 16 组全部落在 seed13 的 8 个 lookup 与 8 个 revision 上——这 16 组里两两输出相同的组数为 **0**，即三枚终端在同题同错时错成三个不同的串。这些是"下一步该测哪个变量"的关联读数，不是机制、不是臂差、不是任何能力格。详见[甲路逐题错因归因判读](plans/reference/EXP-N1-91_run91_error_attribution_adjudication_20261017.md)、[v5 未见确认判读](plans/reference/EXP-N1-90_v5_confirmation_adjudication_20261017.md)、[run88c 整批评判](plans/reference/EXP-N1-88_run88c_batch_adjudication_20261017.md) 与[当前执行计划](plans/active/roadmap/03_CURRENT_EXECUTION.md)。 随后的 oracle 地址探针（PLAN-N1-92，owner 裁定的乙′）已跑完九枚端点：两条准入锚点都为真（题面提取器对封存件 96/96 逐个整数相等、适配器 native 支与 `teacher_logits` 逐位 144 对全等），420 秒窗口内实耗 132.7969 秒、432 条 teacher 序列；预注册 §4 点名的出口是 **⑷ 尺缺动态范围**——可用端点 5＜6（seed11/12 的 position 与 cut 在这 16 枚可寻址题上 native 就 16/16，三枚 terminal 0/16 且钉住磁带地址后仍 0/16），`oracle_lift_supported=false`。只可披露的一格：seed13 内部钉地址后 cut 恢复到 11/16、position 只到 2/16，它是下一格的取题依据，不是机制结论也不是臂差主张。详见 [oracle 地址探针判读](plans/reference/EXP-N1-92_oracle_address_probe_adjudication_20261017.md)。随后 owner 裁定**另立未见题集**（[PLAN-N1-93](plans/reference/PLAN-N1-93_v6_unseen_face_construction_20261017.md)，其 §1 形状参数仍待签字，因此题面未生成、端点一次未查）。在请签之前我先补了一格**仓外预飞**（九枚冻结端点、零消费、仓内四枚真写靶前后双向复核为"不存在"），两处都有收获。仪器面：三台 v6 仪器的**成功支第一次被真端点走通**（每枚端点 20 题、九枚共 180 条 native 序列，95～137 CPU 秒 ≤180 秒上限），并迫使范围检验仪升版补出逐类错误列——此前它的判据无法从自己的件里复算（DEBT-G151 已修）。读数面：**草稿的难度旋钮开不出动态范围**——每题观测数取 4／6／8／10／12 五档 × 抽样 91／92／93 共九次读数，可用端点恒为 **5/9**（冻线 6/9），唯一一次 6/9 只出现在某一个抽样上、同档换抽样即回 5；卡在线下的永远是同样四枚（seed11/12 的 position 与 cut），而过线那五枚里有三枚是损伤的 terminal 臂白送。另有一条我自己想出来的"更难"方案（多步更新链）在写代码之前就被证据否掉：课程生成器固定只发单步更新，加长链测到的是没见过的句式而不是能力。详见 [v6 预飞判读](plans/reference/EXP-N1-95_v6_preflight_range_sweep_20261017.md) 与其 [九次读数件](reports/taiji_n1_v6_preflight_range_sweep.json)；现行队首原为 owner 三裁（§1 旋钮签什么值／范围线是否改按能力端点重述／§3 两条线哪条算判据＝DEBT-G152）。〔㊵-1035 第二轮预飞另测一条**同分布**难度维——把干扰实体名钉在与主体名只差 d 个字符上（句式一维不动，只压寻址绑定）：**这条维打得开尺**，d=1 在四个独立抽样给可用端点 **7／7／7／8**（冻线 6），观测 8 时给 **9/9**，d=2 退回 5/9，且 `unknown` 逐类平均错误 0.56→3.89（弃权类第一次有动态范围）。同时暴露一条**结构性互斥**：terminal 臂本来就在底，同压下 `advantage_*` 双侧降级甚至反号（seed13 预测 −2／−3），八次读数里"过范围门"与"预测红格少"从不同时出现 ⇒ 这九枚端点上"有动态范围的未见面"与"能过已冻四门的面"互斥。〔㊵-1036 收回：那句"互斥"取自我把 `advantage_*` 按每 seed 与 12 相比的**手算**，而 PLAN-N1-94 冻的是 **12/60＝三 seed 合计**。按执行器自己的 `evaluate_gates` 重算 16 次读数：`advantage_cut` **0 次过**（值 −3～＋7）、`advantage_terminal` 15 次过、`each_position_kind` 0 次过（seed13 的 lookup／revision 每张面都 0/5）、四门合取 0 次全绿，而范围门 6 次过 ⇒ **范围门与四门是两条独立轴**：四门不过的原因是"position 与 cut 在这族题上不可分"（第三次独立复现）与"seed13 两类零检索"，与题面难度无关。更正件 [reports/taiji_n1_v6_preflight_gate_forecast.json](reports/taiji_n1_v6_preflight_gate_forecast.json)，判读 EXP-N1-95 §10／§11；裁定重心因此从"签哪个 d"移到"这一格还跑不跑"（甲＝签 d=1 并照预注册发射一次／乙＝先重述判据／丙＝按 v5 结项并转保持集・C4・三器官／丁＝签 8／1）。〕读数件 [reports/taiji_n1_v6_preflight_confusability_sweep.json](reports/taiji_n1_v6_preflight_confusability_sweep.json)，队首据此改为 owner **四裁**（①签 d=1，其第一次确认查询预期落预注册出口 ⑵，那是合法出口且买到的正是上限卡在哪一格；②或签 8／1，范围最大但预测九格全红；③或在消费之前重述四门，须另签；④§3 的分母与"哪条线 gate"＝DEBT-G152）；确认查询一次未发射。〔㊵-1042 更正：这句已作废——确认查询已于 2026-10-17 发射并一次性消费。〕产品阶段交付与研究能力验收分别判断。〔㊵-1042／1043 · v6 已跑完三步：§1 签署为 d=1、DEBT-G152 裁"以仪器为准"并据此改写题面文句；牺牲半范围检验 **7/9 过线**（门槛 6）⇒ 首次在未见面上证明尺有动态范围；确认半 180 条查询出值＝`advantage_terminal` 17/60 真、`advantage_cut` 0/60 假、`each_position_seed` 12/12/1 假、`each_position_kind` 假（弃权三格全 0）⇒ 落预注册出口 ⑵，**C3 未达成**，`retry_allowed=false`。卡口两格：seed13 检查点（position 1/20、三臂同错 14/20）与位置臂从不弃权（0/15）。判读 [EXP-N1-96](plans/reference/EXP-N1-96_v6_confirmation_adjudication_20261017.md)；队首＝owner 在其 §8 三候选里裁。〕

## 架构与学习

- `taiji/`：原生认知机制和研究原型。
- `seed/`：兼容 API；`seed_platform/`：存储、谱系和运行时服务；`api/`：后端接口。
- `taiji-harness/`：独立 TypeScript workspace 中的客户端与产品外壳。
- `neuroplex/`：历史对照及兼容代码。

底层基座采用局部预测误差学习，部分发育训练序列原型使用优化器及 autograd，学习主张必须对应具体实现与实验。长期目标见[核心需求](plans/active/TAIJI_CORE_REQUIREMENTS.md)与[架构合同](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)，实现细节见[代码 Wiki](TAIJI_CODE_WIKI.md)。

## 快速开始

需要 Python 3.12+ 与 PyTorch。以下只演示小型字节学习 API，不代表通用聊天模型。

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/seed/test_project_identity.py -q
```

```python
from seed import Seed

model = Seed()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)
checkpoint = model.checkpoint()
restored = Seed.from_checkpoint(checkpoint)
print(restored.generate(b"a", length=8))
```

完整 Python 检查使用 `python -m pytest tests -q`；部分测试需要可选依赖或本地产物。局部通过不代表完整 CI 健康。

## 产品外壳

```bash
python -m pip install -e ".[dev,legacy]"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

客户端 workspace 的启动方式见 [Taiji Harness 贡献说明](taiji-harness/CONTRIBUTING.zh.md)，本地提供者路由为 `/api/chat/stream`。

## 仓库导航

| 位置 | 唯一职责 |
| --- | --- |
| `taiji/`、`seed/`、`seed_platform/`、`api/` | 模型机制、运行时与后端 |
| `taiji-harness/` | 产品客户端与打包 |
| `scripts/`、`instruments/`、`tests/` | 可复用任务、测量与验证 |
| `plans/` | 需求、活动调度、冻结协议和决策 |
| `docs/` | 开发说明与项目导航 |
| `reports/` | 机器可读研究证据 |
| `design/` | 品牌资产与生成工具 |
| `data/`、`checkpoints/`、`taiji_data/` | 本地数据集和模型状态，迁移前须保护 |
| `output/`、`outputs/`、`artifacts/` | 历史运行及归档路径；已跟踪JSON归入 `reports/legacy/` |

遵守[目录规范](docs/FOLDER_STRUCTURE_RULES.md)和[仓库卫生规则](docs/REPO_HYGIENE_RULES.md)。训练须遵循对应冻结协议及预算，并先通过 checkpoint 保存恢复检查。历史失败实验保留为证据。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。
