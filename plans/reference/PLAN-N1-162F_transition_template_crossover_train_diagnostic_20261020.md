# PLAN-N1-162F：transition问句模板交叉train-only诊断

**状态：已依用户授予的执行计划批准权批准并冻结**  
**日期：2026-10-20**  
**依据：** [PLAN-N1-162E结案](PLAN-N1-162E_execution_closeout_20261020.md)  
**原训练执行：** [PLAN-N1-162B结案](PLAN-N1-162B_execution_closeout_20261020.md)

## 诊断问题

N1-162D/E表明seed12增强模型在转向factorial中大量输出反向turn结果；seed11也出现方向与其他字节错误；seed13整体稳定。N1-162B control与augmented在同seed下由相同N1-155A父bundle分叉，训练问句模板同时包含两种direction-after-turn形式。下一步只复测两种已消费过的direction-after-turn问句在相同训练transition上下文上的读数，区分“某个问句形式触发方向错误”与“训练后transition状态改变”；结果不作为未见泛化证据。

## endpoint、输入与题数

- 九个冻结endpoint：N1-155A `unknown8_rebalanced` candidate×3，N1-162B `surface_control`×3，N1-162B `surface_augmented`×3。全部checkpoint必须完整bundle独立恢复9/9，且来源SHA/指纹与result记录相符。
- 使用N1-132同一192个train任务中的48个transition父项；读取original与rename视图，并按冻结N1-132 factorial helper派生original、direction_opposite、turn_flip、both四条件。共48×2 views×4 conditions=384种父项上下文/endpoint。
- 对每种上下文分别生成两种已在N1-161使用过的问句：`which direction does {entity} face after turning?`与`what direction is {entity} facing after its turn?`。九endpoint共**6,912次train-only生成**。原v3问句结果直接复用已封存gate行，0次重复原问句查询。
- 每行按保存的初始方向/turn及条件重建expected answer，逐行复算`free_exact`，并分类correct、initial-direction copy、inverse-turn、opposite direction、unknown、empty及other bytes；模型fingerprint必须在整endpoint计分前后相同。

## 资源、守卫与停止

- CPU≤1,200秒；关联基线21,266.234秒、关联硬上限36,000秒。本阶段没有训练或checkpoint写入。
- 9个endpoint先独立恢复9/9；任何SHA、fingerprint、输入配对、expected answer或预算守卫失败即停止，不查询后续题、不重试。
- 0训练、0参数更新、0 dev、0 holdout、0 confirmation；新查询全部是已消费v3 train任务的问句变体，不重测或消费任何fresh face，`can_promote=false`。

## 产物

- 诊断器：`scripts/training/run_taiji_n1_plan162f_transition_template_crossover_diagnostic.py`
- 契约测试：`tests/taiji_native/test_n1_plan162f_transition_template_crossover_contract.py`
- 一次性progress/result写入`reports/taiji_n1_plan162f_transition_template_crossover_20261020.jsonl`及`.json`。
