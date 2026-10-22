# Phase 7 第三轮独立复审：提交 4ce91e141

日期：2026-10-22。复审对象为 `4ce91e14125e41b6a9ce5387af67e53763719bdd`（`fix: supervise R2 evaluator IPC deadlines`）及其评测器、synthetic 合同测试与Phase 2/3/4/6合同。没有读取真实dev/reference、加载模型、运行selector或`--execute`。根任务已验证整套40项；本轮另行运行8项聚焦synthetic测试（6.07秒，32项deselected）。

## 结论

**Phase 7仍不放行。** 旧P1-07的主要路径已修复：IPC写入转入独立writer线程，父进程在deadline循环中监督写入；请求帧上限2 KiB，generation只传已持久化prompt face的路径、SHA和索引。P2-04已关闭：worker以`CREATE_SUSPENDED`启动，采样启动CPU、加入并核验Job Object后才恢复主线程；启动及清理失败均不发送任务授权。

仍有一项P1-07尾部缺口：如果deadline后writer在有限join期内仍存活，worker调用先拒绝写query终态，但外层异常收敛没有保留writer状态，并可能把该未闭合query作为已关闭评测失败写出。无法证明writer停下时不能将该任务/尾账表示为已确认。

计数：P0=0，P1=1，P2=0。旧回流预算2/2及第二轮不放行结论保持原样；本次只审查新提交，不构成预算豁免或预先放行。

## 本轮问题

|级别|位置|发现与影响|整改及复检要求|
|---|---|---|---|
|P1-07（部分关闭）|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:2096-2137,2810-2843,3278-3288,3467-3517`|请求写入已改为daemon writer线程，并受父进程deadline轮询；超时分支终止Job树、等待writer一个终止宽限期，并把`ipc_write_thread_stopped`放入`WorkerTaskFailure`。若该字段为false，`failed_task()`在写入`query_timeout`前抛普通`Refuse`。生成异常分支在调用`failed_task()`前先清空`active_task`；因此该`Refuse`到达最外层后不再是带writer状态的`WorkerTaskFailure`。外层只检查worker树状态，不检查IPC writer状态，并可能写`evaluation_stopped`，其中task/operation为空，`unknown_tail`在face已持久化时为false，也没有writer退出字段。attempt仍会被消耗且不会继续generation，但未闭合query可能未被当前终态标成unknown，最终失败记录也无法证明writer已停止。|贯穿传递writer退出结果到最外层失败收敛。若writer在规定宽限期后仍存活，拒绝生成“已确认”终态；应把当前query记作unknown尾账、标注writer退出未确认并保持attempt consumed。补synthetic故障：Job树已终止但writer仍阻塞，验证主进程有限返回/不继续工作、账本与failure文件准确标明writer/query unknown、attempt不可重试；并验证writer正常解除后能落唯一timeout终态。|

## 两项重点复核

### P1-07：IPC边界及prompt绑定

- `MAX_WORKER_IPC_FRAME_BYTES=2048`固定于脚本82行。`_json_size_upper_bound()`在构造完整JSON前给输入做上界检查（1063-1105）；超界在写入前拒绝并消费worker（2065-2095）。worker也以`readline(2049)`读取、要求换行且限制UTF-8字节数（1460-1467）。
- `request()`将`stdin.write/flush`放到writer线程；主循环继续采样CPU/wall，并在deadline触发时终止Job、join writer（2096-2137）。正常worker侧一次只消费一个已耐久授权任务。
- generation payload不再携带`prompt_row`，而携带固定face路径、face SHA、prompt SHA、index及其余授权元数据（3162-3176）。worker把face路径解析后与本次固定输出路径精确比较；要求唯一`prompt_face_persisted`事件；重算选中row的SHA，并同时绑定face中SHA列表、durable query intent与request SHA（1216-1254）。索引与durable intent再次核对（1385-1392）。因此IPC帧内额外字段不会成为生成提示词来源；固定持久face与query授权SHA锁定当前单条prompt。
- 当前测试覆盖被阻塞writer在deadline进入Job终止、超2KiB帧在write前拒绝（992-1120）。阻塞writer测试的FakeJob会主动release writer（1041-1045），没有覆盖Job已终止但writer仍存活时最外层失败汇总；这正是P1-07残留。

### P2-04：startup CPU与Job Object顺序

- `Popen`设置`CREATE_SUSPENDED`（1842）；紧接着读取process CPU并保存初始值（1844-1849）。随后建立Job、检查`ActiveProcesses`/`TotalProcesses`、记录CPU基线，然后才调用resume并复核worker仍属于活动Job（1863-1873）。因此assignment及核验期间主线程未执行导入或任务代码；创建到CPU采样的已观察开销进入`initial_child_cpu_seconds`，resume后的bootstrap CPU由Job采样并受startup deadline监督（1921-2012），worker ready后计入预算（2941-2963）。
- Job建立/assignment/Resume失败会尝试Job终止并确认；若尚未resume，则再kill/wait仍暂停的根worker，并fail closed，不创建reader或发送IPC授权（1874-1895）。
- 定向测试覆盖assignment失败不resume/不写IPC，以及`created_suspended → job_assigned → resumed`顺序；另有Job终止无法确认时不退化为psutil树推断的负向测试（1195-1396）。P2-04关闭。

## 旧问题回归核对

|旧问题|复审状态|依据|
|---|---|---|
|P1-01 worker直调绕过attempt/预算|未见回归|私有worker核验固定attempt/ledger路径、Job成员、唯一durable intent、随机token、开放任务与exclusive claim（307-400、1284-1324）；数据加载在授权后。|
|P1-02 子进程module启动|未见回归|`_worker_command()`使用repo root下的`python -m`入口（1678-1690）；worker模式拒绝非root cwd（1823-1825）。|
|P1-03/P1-04 restore/generation deadline和Job树终止|未见回归，但受本报告P1-07尾账限制|bootstrap、restore与generation均有有限期限；deadline路径终止Job并核验活动进程和已知PID（1921-2026、2061-2256）。源代码保留hang、忙循环及后代测试。|
|P1-05 selector前重新绑定来源|未见回归|四端restore后、selector前重跑preflight并比较readiness/source/report/checkpoint SHA（3100-3113）；intent先durable写入，随后持久face及事件（3115-3143）。|
|P1-06 attempt/ledger write-through耐久|未见回归|平台探测限定本地固定NTFS；exclusive temp、FlushFileBuffers、仅MOVEFILE_WRITE_THROUGH无覆盖发布；门槛及API在worker前运行（182-280、2688-2750）。|
|P2-01 final成本receipt边界|未见回归|结果标注finalization-start成本，独立receipt注明result/receipt publication excluded并记录reserve（3392-3459）。|
|P2-02 reserve算术|未见回归|固定reserve为4×120+156×60+60+300=10,200；预算函数按156生成和4次恢复计算（82-98、1040-1053）。|
|P2-03 selector durable意图与unknown tail|未见回归|`prompt_selection_started`、`prompt_face_persisted`先后持久化；中断审计保留selector unknown tail（2381-2458、3115-3143）。|
|156分母、validate-only边界|未见回归|完成门仍要求156 attempts/outputs；validate-only只运行metadata preflight与readiness写/核验（1012-1021、3293-3294）。|

## 验证与回流判定

- 根任务报告40项synthetic合同测试通过；本审查独立定向运行8项通过、32项deselected，6.07秒。相关范围为IPC deadline/超界、worker授权、source drift、Job assignment/resume与未确认终止。
- 未运行真实worker、模型、selector、dev/reference或`--execute`；未修改实现代码、训练数据或冻结合同。
- 回流结论：P1-07仍未关闭。补齐writer状态传递、未知query尾账和对应synthetic故障注入后，才可再判Phase 7放行；真实156次评测继续冻结。P2-04关闭，其余旧P1/P2未见回归。
- 历史2/2回流预算与第二轮打回结论保持原样。本报告记录新提交的第三轮独立复核，不改Phase 8顶部主指针。