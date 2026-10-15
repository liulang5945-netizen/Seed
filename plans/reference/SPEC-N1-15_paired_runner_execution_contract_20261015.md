# SPEC-N1-15 · 已批准14配对小试执行合同（2026-10-15）

用户直接批准14的三seed、三臂、共3456次episode更新、独立CPU上限5分钟，并授权此类低成本实验以后自主开展。原14方案与行为门保持不变，本件补充运行器机械纪律，非改测面或降低门槛。

运行器 `scripts/training/run_taiji_n1_native_evidence_paired.py` 默认只检查manifest/配方/代码身份，无optimizer步骤、无模型查询；仅 `--execute-approved` 启动本次获批正式包。seed11/12/13，position/terminal/cut，每模型384更新，固定train96轮转4轮，lr0.001 Adam，16/32/16维，UTF8与boundary256、cap64。训练目标仅损失接口可见；推理投影只保留观察文本和问题，不接题型/金答案/评测ID。

## 执行与证据

1. 正式开始保存全部配方、v2 manifest SHA、源码SHA、Python/Torch版本。v2 manifest必须为 e95a2e3be6a3a4abca92ea583cf35522545961280b575e5d5373857973d5e979；代码在评价前后复核，不允许运行中修补。
2. 每seed三臂初始参数摘要逐位相同，学习和推理均使用各自臂：每来源末态仍保留全部来源，cut只切断写入状态信用。每episode使用独立会话，观察来源/timestamp稳定，问题和生成不会写回事实。已有训练会话保留在候选bundle中；不是只存权重。
3. 每模型任何更新前保存zero checkpoint，独立新进程weights-only恢复，对**父进程存档前实际状态**核对参数/optimizer/来源/RNG与train问题第一步logits；不是两个恢复副本互相比对。全部更新后再做一次相同独立审计，含非空Adam状态。worker零参数更新，正式更新总数不超过3456。13恢复夹具证明的续步能力不能代替这18次实际包审计。
4. 九模型全部训练与恢复审计完成、权重/源码冻结后，才运行末尾dev/holdout各32题。每题严格raw byte exact（同时记录UTF8安全输出），自然boundary和cap分别记录；UTF8尾截断不能伪装自然停止。所有576条原始结果入报告，无中途dev选参或holdout重跑。
5. CPU预算累计父进程process_time（含import）和各restore worker时间；worker由psutil采样监控，每次采用worker报告与采样最大值。单Torch/BLAS线程，保留2秒结束存档余量，触边即停并保存partial，报告budget-stop而非缩分母；计时不完整/超限不得放行。末尾序列化端点限制明确记录。不借C2预算，自动重跑/续跑禁止。
6. checkpoint走 `get_external_path('data/checkpoints')` 下的独立正式包子目录，受既有data根台账和ignore保护；不散建根目录或覆盖母档。目录与报告独占创建，已存在则拒覆写。完整报告名 `taiji_n1_native_evidence_paired_20261015_final.json`，放reports；不在执行前冒充存在的读数。

## 原门与后续

position在每seed holdout至少16/32、每类至少2/8，三seed平均分别高于terminal和cut至少4/32；9×384更新及9×64评价全部完整、18次恢复通过、源码身份未漂移、预算计量完整且≤300秒才可进入后续保持/联合能力评估。不是默认采用、C4或N主线收官。任何失败先记录主要失效，下一步必须针对证据，不改门追绿。
