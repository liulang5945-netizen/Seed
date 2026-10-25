# SPEC-N1-172：fit-first 探针的训练、早停与判据

日期：2026-10-25。真实更新前冻结，承接[PLAN-N1-172](PLAN-N1-172_fit_first_exposure_20261025.md)。本包只验证「训练侧拟合后，留出题是否出现题-答绑定」；不是正式算法胜负、不是 L2、不是默认采用。

## 1. 选择与隔离

1. 语料仍只来自冻结的 assistant train 加载函数；选择函数与 N1-171 相同，且 selection digest 必须等于 `5db33945cd22ce71d70d8d97f6af72462d4871aa239559876e79cd1e1edd6b40`；不等即工程失败，不重新选题。
2. 只使用秩 ≡0（折 0）与秩 ≡1（折 1）的 8 个留出题；每折训练侧为另外 12 题。留出与训练侧在 prompt/target 前缀 SHA256 上不得相交。
3. `dev_queries=0`；默认模型、旧 parent、旧 native 入口字节不变；`can_promote=false`。

## 2. 训练与早停

- 模型 `ConditionalLanguageCandidateV2`（v2，970,945 参数），`TaskCoreConfigV2()` 默认，seed 169，bundle 名含折号。
- 每折 12 题整批、target 前 128 byte、AdamW lr 0.001、wd 0、betas/eps 默认、clip 1；更新上限 176。
- 早停只用 12 个训练题：在更新数 32、48、…、176 测 TF top-1 并取中位数；首次 ≥0.90 即停止。留出题不参与停止判定。
- 停止后记录该折 16 题（12 训＋4 留出）的 TF/NLL/完整汉字/自由 128 byte；留出题对照 A（同折下一道留出 target，末尾回绕）、B（训练 target `(3i+1)%12`）与错配 target 的 TF。
- 0 步读数只测一次（两折零模型由同一 seed 构造，逐位相同），报告中标注其为共享零读数。

## 3. 存档与恢复

- checkpoint 沿用 `taiji-conditional-language-trainer-v2`；保存 `zero.pt`（折 0 更新前）与 `fold0_final.pt`、`fold1_final.pt`（各自停止时的更新数）。全部 write-once，不覆盖 N1-171 存档。
- 每份存档在**新进程**恢复并核对 checkpoint digest、参数 revision/steps/曝光、下一生成步 logits digest 与 15 byte 继续生成。

## 4. 判据与出口

- 前置 P：两折训练 TF 中位数均 ≥0.90。
- 主判据 G：8 个留出题中 own 自由匹配 ≥0.20 的题数 ≥4，且这些题 own ≥对照 A +0.15 且 ≥对照 B +0.15。
- 出口：G 成立 `PASS_BINDING`；两折均拟合但 G 不成立 `FAIL_NO_GENERALIZATION`；任一折未拟合 `INCONCLUSIVE_UNDERFIT`。
- 次判读 S：逐折 ≥0.20 题数、own/错配 TF 均值、A/B 均值、不同自由输出数、训练题自由匹配分布；只报告，不设门。

## 5. 失败与资源出口

- 工程失败（选择 digest 不符、相交、存档/恢复、来源变化）先修可逆实现；修复不得改选择或判据。
- 费用上限 720 CPU / 1,300 wall（脚本内部 700/1,240）；触顶保存 partial、failure 与真实费用，不续跑、不换配方。
- 报告 write-once；已存在即拒绝重跑。
