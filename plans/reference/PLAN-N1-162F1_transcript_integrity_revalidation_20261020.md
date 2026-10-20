# PLAN-N1-162F1：transition交叉探针输出与来源复核

**状态：已依用户授予的执行计划批准权批准并冻结**<br>
**日期：2026-10-20**<br>
**原测量计划：** [PLAN-N1-162F](PLAN-N1-162F_transition_template_crossover_train_diagnostic_20261020.md)<br>
**分析依据：** [PLAN-N1-162E结案](PLAN-N1-162E_execution_closeout_20261020.md)

## 触发证据

F路径已存在完整result/progress：6,912/6,912条train-only输出、九endpoint独立恢复9/9、0更新、0新面；旧结果SHA-256 `bb42a2017a7935c600a5b91c43025aa2381dedf7a9335ed86f071283c52498d8`，progress SHA-256 `baed16dc5cad86e6ceabcf457f5cddec561f89e65ba370666e9b1417ed5d47f6`。但result记录的F runner SHA `10e1772f2c6bafc3ef3feb481a923dffa327f14b6fdb64d81c7074dfc3681999`与当前runner SHA `a6faaa1405f3239a671534b9666cb52201d635edc613206c44ba326e78d4594c`不同；原脚本内容未在result中打包。原result与progress已保留，不覆盖、不重查。

## 复核范围

只读取F的attempt/result/progress与source hashes，重建九endpoint、两view、四factorial条件、两模板的预期query key，要求6,912个key完整唯一且每行的question、expected answer、initial direction、turn、template id都与训练输入及factorial helper一致。再独立重算raw output类别和`free_exact`，与封存字段逐行核对；检查progress每一行与result `output_rows`逐字段匹配。对source map复核除旧F runner路径外的所有hash，明确报告旧/当前runner SHA差异；复核九个checkpoint hashes、恢复receipt和fingerprint。

本复核只判定现存逐题输出是否满足冻结题面、分母、答案重建及复算合同，不能恢复缺失的旧runner源码，也不把旧result升级为当前runner的重新执行。若出现除runner以外的source drift、query key缺漏/重复、expected answer或free-exact不一致，均停止并报告。

## 资源与边界

CPU≤120秒，关联CPU基线21,636.844秒、硬上限36,000秒。0模型/checkpoint加载、0 inference/query、0参数更新、0新面；仅哈希核验与离线JSONL复算，不修改原attempt/progress/result，`can_promote=false`。

## 产物

- 复核器：`scripts/training/audit_taiji_n1_plan162f_transcript_integrity.py`
- 契约测试：`tests/taiji_native/test_n1_plan162f1_transcript_integrity_contract.py`
- 报告：`reports/taiji_n1_plan162f_transcript_integrity_20261020.json`
