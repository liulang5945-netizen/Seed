# PLAN-N1-153A：N1-153逐题结果绑定修复与同面完成

**状态：已依据用户授予的执行计划批准权批准并冻结。**N1-153一次性查询在seed11/length_mixed endpoint完整执行480条后停止：summary helper误把单endpoint行表当作六endpoint全集校验，因此抛错发生在将480行追加到最终结果之前。N1-153原result记录`query_count=480`、`diagnostic_rows=0`、0更新，final checkpoint前置6/6通过；result SHA为`cabd36441b8619875f6d17c70f0114f64c7986bb5abb9b1008071b07dd3be029`。原N1-153 face、attempt和停止result全部保留，不修改。

## 唯一修复与冻结范围

- 不生成、不改写、不重排N1-153 seed127 face；继续使用已封存的160题、四类各40、8/16/32行嵌套上下文和原来源排除审计。
- 新增独立runner，逐endpoint汇总当前endpoint的480行；逐题行先写入内存结果，再计算summary，停止时也保留已完成行。
- 为获得一致的六endpoint完整报告，N1-153A在原face上按原顺序重新执行全部2,880条查询。其中seed11/length_mixed的480条将与N1-153失败attempt重叠，另外2,400条是原失败前未执行部分。累计查询为3,360条；不新增题面、内容或模型endpoint。
- 六个N1-150 final full bundle须再次独立恢复6/6；测量前后fingerprint相同。结果逐endpoint保存480条输出及12个长度×task kind汇总；训练无更新。
- CPU≤3,600秒，累计包含N1-153失败result中记录的113.78125秒、N1-153A readiness、六bundle恢复及2,880条查询。任何SHA、face、endpoint分母、summary或CPU守卫失败即停，不再重试。
- 0 dev、holdout、confirmation查询；不改变既有门，不作独立确认或能力晋级结论，`can_promote=false`。

## N1-153A产物

- Runner：`scripts/training/run_taiji_n1_plan153a_fresh_context_result_binding_recovery.py`
- 合同测试：`tests/taiji_native/test_n1_plan153a_fresh_context_result_binding_recovery_contract.py`
- Readiness、六final restore preflight、attempt和逐题result使用`reports/taiji_n1_plan153a_..._20261020.json`独立write-once路径；不覆盖N1-153原结果。
