# 增量需求

## 结论
R-01 验证06/07完整结果、源绑定、checkpoint恢复和更新量后才能访问dev prompt。
R-02 复用06已保存39条prompt及两端点输出，仅为07两臂各生成一次；不加载reference response。
R-03 输出逐条先落盘，write-once保护；结果记录全部查询、来源hash、恢复回执和CPU开销，can_promote=false。
R-04 人工样例固定同一12 families；格式统计不得冒充语义评价。

## 验收标准
AC-R-01 缺失或partial训练结果在任何prompt访问前拒绝。
AC-R-02 prompt身份不一致、重复family、hash漂移拒绝。
AC-R-03 模拟运行验证78次查询、156条组合输出及失败留证，测试不读取真实dev。
AC-R-04 无训练更新；原checkpoint文件hash保持不变；执行预算明确且fail closed。

## 遗留风险
所有目标依旧是局部研究证据，不自动晋升模型。

## 给下游
审查现有runner实际字段再定义接口，禁止猜测checkpoint和结果格式。
