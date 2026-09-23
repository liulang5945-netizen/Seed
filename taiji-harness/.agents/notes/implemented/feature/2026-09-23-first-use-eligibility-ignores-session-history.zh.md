# Agent Note: 首次使用资格不再看 Session 历史

Status: implemented

[English](2026-09-23-first-use-eligibility-ignores-session-history.md) | 中文

## Problem

注册表里一个 Workspace 都没有、磁盘上却留着 Session 历史时，产品不可用。Client 只在 Session 列表为空时才准备默认 Workspace，Host 只在 live、持久化、已归档 Session 都不存在时才允许，而目录选择器不可用的 Host 没有任何其他途径拿到 Workspace。普通删除就会到达这一状态：删除 Workspace 按契约保留其会话日志，于是注册表空了而历史还在。

## Decision

资格判断改为「注册表里没有任何 Workspace」。live、持久化与已归档的 Session 一律不否决准备，Client 也不再要求 Session 列表为空才发起请求。无法加入新 Workspace 的 Session——工作目录不同或根本没有——保持未分组，归档集合不受影响。注册表中已存在 Workspace 时仍然拒绝，因此默认 Workspace 只是唯一的恢复手段，不会变成第二个登记项，请求中的名称也依旧不会重命名既有登记。

去掉这项否决也一并去掉了目录准备期间的两处 Session 复查。注册表的修改队列本就串行化每一次 Workspace 写入，资格判断与提交之间不可能出现新的 Workspace；而在准备期间启动的 Session 只是一个普通的未分组 Session，不再构成放弃这次合法恢复的理由。

## Alternatives considered

- 保留否决、依赖目录选择器，会在选择器不可用的环境里原地留下死路。
- 重置初始化标记、让下次启动从持久化 header 重新推导 Workspace，会复活用户已删除的登记——因为删除按契约保留会话日志。
- 只放宽 Client 侧的前置毫无作用：Host 会因同样的历史拒绝。

## Consequences

首次启动可能在与既有未分组历史并存的情况下创建默认 Workspace，侧栏仍像以前一样把这些 Session 列在任何项目之外。[Workspace registry](../../../../packages/workspace/workspace/README.zh.md#first-use-workspace)拥有当前的资格契约。Registry 与 Client 测试覆盖持久化、live、已归档三种情形、注册表清空后重新获得资格，以及在缺少 Session store peer 时仍能初始化。

[首次使用默认 Workspace](2026-09-20-default-workspace.zh.md)的资格条款已被取代；其目录解析、持久身份与恢复行为继续有效。