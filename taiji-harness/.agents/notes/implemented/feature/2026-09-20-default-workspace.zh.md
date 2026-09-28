# Agent Note: 首次使用默认 Workspace

Status: implemented

[English](2026-09-20-default-workspace.md) | 中文

## Problem

新安装的应用要求用户先选目录才能发送第一条消息。去掉这一步前置操作时，必须保留 Session 的固定工作目录，也不能把隐藏或已归档的历史误判为新安装。

## Decision

启动时等待完整的 Workspace 和 Session 基线。Workspace 列表为空时，Client 请求 Host 准备默认 Workspace；它会复用一条仍符合条件的已保存空白 Session，否则让该 Workspace 保持关闭，因此[启动不会打开任何 Session](2026-09-29-startup-never-opens-a-session.zh.md)，第一条 Session 由 Hero 的发送动作拥有。Session 存在前输入框是 Hero 的 composer，之后使用常规 composer 流程。[Session scope 与供给通道](../architecture/2026-07-25-web-client-session-scope-and-provide-channel.zh.md)负责 blank Session 复用与供给通道的理由。

[Workspace registry](../../../../packages/workspace/workspace/README.zh.md#first-use-workspace)负责资格判断和目录准备：资格即注册表为空，且 [Session 历史不再否决准备](2026-09-23-first-use-eligibility-ignores-session-history.zh.md)。操作运行在注册表修改队列内，该队列串行化每一次 Workspace 写入。

Client 在启动时按其语言解析初始目录名和标题。Host 控制器解析 Documents 位置；注册表接收不依赖 locale 的目录解析器。解析器仅在允许创建时于变更队列内运行，因此重复请求直接复用持久化的 Workspace，无需再次查询操作系统目录。持久化 Workspace id 独立于名称记录初始化成功。改名或切换语言不能再次初始化默认工作区。删除登记会在同一次写入中清除该身份，启动时也会清除「Workspace 已不在表中」的持久身份，因此后续满足条件的准备会重建一个默认工作区，而不是留下「无可选工作区」。标记与登记一起提交，因此登记失败可以重试。目录内容仍遵循已有的[仅删除元数据策略](2026-07-27-workspace-registration-deletion.zh.md)。

不符合首次使用条件时不返回 Workspace，也不显示失败弹窗。准备失败时提供现有文件夹选择器；后续列表通知不会触发启动流程重试。成功的登记会在 Session 创建或提示词发送失败后保留。选中默认或手动选择的 Workspace 均不会提交消息。

## Alternatives considered

- 仅在发送时创建可以避免未使用的目录，但需要独立的本地草稿、向 Session 转移草稿及首次提交协调逻辑。启动时创建接受提前写入文件系统的副作用，始终使用普通 Session 输入。
- 系统查询不可用时回退到 `<home>/Documents`，会选择未经确认的 Documents 位置；需要其他目录的部署使用显式 Host 覆盖配置。
- 根据侧栏可见行推断首次使用，会忽略已归档、隐藏和无 cwd 的 Session。
- 将本地化路径作为初始化标记，会允许切换语言或删除后再次创建默认工作区。
- 在 Session 创建后更改 cwd，会改变其已记录工具和附件的含义。

## Consequences

打开新安装的应用即可在用户发送消息前创建目录。目录授权与准备失败提示可能在启动期间出现。Desktop 和远程 Web 客户端均由 Host 准备目录，因此远程用户使用 Host 账户的 Documents 位置。Documents 查询不可用时，采用与目录创建失败相同的文件夹选择恢复路径。

该功能保留[由引用持有的 Client Session 生命周期](../architecture/2026-09-15-client-session-references.zh.md)，无需修改 agent-loop 或 Session 事件。Registry 和客户端测试覆盖重试、隐藏历史、并发及启动取消；[浏览器场景](../../../../apps/web/tests/default-workspace.e2e.ts)通过隔离的 Documents 目录和已录制 Session 回放验证完整装配下的启动、刷新、首次发送与选择器路径。
