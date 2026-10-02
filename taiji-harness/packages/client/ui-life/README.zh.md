---
description: "Web 侧栏的生命面板：Taiji 运行时的器官读数、训练控制与宿主投影，聚合在一个全局页面里。"
kind: "package-reference"
---

# @taiji/dsh-client-ui-life

[English](README.md) | 中文

## 概述

点开 Web 侧栏的**生命**入口，检查并驱动 Taiji 本地运行时。六个分区——来源、生命器官、训练（含检查点）、知识、记忆与宿主——渲染最新读数并携带控制组：调度启停、喂食/睡眠/玩耍、按可勾选数据集名册启动训练并可上传与删除、检查点续训与激活、暂停、继续、停止与强制解锁、一次巩固 pass，以及知识文件上传与删除。数字全部来自 controller 的快照流；拒绝以 Host 的稳定错误码呈现，绝不吐裸 RPC 文本。

## 目录

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

把本插件挂进一个 Host 携带 [`@taiji/dsh-api-life-controller`](../../api/life-controller/README.zh.md) 的 web 组合——面板读取 `ctx.life`（由 controller 的浏览器半边安装的 Client 门面），并在 `dsh.client` manifest 里声明这一依赖。它只贡献一个侧栏入口与一个主面板，别无其它。

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

页面以 `useSyncExternalStore` 订阅 controller 的身份稳定快照态：一帧流到达即换一个对象，面板随之重渲染。六个分区投影同一份 `LifeSnapshot`：来源分区指出应答的器官并列出每个未应答者；生命分区渲染原生器官的需求与驱力条形，或传统调度器的事实与条形；训练分区显示状态徽标、最新进度样本与 runtime 自己的运行警告（例如续训时的语料漂移提示）、可勾选的数据集名册——按目录分组、点击分组即折叠，只要数据环规格仍点名盘上文件即按其预选——连同上传入口与“删除所选数据集”动作，以及检查点清单——整块可折叠、带一列用于“删除所选检查点”的勾选——每行提供续训与激活两个动作，活跃与已配置的那两枚不可勾选，在用的检查点带徽标，「当前应答的」与「下次启动将用的」不一致时如实列出；知识分区显示索引规模与已挂载文档及其 `indexed` 或 `pending` 状态，提供上传入口并可删除所选文档——若该面未被提供，则如实说明而不显示空白；记忆与巩固分区显示按类型的记忆日志计数、巩固趟数与最近语料、就绪门背后的数据环规格，以及最近一次 pass 报告——若该面未被提供，则如实说明而不显示空白；宿主分区显示健康、模型、Seed 活跃、工作台能力快照（数量、修订、来源与归属——快照报错时逐字带其失败文本）、运行时鉴权态与内存。

控制按钮在动词执行期间禁用，且其结果不触碰快照：被接受的动词经流自然刷新，被拒绝的把 Host 的错误码——`life/conflict`、`life/unavailable` 等——翻译成面板文案。停止训练、强制解锁、切换应答模型以及每一类删除都需要第二次点击确认；删除类动词把确认文案直接写在各自的按钮上，而运行巩固 pass 不需要这样的确认，因为它只写排练语料，绝不动权重。

-----

**运行时不变式：** 不发布伴随包。本包只在 life controller 的 Client 门面已携带的读数与控制动词之上提供一个浏览器面板。

-----

<a id="model-experience"></a>
## Model Experience

本面板只渲染面向操作者的读数与控制，不向模型请求添加任何输入。

#### KV Cache effect

没有；页面面向操作者渲染，不向任何模型请求贡献内容。

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- 启动训练的数据集由面板名册勾选；参数预算、种子与符号上限仍用运行时默认值，表单留待出现消费方再做。
- 知识分区可上传与删除文档；索引重建与整体清空属于运行时自己的表面。
- 轮询间隔在此只展示不可编辑；它是 controller 配置的属性。

<a id="dev-note"></a>
### 开发备注

<details>
<summary>维护者工作上下文——点击展开</summary>

面板不保存任何读数副本：每一格都从 controller 的当前快照帧投影，被接受的动词也是经流刷新，而不是先改本地状态。运行时未服务的分区以文字说明，而不是渲染成零值——把滞后读数呈现为当前值，正是本面板要避开的失败形态。

</details>
