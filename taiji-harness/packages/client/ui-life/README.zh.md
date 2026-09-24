---
description: "Web 侧栏的生命面板：Taiji 运行时的器官读数、训练控制与宿主投影，聚合在一个全局页面里。"
kind: "package-reference"
---

# @taiji/dsh-client-ui-life

[English](README.md) | 中文

## Summary

点开 Web 侧栏的**生命**入口，检查并驱动 Taiji 本地运行时。六个分区渲染最新读数——来源与新鲜度、生命器官、训练（含检查点清单）、知识库、记忆与巩固产物、宿主投影——并携带控制组：调度启停、强制喂食/睡眠/玩耍，带可勾选数据集名册的训练启动、从任一在列检查点续训、暂停、继续、停止与强制解锁，以及一次巩固 pass。数字全部来自 controller 的快照流；拒绝以 Host 的稳定错误码呈现，绝不吐裸 RPC 文本。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)

-----

<a id="use-this-package"></a>
## Use this package

把本插件挂进一个 Host 携带 [`@taiji/dsh-api-life-controller`](../../api/life-controller/README.zh.md) 的 web 组合——面板读取 `ctx.life`（由 controller 的浏览器半边安装的 Client 门面），并在 `dsh.client` manifest 里声明这一依赖。它只贡献一个侧栏入口与一个主面板，别无其它。

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

页面以 `useSyncExternalStore` 订阅 controller 的身份稳定快照态：一帧流到达即换一个对象，面板随之重渲染。六个分区投影同一份 `LifeSnapshot`：来源分区指出应答的器官并列出每个未应答者；生命分区渲染原生器官的需求与驱力条形，或传统调度器的事实与条形；训练分区显示状态徽标、最新进度样本与 runtime 自己的运行警告（例如续训时的语料漂移提示）、可勾选的数据集名册——只要数据环规格仍点名盘上文件即按其预选——以及检查点清单：每行提供续训与激活两个动作，在用的检查点带徽标，「当前应答的」与「下次启动将用的」不一致时如实列出；知识分区只在门控面应答时显示索引规模；记忆与巩固分区显示按类型的记忆日志计数、巩固趟数与最近语料、就绪门背后的数据环规格，以及最近一次 pass 报告——若该面未被提供，则如实说明而不显示空白；宿主分区显示健康、模型、Seed 活跃与内存。

控制按钮在动词执行期间禁用，且其结果不触碰快照：被接受的动词经流自然刷新，被拒绝的把 Host 的错误码——`life/conflict`、`life/unavailable` 等——翻译成面板文案。停止训练、强制解锁与切换应答模型都需要第二次点击确认；运行巩固 pass 不需要这样的确认，因为它只写排练语料，绝不动权重。

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
- 知识分区按面板合同为只读；上传、重建与清除属于运行时自己的表面。
- 轮询间隔在此只展示不可编辑；它是 controller 配置的属性。
