---
title: Seed 产品官网与部署记录
summary: 记录 Seed 中文静态产品官网、技术内容、GitHub 入口、源码位置和部署身份。
read_when: 需要维护、重新部署或核对 Seed 中文产品官网时阅读。
---

# Seed 产品官网与部署记录

## 结论

2026-11-23 新增 Seed 中文产品官网并完成静态部署；同日补入真实 GitHub 仓库入口、TSK-v8 单步算法流程、稀疏学习公式、学习/生成/恢复调用链和源码深链。该网站是项目技术与研究展示入口，不改变训练主线、认知架构合同、实验状态或能力边界。

## 源码与内容

- 应用目录：`design/seed-product-site/`
- 主入口：`design/seed-product-site/index.html`
- 品牌资产：`design/seed-product-site/seed-mark-color.svg`，来自既有 `design/logo/seed-mark-color.svg`
- 技术形态：无构建依赖的 StaticWeb，包含响应式布局、移动导航、认知架构选项卡、算法单步解说、滚动呈现与代码复制交互
- 技术资源：`technical.css` 与 `technical.js` 独立承载新增技术区域，避免继续扩大单文件页面
- GitHub 仓库：https://github.com/liulang5945-netizen/Seed
- 技术依据：`TAIJI_CODE_WIKI.md`、`seed/model.py`、`taiji/model.py`、`taiji/sparse.py`、`taiji/organs.py` 与 `plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md`
- 内容边界：明确区分当前 TSK-v8 可执行实验内核与完整 Taiji 长期研究目标，不宣称已经实现完整通用认知能力

## 当前部署身份

| 项目 | 值 |
|---|---|
| Site 名称 | Seed-产品官网 |
| Site ID | `37lghjpm5rwqclfynjb` |
| 最新 Deployment ID | `dep_6b03efba7d046a344f160e25` |
| URL | https://taiji-seed-liulang.traesite.cn |
| 状态 | 技术版发布成功，截图已刷新 |
| 可见性 | 所有人可见 |

首次展示版 Deployment 为 `dep_6b03eb61e8b851344a8c7afd`；技术原理与 GitHub 源码版 Deployment 为 `dep_6b03efba7d046a344f160e25`。后续重新部署应复用上述 Site，并从当前源码重新准备静态发布文件；不得通过本记录推断训练或模型能力发生任何变化。
