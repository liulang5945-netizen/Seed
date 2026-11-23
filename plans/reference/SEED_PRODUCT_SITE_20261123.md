---
title: Seed 产品官网与首次部署记录
summary: 记录 2026-11-23 新增的 Seed 中文静态产品官网、源码位置和首次发布身份。
read_when: 需要维护、重新部署或核对 Seed 中文产品官网时阅读。
---

# Seed 产品官网与首次部署记录

## 结论

2026-11-23 新增 Seed 中文产品官网并完成首次静态部署。该网站是项目展示与快速开始入口，不改变训练主线、认知架构合同、实验状态或能力边界。

## 源码与内容

- 应用目录：`design/seed-product-site/`
- 主入口：`design/seed-product-site/index.html`
- 品牌资产：`design/seed-product-site/seed-mark-color.svg`，来自既有 `design/logo/seed-mark-color.svg`
- 技术形态：无构建依赖的 StaticWeb，包含响应式布局、移动导航、认知架构选项卡、滚动呈现与代码复制交互
- 内容边界：依据 `README.zh-CN.md` 描述研究目标与当前原型，不宣称已经实现完整通用认知能力

## 首次发布身份

| 项目 | 值 |
|---|---|
| Site 名称 | Seed-产品官网 |
| Site ID | `37lghjpm5rwqclfynjb` |
| Deployment ID | `dep_6b03eb61e8b851344a8c7afd` |
| URL | https://37lghjpm5rwqclfynjb.traesite.cn |
| 状态 | 发布成功 |
| 可见性 | 仅自己可见 |

后续重新部署应复用上述 Site，并从当前源码重新准备静态发布文件；不得通过本记录推断训练或模型能力发生任何变化。
