# Agent 记录：聊天内联本地媒体按两种拼法转发绝对路径

状态：已实现

[English](2026-09-30-chat-inline-local-media-either-spelling.md) | 中文

## 问题

`AssistantMarkdown` 只在图片目标以 `/` 开头时才解析，因此 Windows 绝对路径（`E:\...`）在聊天里渲染成静态文本，而同一份文档在 Sidebar 里能正常预览。两个表面对同一个字符串问了不同的问题：Sidebar 用平台感知的 `isAbsoluteWorkspacePath`，聊天用只认 POSIX 的前缀测试。[本地媒体展示记录](2026-09-07-session-prose-local-media-display.zh.md)把 Windows 风格路径列在有意保持静态的形状之中；本记录取代该列表项，该记录继续保留渲染器归属、相对所服务文档来源挂载的解析规则及其理由。

## 决定

`localPathMediaUrl` 现在沿用 `markdownImageUrl` 已有的判定顺序：先拒空值、含 NUL 的值、以及以两个任意分隔符开头的目标；再要求 `isAbsoluteWorkspacePath`，它同时接受 POSIX 绝对路径与 Windows 盘符形式；最后要求文档基址为 HTTP(S)。UNC 目标在两个表面都不在范围内，因此同一份文档在聊天与 Sidebar 得到同一组图片。

浏览器无法得知 Host 所在系统，所以渲染器不做判断：绝对路径一律转发，由 Host 接受或拒绝。`GET /api/file` 的门槛是 `node:path` 的 `isAbsolute` 加上所组合提供方的读取策略——这正是盘符路径在 Windows Host 上成功、在 POSIX Host 上返回 400 的原因，而此前它在 POSIX 上根本不发起请求。对 POSIX 用户的可见结果不变——图片仍不显示并保留 alt 文本——多出的代价是一次被拒绝的请求。

## 执行保障

`assistant-markdown-path-images.client.spec.tsx` 两个方向都钉住：盘符形式在根基址与挂载基址下都解析成 URL，而空值、双分隔符、相对路径、非 HTTP 传输保持静态，于是这次改宽不会悄悄吸收一个从来不是本地文件的形状。`markdown-images.e2e.ts` 用所在平台的分隔符书写目标，使被请求的键与断言取值的键在任何 Host 上都一致。`ui-deliverables` 的 README 双语对写明仍然存在的页面限制（`dsh-app:` 仍不渲染）。
