/**
 * 预加载脚本 —— 把 desktop/main.py 的 _WindowBridge 以「前端零改动」的形态重建。
 *
 * 前端契约**逐字读自** frontend/src/components/AppTitlebar.vue（不得凭记忆改写）：
 *
 *   const transport = window.qt?.webChannelTransport
 *   if (transport && typeof window.QWebChannel === 'function') {
 *     new window.QWebChannel(transport, (channel) => {
 *       const obj = channel.objects?.seedWindow
 *       if (!obj) return
 *       ...
 *     })
 *   }
 *   // 随后调用 obj.startDrag() / obj.minimize() / obj.toggleMaximize() / obj.close()
 *   // 最大化状态另有 document.documentElement 上的 data-maximized 属性（主进程回写）
 *
 * 所以这里必须复刻 @qt/qwebchannel 的**形状**，而不是另设一套 IPC API —— 后者会迫使
 * AppTitlebar.vue 改动，而本次迁移的硬约束正是「零 .vue 改动」。
 *
 * 该层能否生效由主进程在 dom-ready 阶段做显式自检并记日志（见 main.ts verifyBridge）：
 * 若 contextBridge 对「类构造器 / 回调内传函数」的支持不如预期，日志里会立刻有一条明确
 * 的 error，而不是表现为标题栏三个按钮静默消失。
 */

import { contextBridge, ipcRenderer } from 'electron'

const WINDOW_COMMANDS = {
  minimize: 'minimize',
  toggleMaximize: 'toggle-maximize',
  close: 'close',
  startDrag: 'start-drag',
} as const

type WindowCommand = (typeof WINDOW_COMMANDS)[keyof typeof WINDOW_COMMANDS]

function send(command: WindowCommand): void {
  ipcRenderer.send('seed:window', command)
}

/** 与 Qt 侧 seedWindow 对象同名同形的方法集。 */
const seedWindow = {
  minimize: (): void => send(WINDOW_COMMANDS.minimize),
  toggleMaximize: (): void => send(WINDOW_COMMANDS.toggleMaximize),
  close: (): void => send(WINDOW_COMMANDS.close),
  startDrag: (): void => send(WINDOW_COMMANDS.startDrag),
  isMaximized: (): boolean => ipcRenderer.sendSync('seed:window:is-maximized') === true,
}

interface QWebChannelChannel {
  objects: Record<string, unknown>
}

/**
 * QWebChannel 的最小兼容实现。
 *
 * Qt 的真品会走 websocket 传输建立通道；这里前端只用到「构造时回调 channel，
 * channel.objects.seedWindow」这一条路径，因此直接同步回调即可，不需要 transport。
 * 写成函数而非 class：contextBridge 对函数（含可 new 的构造器）的支持是明确保证的，
 * 且函数内部不使用 this，new 与直接调用等价。
 */
function QWebChannel(_transport: unknown, callback: (channel: QWebChannelChannel) => void): void {
  callback({ objects: { seedWindow } })
}

contextBridge.exposeInMainWorld('qt', { webChannelTransport: { impl: 'seed-electron' } })
contextBridge.exposeInMainWorld('QWebChannel', QWebChannel)
