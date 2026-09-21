"""
[打包配套] Seed WebSocket 核心服务器工作进程入口
=================================================

8765 的 WebSocket 服务器在两种模式下需要不同的承载方式：

- PyQt6 侧（desktop/main.py）的 frozen 分支用**进程内守护线程**跑它——Python 进程
  可以直接 import 该模块，不需要额外进程；
- Electron 侧（desktop-electron/）是 Node 进程，无法 import Python 模块，必须有一个
  可独立执行的子进程入口，即本脚本打包出的 `SeedWs.exe`。

等价关系：本入口等价于开发模式的 `python -m neuroplex.core.websocket_server`
（见 desktop/main.py 的非 frozen 分支）。

用法（由 desktop-electron/src/websocket.ts 自动调用）：
    SeedWs.exe [port]

开发模式不使用本入口。本脚本不修改 neuroplex 侧任何行为，只做进程入口。
"""

import asyncio
import logging
import sys

DEFAULT_PORT = 8765


def main() -> None:
    # host 保持 neuroplex.core.websocket_server.start_server() 的默认值 "localhost"，
    # 与 PyQt6 侧 frozen 分支的 `asyncio.run(start_server())` 完全一致；
    # 这里只把 port 开出来，便于由 desktop-electron/src/config.ts 单一来源驱动。
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    from neuroplex.core.websocket_server import start_server

    asyncio.run(start_server(port=port))


if __name__ == "__main__":
    main()
