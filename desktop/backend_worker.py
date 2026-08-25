"""
[打包配套] Seed 后端工作进程入口
=================================

打包（frozen）模式下，主 GUI 进程（Seed.exe）用本脚本打包出的
`SeedBackend.exe` 拉起后端，等价于开发模式的
`python -m uvicorn api.app:app`：

- 独立进程 → logging 配置与 GUI 入口天然隔离
  （GUI 入口已 basicConfig 过 root，uvicorn 再 dictConfig 同名
  formatter 'default' 会直接报错退出）；
- 独立进程 → 冷启动导入不与 PyQt 事件循环争 GIL；
- 控制台 exe → stdout/stderr 可见，主进程将其重定向到日志文件。

用法（由 desktop/main.py 自动调用）：
    SeedBackend.exe <host> <port>

开发模式不使用本入口（直接 `python -m uvicorn`）。
"""

import sys


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    import uvicorn

    from api.app import app

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
