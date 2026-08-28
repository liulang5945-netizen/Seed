"""Seed桌面客户端

这里刻意不导入 main：包 __init__ 里 `from desktop.main import main` 会让
`python -m desktop.main` 把同一份源码执行两次（先作为 desktop.main 被包
__init__ 载入 sys.modules，再由 runpy 作为 __main__ 重新执行）。后果不只是
日志 handler 装两遍导致每行重复，更严重的是模块级全局会出现两份副本
（如子进程回收用的 Job Object 句柄），排查时表现为「机制明明正确却像没生效」。
需要入口时请直接用 `python -m desktop.main` 或导入 `desktop.main`。
"""
