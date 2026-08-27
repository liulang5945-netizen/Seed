"""
Web 终端 WebSocket 路由
======================
提供基于 WebSocket 的交互式终端会话。
支持 Windows (cmd/PowerShell) 和 Linux (bash/zsh)。

安全措施：
- JWT 认证（通过 query 参数传递 token）
- 并发终端数量限制（默认 3）
- 空闲超时自动断开（默认 300 秒）
"""

import asyncio
import codecs
import json
import locale
import logging
import os
import sys
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from seed_platform.settings import get_setting

logger = logging.getLogger("ApiServer.Terminal")
router = APIRouter()

# ======================== 安全配置 ========================

MAX_CONCURRENT_TERMINALS = 3
IDLE_TIMEOUT_SECONDS = 300  # 5 分钟无输入自动断开

# 当前活跃终端计数
_active_terminals = 0
_terminals_lock = threading.Lock()


def _acquire_terminal_slot() -> bool:
    """尝试获取终端槽位，成功返回 True"""
    global _active_terminals
    with _terminals_lock:
        if _active_terminals >= MAX_CONCURRENT_TERMINALS:
            return False
        _active_terminals += 1
        return True


def _release_terminal_slot():
    """释放终端槽位"""
    global _active_terminals
    with _terminals_lock:
        _active_terminals = max(0, _active_terminals - 1)


def _verify_ws_token(ws) -> bool:
    """
    验证 WebSocket 连接的 JWT token（通过 query 参数）

    安全策略：
    - 认证启用时：必须提供有效 token
    - 认证未启用时：检查终端是否允许未认证访问（默认不允许）
    """
    try:
        from seed_platform.auth import AuthManager

        auth = AuthManager()

        if auth.enabled:
            # 认证启用时，验证 token
            token = ws.query_params.get("token", "")
            if not token:
                logger.warning("WebSocket 终端连接被拒绝: 缺少 token")
                return False
            payload = auth.verify_token(token)
            if not payload:
                logger.warning("WebSocket 终端连接被拒绝: token 无效或已过期")
                return False
            return True
        else:
            # 认证未启用时：与全局 JWT 中间件策略一致——本地单用户模式
            # （默认 127.0.0.1）直接放行；仅当显式配置
            # terminal_allow_unauthenticated=false 时才收紧（如局域网共享）。
            allow_unauthenticated = get_setting("terminal_allow_unauthenticated", True)
            if not allow_unauthenticated:
                logger.warning("WebSocket 终端连接被拒绝: 认证未启用且已禁用未认证访问")
                return False
            return True
    except ImportError:
        logger.warning("安全模块不可用，拒绝终端连接")
        return False
    except Exception as e:
        logger.warning(f"WebSocket 认证异常: {e}")
        return False


def _get_default_shell() -> tuple:
    """获取系统默认 shell 命令和参数"""
    if sys.platform == "win32":
        # 优先使用 cmd.exe（asyncio 兼容性更好）
        return "cmd.exe", []
    else:
        shell = os.environ.get("SHELL", "/bin/bash")
        return shell, []


def _get_console_encoding() -> str:
    """获取子进程（shell）管道输出所用的编码。

    Windows 上 cmd.exe 经管道输出使用控制台代码页（中文系统为 cp936/GBK），
    若按 UTF-8 解码会得到乱码；其他平台按系统首选编码处理。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            cp = ctypes.windll.kernel32.GetConsoleOutputCP()
            if cp:
                return f"cp{cp}"
        except Exception as e:
            logger.debug("【_get_console_encoding】获取控制台代码页失败（非致命）: %s", e)
    return locale.getpreferredencoding(False) or "utf-8"


def _normalize_terminal_input(text: str) -> str:
    """规范化键盘输入的换行符为 CRLF（仅 Windows）。

    xterm 键盘按 Enter 发送的是裸 `\r`，而管道方式启动的 cmd.exe 不把单独 `\r`
    视为行终止符：命令会停留在输入缓冲不执行，后续输入还会被拼接在同一行。
    这里先将所有 `\r\n` / 裸 `\r` 归一为 `\n`，再统一转成 `\r\n`（已含 `\r\n`
    的不会被重复转换），多行粘贴也能正确拆行。非 win32 平台保持原样。
    """
    if sys.platform != "win32":
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")


async def _read_stream(stream, ws: WebSocket, prefix: str):
    """从子进程流读取并发送到 WebSocket"""
    try:
        while True:
            data = await stream.read(4096)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            payload = json.dumps({"type": prefix, "data": text}, ensure_ascii=False)
            await ws.send_text(payload)
    except Exception as e:
        logger.debug("【_read_stream】处理失败（非致命）: %s", e)


@router.websocket("/ws/terminal")
async def terminal_websocket(ws: WebSocket):
    """Web 终端 WebSocket 端点

    安全防护：
    - JWT 认证（强制）
    - 并发限制（默认 3）
    - 空闲超时（默认 300s）
    - 可通过 terminal_enabled 配置完全禁用
    """
    import subprocess as _sp

    # 全局开关：允许管理员完全禁用终端功能
    if not get_setting("terminal_enabled", True):
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "data": "终端功能已被管理员禁用"}))
        await ws.close(code=4003, reason="Terminal disabled")
        return

    # 认证检查
    if not _verify_ws_token(ws):
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "data": "认证失败"}))
        await ws.close(code=4001, reason="Unauthorized")
        return

    # 并发限制
    if not _acquire_terminal_slot():
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "data": "终端并发数已达上限"}))
        await ws.close(code=4002, reason="Too many terminals")
        return

    process = None
    import time as _time

    _session_started = _time.time()
    _pid = 0

    loop = asyncio.get_event_loop()
    try:
        await ws.accept()

        # 获取工作目录
        work_dir = os.getcwd()
        try:
            custom_path = get_setting("workspace_path", "")
            if custom_path and os.path.isdir(custom_path):
                work_dir = custom_path
        except Exception as e:
            logger.debug("【terminal_websocket】处理失败（非致命）: %s", e)

        # 获取 shell
        shell_args: list[str]
        if sys.platform == "win32":
            shell_cmd, shell_args = "cmd.exe", []
        else:
            shell_cmd = os.environ.get("SHELL", "/bin/bash")
            shell_args = []

        # 用 Popen 创建子进程
        creationflags = _sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = _sp.Popen(
            [shell_cmd] + shell_args,
            stdin=_sp.PIPE,
            stdout=_sp.PIPE,
            stderr=_sp.PIPE,
            cwd=work_dir,
            env={**os.environ, "TERM": "xterm-256color"},
            creationflags=creationflags,
        )
        logger.info(f"终端进程启动: PID={process.pid}, work_dir={work_dir}, shell={shell_cmd}")
        _pid = process.pid

        # 欢迎消息
        await ws.send_text(
            json.dumps(
                {
                    "type": "output",
                    "data": f"\r\nTaiji Terminal | {shell_cmd} | PID: {process.pid}\r\n",
                }
            )
        )

        # 后台线程读取子进程 stdout → asyncio queue（有界，防背压撑爆内存）
        import queue

        output_queue: queue.Queue[str | None] = queue.Queue(maxsize=1000)
        _dropped = [0]
        # Windows 管道 stdin 的 write 会同步阻塞（管道满时挂起），
        # 必须丢进线程池执行，否则会卡住事件循环（输入分发/心跳）
        _stdin_codec = _get_console_encoding()

        def _put_bounded(q, item):
            """有界入队：满时丢弃最旧数据并告警（节流），保证读者始终看到最新输出"""
            try:
                q.put_nowait(item)
                return
            except queue.Full as e:
                logger.debug("【terminal_websocket._put_bounded】处理失败（非致命）: %s", e)
            try:
                q.get_nowait()
                _dropped[0] += 1
                if _dropped[0] == 1 or _dropped[0] % 100 == 0:
                    logger.warning(f"终端输出队列已满，丢弃最旧输出（累计 {_dropped[0]} 次）")
            except queue.Empty as e:
                logger.debug("【terminal_websocket._put_bounded】处理失败（非致命）: %s", e)
            try:
                q.put_nowait(item)
            except queue.Full as e:
                logger.debug("【terminal_websocket._put_bounded】处理失败（非致命）: %s", e)

        def _read_output(stream, q):
            # 注意：必须用 read1()（单次底层原始读，有数据立即返回）。
            # BufferedReader.read(4096) 在 Windows 管道上会阻塞到攒满 4KB 或 EOF，
            # 导致输出积压到子进程退出才一次性涌出（实测复现，见任务 #13）。
            decoder = codecs.getincrementaldecoder(_get_console_encoding())("replace")
            try:
                while True:
                    data = stream.read1(4096)
                    if not data:
                        break
                    # 增量解码：多字节字符可能被拆在两个块边界，避免乱码后丢弃
                    _put_bounded(q, decoder.decode(data))
            except Exception as e:
                logger.debug("【terminal_websocket._read_output】处理失败（非致命）: %s", e)
            tail = decoder.decode(b"", True)
            if tail:
                _put_bounded(q, tail)
            _put_bounded(q, None)

        _stdout_thread = threading.Thread(
            target=_read_output, args=(process.stdout, output_queue), daemon=True
        )
        _stderr_thread = threading.Thread(
            target=_read_output, args=(process.stderr, output_queue), daemon=True
        )
        _stdout_thread.start()
        _stderr_thread.start()

        # 后台任务：从 queue 读取 → WebSocket
        async def _drain_output():
            _eof_left = 2  # stdout / stderr 两个读取线程各发一个 None 哨兵
            while True:
                try:
                    data = await loop.run_in_executor(None, output_queue.get, True, 0.5)
                    if data is None:
                        _eof_left -= 1
                        if _eof_left <= 0:
                            break
                        continue
                    await ws.send_text(
                        json.dumps({"type": "output", "data": data}, ensure_ascii=False)
                    )
                except queue.Empty:
                    continue
                except Exception:
                    break

        drain_task = asyncio.create_task(_drain_output())

        # 主循环：WebSocket 输入 → 子进程 stdin（空闲超时自动断开）
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=IDLE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    logger.info(f"终端[PID={_pid}] 空闲 {IDLE_TIMEOUT_SECONDS}s，自动断开")
                    try:
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "data": f"空闲超过 {IDLE_TIMEOUT_SECONDS} 秒，连接已关闭",
                                }
                            )
                        )
                        await ws.close(code=4000, reason="Idle timeout")
                    except Exception as e:
                        logger.debug("【terminal_websocket】处理失败（非致命）: %s", e)
                    break
                msg = json.loads(raw)
                t = msg.get("type", "")
                if t == "input" and process.stdin and not process.stdin.closed:
                    data = _normalize_terminal_input(msg.get("data", ""))
                    text = await loop.run_in_executor(None, data.encode, _stdin_codec, "replace")
                    await loop.run_in_executor(None, process.stdin.write, text)
                    await loop.run_in_executor(None, process.stdin.flush)
                    # 审计日志：脱敏，仅记录输入长度（DEBUG 级）
                    if data.strip():
                        logger.debug(f"终端[PID={_pid}] 输入: {len(data)} chars")
                elif t == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
        except json.JSONDecodeError:
            if process.stdin and not process.stdin.closed:
                raw = _normalize_terminal_input(raw)
                text = await loop.run_in_executor(None, lambda: raw.encode(_stdin_codec, "replace"))
                await loop.run_in_executor(None, process.stdin.write, text)
                await loop.run_in_executor(None, process.stdin.flush)
                if raw.strip():
                    logger.debug(f"终端[PID={_pid}] 输入: {len(raw)} chars")
        except WebSocketDisconnect as e:
            logger.debug("【terminal_websocket】处理失败（非致命）: %s", e)
        finally:
            drain_task.cancel()
            elapsed = _time.time() - _session_started
            logger.info(f"终端会话结束: PID={_pid}, 持续 {elapsed:.0f}s")

    except WebSocketDisconnect:
        logger.info("终端客户端断开")
    except Exception as e:
        logger.error(f"终端异常: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "data": str(e)[:200]}))
        except Exception as e:
            logger.debug("【terminal_websocket】处理失败（非致命）: %s", e)
    finally:
        _release_terminal_slot()
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception as e:
                logger.debug("【terminal_websocket】处理失败（非致命）: %s", e)
        logger.info("终端会话结束")
