"""
Seed WebSocket 服务器（8765）
=============================

前端 `useWebSocket.js` 的实时通道：生命事件推送与状态查询。

2026-08-23 重写：旧实现依赖已在结构清理中删除的 Cortex 核心门面
（`neuroplex.core.api.get_core`），独立运行时直接 ImportError。
现改为 Seed 原生最小实现：
- 可独立进程运行（桌面端 `desktop/main.py` 以子进程拉起）；
- 注册 EventBus 广播回调，生命事件实时推送到所有客户端；
- `status` → 防御式组装的生命状态（无核心依赖，取不到就降级）；
- `chat` → 经本机 HTTP API（/api/chat/stream）转发，复用运行时聊天路径；
- 其余遗留动作命令（feed/train/sleep/play/voice）如实返回不支持，
  前端协议对 `error` 类型已有容错。
"""

import asyncio
import json
import logging
import urllib.request
from typing import Optional, Set

import websockets

logger = logging.getLogger("Taiji.WebSocket")

_HTTP_API_BASE = "http://127.0.0.1:8000"


def _life_snapshot() -> dict:
    """防御式组装生命状态；任何组件缺失都降级为空段，不抛异常。"""
    status: dict = {"runtime": "seed-native"}
    try:
        from neuroplex.core.app_state import app_state

        status["model_loaded"] = bool(getattr(app_state, "model", None)) or bool(
            getattr(app_state, "seed_runtime", None)
        )
        status["is_training"] = bool(getattr(app_state, "is_training", False))
    except Exception as e:
        logger.debug("【_life_snapshot】处理失败（非致命）: %s", e)
    try:
        from neuroplex.core.life_scheduler import life_scheduler

        getter = getattr(life_scheduler, "get_status", None)
        if callable(getter):
            status["life"] = getter()
    except Exception as e:
        logger.debug("【_life_snapshot】处理失败（非致命）: %s", e)
    return status


class TaijiWebSocketServer:
    """Seed WebSocket 服务器：事件推送 + 状态查询 + HTTP 聊天转发。"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None

    async def start(self):
        """启动 WebSocket 服务器"""
        # 注册 EventBus 广播回调 — 生命事件实时推送到前端
        try:
            from neuroplex.infra.events import get_event_bus

            event_bus = get_event_bus()
            event_bus.set_broadcast_callback(self._on_life_event)
            logger.info("EventBus broadcast callback registered")
        except Exception as e:
            logger.warning(f"Failed to register EventBus broadcast: {e}")

        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
        )
        logger.info(f"Seed WebSocket 服务器启动: ws://{self.host}:{self.port}")
        await self.server.wait_closed()

    def _on_life_event(self, message: dict):
        """
        生命事件广播回调 — EventBus 发布事件时调用。

        将事件异步推送到所有连接的 WebSocket 客户端。
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在异步上下文中，创建任务
                asyncio.ensure_future(self.broadcast(message))
            else:
                loop.run_until_complete(self.broadcast(message))
        except RuntimeError as e:
            # 没有事件循环，跳过
            logger.debug("【TaijiWebSocketServer._on_life_event】处理失败（非致命）: %s", e)

    async def stop(self):
        """停止 WebSocket 服务器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Seed WebSocket 服务器已停止")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """处理客户端连接"""
        self.clients.add(websocket)
        logger.info(f"新客户端连接: {websocket.remote_address}")

        try:
            # 发送欢迎消息
            await self.send_to_client(
                websocket,
                {
                    "type": "welcome",
                    "message": "你好，我是 Seed！",
                    "status": _life_snapshot(),
                },
            )

            # 监听客户端消息
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开: {websocket.remote_address}")
        finally:
            self.clients.discard(websocket)

    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "chat":
                await self.handle_chat(websocket, data)
            elif msg_type == "status":
                await self.handle_status(websocket, data)
            elif msg_type in ("feed", "train", "sleep", "play", "voice"):
                # 遗留 Cortex 动作命令：原生运行时经 HTTP API 提供等价能力，
                # ws 通道不再承载，如实告知前端（协议对 error 已有容错）。
                await self.send_to_client(
                    websocket,
                    {
                        "type": "error",
                        "message": f"命令 '{msg_type}' 请使用对应的 HTTP API",
                    },
                )
            else:
                await self.send_to_client(
                    websocket,
                    {
                        "type": "error",
                        "message": f"未知消息类型: {msg_type}",
                    },
                )

        except json.JSONDecodeError:
            await self.send_to_client(
                websocket,
                {
                    "type": "error",
                    "message": "无效的 JSON 格式",
                },
            )
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await self.send_to_client(
                websocket,
                {
                    "type": "error",
                    "message": f"处理失败: {e}",
                },
            )

    async def handle_chat(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理聊天消息 — 转发到本机 HTTP API 的流式聊天端点。"""
        message = data.get("message", "")
        if not message:
            return

        await self.send_to_client(
            websocket,
            {
                "type": "thinking",
                "message": "思考中...",
            },
        )

        def _http_chat() -> str:
            req = urllib.request.Request(
                f"{_HTTP_API_BASE}/api/chat/stream",
                data=json.dumps({"prompt": message}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            chunks = []
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    # /api/chat/stream 契约：{"type":"final","data":{"answer":...}}
                    if isinstance(evt, dict):
                        inner = evt.get("data") or {}
                        piece = inner.get("answer") if isinstance(inner, dict) else None
                    else:
                        piece = evt  # 错误分支会直接 yield 字符串
                    if piece:
                        chunks.append(str(piece))
            return "".join(chunks)

        try:
            response = await asyncio.to_thread(_http_chat)
            await self.send_to_client(
                websocket,
                {
                    "type": "chat_response",
                    "message": response,
                },
            )
        except Exception as e:
            await self.send_to_client(
                websocket,
                {
                    "type": "error",
                    "message": f"聊天失败: {e}",
                },
            )

    async def handle_status(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理状态查询"""
        try:
            await self.send_to_client(
                websocket,
                {
                    "type": "status_response",
                    "status": _life_snapshot(),
                },
            )
        except Exception as e:
            await self.send_to_client(
                websocket,
                {
                    "type": "error",
                    "message": f"获取状态失败: {e}",
                },
            )

    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """发送消息给客户端"""
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

    async def broadcast(self, data: dict):
        """广播消息给所有客户端"""
        for client in self.clients.copy():
            try:
                await self.send_to_client(client, data)
            except Exception:
                self.clients.discard(client)


# 全局服务器实例
_server: Optional[TaijiWebSocketServer] = None


async def start_server(host: str = "localhost", port: int = 8765):
    """启动 Seed WebSocket 服务器"""
    global _server
    _server = TaijiWebSocketServer(host, port)
    await _server.start()


async def stop_server():
    """停止 Seed WebSocket 服务器"""
    global _server
    if _server:
        await _server.stop()
        _server = None


def get_server() -> Optional[TaijiWebSocketServer]:
    """获取服务器实例"""
    return _server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
    # 直接运行服务器
    asyncio.run(start_server())
