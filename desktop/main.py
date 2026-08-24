"""
[产品入口] Seed桌面客户端 — 开发环境版本
==========================================

原生桌面应用，嵌入 Web 前端，通过子进程管理后端生命周期。

功能：
1. 嵌入 Vue 前端（QWebEngineView）
2. 系统托盘（最小化到托盘、通知）
3. 窗口管理（记住大小、位置）
4. subprocess 启动 uvicorn（端口 8000）和 WebSocket 服务器（端口 8765）
5. 子进程崩溃自动启动

启动方式：python desktop/main.py

注意：此文件与 api/run_app.py 功能重叠。
- 此文件：开发环境，子进程模式，管理 WebSocket 服务器
- api/run_app.py：打包环境，进程内 QThread，有依赖自检和热更新
- 未来计划：合并为一个入口，以 api/run_app.py 为基础，补充 WebSocket 管理
详见 docs/ENTRYPOINTS.md
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from pathlib import Path

# 设置日志：开发模式输出到控制台；打包（windowed）模式无 stderr，
# 增加文件 handler 便于冒烟排查（含 frozen 启动卡死定位）。
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("SeedDesktop")

# 项目根目录：打包（frozen）模式下以 exe 所在目录为准，
# 开发模式下为仓库根（本文件的上级目录）。
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    ROOT_DIR = Path(sys.executable).resolve().parent
else:
    ROOT_DIR = Path(__file__).parent.parent
SETTINGS_FILE = ROOT_DIR / "desktop" / "settings.json"
LOG_DIR = ROOT_DIR / "logs"

# 端口 / API 路径契约（后端与 WebSocket 服务器共用，改动只此一处）
BACKEND_PORT = 8000
WS_PORT = 8765
HEALTH_PATH = "/api/health"
SWITCH_MODEL_PATH = "/api/system/switch_model"


def _find_brand_icon() -> Path | None:
    """解析开发与 frozen 两种布局中的 Seed logo。

    打包后的前端资源位于 ``_internal/frontend/dist``，不是源码目录的
    ``frontend/public``。找不到内置 logo 时才回退到显式的 ico 资产，
    避免窗口和托盘显示 PyQt 默认小图标。
    """
    internal_root = Path(getattr(sys, "_MEIPASS", ROOT_DIR / "_internal"))
    candidates = [
        internal_root / "frontend" / "dist" / "seed-taiji-network.png",
        ROOT_DIR / "frontend" / "public" / "seed-taiji-network.png",
        internal_root / "frontend" / "dist" / "favicon.ico",
        ROOT_DIR / "frontend" / "public" / "favicon.ico",
        internal_root / "frontend" / "dist" / "logo.svg",
        ROOT_DIR / "frontend" / "public" / "logo.svg",
        ROOT_DIR / "frontend" / "dist" / "logo.svg",
        ROOT_DIR / "icon.ico",
        internal_root / "icon.ico",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(LOG_DIR / "desktop_main.log", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    logger.addHandler(_file_handler)
except OSError as e:
    logger.debug("【main】处理失败（非致命）: %s", e)


def open_child_log(name: str):
    """子进程输出重定向到日志文件。

    不能用 subprocess.PIPE：无人消费时 Windows 下约 4KB 就写满管道，
    子进程（uvicorn 启动日志很快超限）会阻塞挂起，看门狗误判为死亡并反复重启。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return open(LOG_DIR / name, "ab", buffering=0)


def load_settings() -> dict:
    """加载窗口设置"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load settings, using defaults: {e}")
    return {}


def save_settings(settings: dict):
    """保存窗口设置（写失败只记日志，不影响退出流程）"""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save settings: {e}")


class BackendManager:
    """后端进程管理器"""

    def __init__(self):
        self.process = None
        self.port = int(os.environ.get("SEED_PORT", str(BACKEND_PORT)))
        # 远程接入（移动端浏览器连同一 Web UI）：设 SEED_HOST=0.0.0.0
        # 即可从局域网内手机/平板访问；默认仅本机。
        self.host = os.environ.get("SEED_HOST", "127.0.0.1")
        self._running = False
        self._log_handle = None

    def start(self):
        """启动后端"""
        if self.is_running():
            return
        self._running = False  # 复位陈旧标志（进程死亡后看门狗重启路径）

        if FROZEN:
            # 打包模式：拉起同目录的 SeedBackend.exe（desktop/backend_worker.py
            # 打包产物，见 desktop/seed.spec 双入口）。不能用 [Seed.exe, "-m", ...]
            # ——sys.executable 是自身会递归启动 GUI；也不宜进程内线程/
            # multiprocessing spawn（logging 配置冲突 / frozen spawn 卡死）。
            self._start_frozen()
            return

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--log-level",
            "info",
        ]

        try:
            self._log_handle = open_child_log("desktop_backend.log")
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=self._log_handle,
                stderr=self._log_handle,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self._running = True
            logger.info(f"Backend started on port {self.port} (PID: {self.process.pid})")

            # 等待后端就绪
            self._wait_for_ready()

            # Seed 模式：环境变量 SEED_RUNTIME=1 时，后端就绪后自动切换到
            # taiji 原生运行时（加载 checkpoints/seed_corpus.pt）。
            if os.environ.get("SEED_RUNTIME") == "1":
                self._activate_seed()

        except Exception as e:
            logger.error(f"Failed to start backend: {e}")

    def _start_frozen(self):
        """打包模式：子进程拉起 SeedBackend.exe（console 版后端入口）。"""
        worker_exe = ROOT_DIR / "SeedBackend.exe"
        if not worker_exe.exists():
            logger.error(f"SeedBackend.exe not found: {worker_exe}")
            return

        cmd = [str(worker_exe), self.host, str(self.port)]
        try:
            self._log_handle = open_child_log("desktop_backend.log")
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=self._log_handle,
                stderr=self._log_handle,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self._running = True
            logger.info(f"Backend worker started on port {self.port} (PID: {self.process.pid})")

            # 等待后端就绪（打包冷启动较慢，给足预算）
            self._wait_for_ready(timeout=120)

            if os.environ.get("SEED_RUNTIME") == "1":
                self._activate_seed()

        except Exception as e:
            logger.error(f"Failed to start backend worker: {e}")

    def _activate_seed(self):
        """请求后端激活 Seed 原生运行时。"""
        import urllib.request

        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}{SWITCH_MODEL_PATH}",
                data=json.dumps({"model_type": "seed"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=120)
            logger.info("Seed native runtime activated")
        except Exception as e:
            logger.warning(f"Seed runtime activation failed: {e}")

    def _wait_for_ready(self, timeout: int = 30):
        """等待后端就绪"""
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}{HEALTH_PATH}", timeout=2)
                logger.info("Backend is ready")
                return True
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

        logger.warning("Backend startup timeout")
        return False

    def stop(self):
        """停止后端"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("Backend stopped")
            except Exception:
                try:
                    self.process.kill()
                except Exception as e:
                    logger.debug("【BackendManager.stop】处理失败（非致命）: %s", e)
            self._running = False
        self._close_log()

    def _close_log(self):
        """关闭子进程日志句柄（进程终止后调用，避免句柄泄漏）。"""
        handle = self._log_handle
        self._log_handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception as e:
                logger.warning(f"Failed to close backend log handle: {e}")

    def is_running(self) -> bool:
        return self._running and self.process and self.process.poll() is None


class WebSocketManager:
    """WebSocket 核心服务器管理器（端口 8765）"""

    def __init__(self):
        self.process = None
        self.port = WS_PORT
        self._running = False
        self._log_handle = None

    def start(self):
        """启动 WebSocket 服务器"""
        if self.is_running():
            return
        self._running = False  # 复位陈旧标志（进程死亡后看门狗重启路径）

        if FROZEN:
            # 打包模式：同后端，改为进程内线程运行（独立事件循环）。
            self._start_inproc()
            return

        # start_taiji.py 已在结构清理中移除；真实服务为可独立运行的
        # neuroplex/core/websocket_server.py（__main__ 入口，端口 8765）。
        cmd = [sys.executable, "-m", "neuroplex.core.websocket_server"]

        try:
            self._log_handle = open_child_log("desktop_ws.log")
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=self._log_handle,
                stderr=self._log_handle,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self._running = True
            logger.info(f"WebSocket server started on port {self.port} (PID: {self.process.pid})")

            self._wait_for_ready()

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")

    def _start_inproc(self):
        """打包模式：进程内线程运行 WebSocket 服务器。"""

        def _run():
            try:
                import asyncio
                from neuroplex.core.websocket_server import start_server

                asyncio.run(start_server())
            except Exception as e:
                logger.error(f"In-process WebSocket server crashed: {e}")

        self._thread = threading.Thread(target=_run, daemon=True, name="seed-ws")
        self._thread.start()
        self._running = True
        logger.info(f"WebSocket server started in-process on port {self.port}")

        self._wait_for_ready(timeout=60)

    def _wait_for_ready(self, timeout: int = 15):
        """等待 WebSocket 服务器就绪（必须是本进程的子进程在监听，
        避免外部占用端口时误判就绪）。"""
        import socket

        start = time.time()
        while time.time() - start < timeout:
            if FROZEN:
                # 进程内线程：线程已崩 → 直接失败；端口探测仅确认监听建立。
                thread = getattr(self, "_thread", None)
                if thread is not None and not thread.is_alive():
                    return False
            # 子进程已退出 → 无需再等，直接返回失败由看门狗处理
            if self.process and self.process.poll() is not None:
                return False
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex(("localhost", self.port))
                s.close()
                if result == 0 and self._port_owned_by_child():
                    logger.info(f"WebSocket server ready on port {self.port}")
                    return True
            except Exception as e:
                logger.debug("【WebSocketManager._wait_for_ready】处理失败（非致命）: %s", e)
            time.sleep(0.5)

        logger.warning("WebSocket server startup timeout")
        return False

    def _port_owned_by_child(self) -> bool:
        """校验端口监听者确实是自己的子进程（Windows）。"""
        if sys.platform != "win32" or not self.process:
            return True
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            TH32CS_SNAPPROCESS = 0x00000002

            class PROCESSENTRY32(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260),
                ]

            snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
            if snap == -1:
                return True
            try:
                entry = PROCESSENTRY32()
                entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
                if not kernel32.Process32First(snap, ctypes.byref(entry)):
                    return True
                while True:
                    if entry.th32ProcessID == self.process.pid:
                        return entry.th32ParentProcessID == os.getpid()
                    if not kernel32.Process32Next(snap, ctypes.byref(entry)):
                        break
            finally:
                kernel32.CloseHandle(snap)
        except Exception as e:
            logger.debug("【WebSocketManager._port_owned_by_child】处理失败（非致命）: %s", e)
        return True

    def stop(self):
        """停止 WebSocket 服务器"""
        if FROZEN:
            self._running = False
            return
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("WebSocket server stopped")
            except Exception:
                try:
                    self.process.kill()
                except Exception as e:
                    logger.debug("【WebSocketManager.stop】处理失败（非致命）: %s", e)
            self._running = False
        self._close_log()

    def _close_log(self):
        """关闭子进程日志句柄（进程终止后调用，避免句柄泄漏）。"""
        handle = self._log_handle
        self._log_handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception as e:
                logger.warning(f"Failed to close ws log handle: {e}")

    def is_running(self) -> bool:
        if FROZEN:
            thread = getattr(self, "_thread", None)
            return self._running and thread is not None and thread.is_alive()
        return self._running and self.process and self.process.poll() is None


def main():
    """启动Seed桌面客户端"""
    if FROZEN:
        # 打包模式下把 cwd 固定到 exe 目录，保证相对路径（data/、checkpoints/）可预期。
        try:
            os.chdir(ROOT_DIR)
        except OSError as e:
            logger.debug("【main】处理失败（非致命）: %s", e)
    try:
        from PyQt6.QtWidgets import (
            QApplication,
            QMainWindow,
            QSystemTrayIcon,
            QMenu,
            QVBoxLayout,
            QHBoxLayout,
            QWidget,
            QMessageBox,  # noqa: F401
            QSplashScreen,  # noqa: F401
            QLabel,
            QPushButton,
        )
        from PyQt6.QtCore import (  # noqa: F401
            QUrl,
            Qt,
            QTimer,
            QSize,
            QPoint,
            QThread,
            QEvent,
            QObject,
        )
        from PyQt6.QtGui import (
            QIcon,
            QAction,
            QPixmap,
            QColor,
            QPainter,
            QFont,
            QPainterPath,
            QRegion,
        )  # noqa: F401
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile  # noqa: F401
    except ImportError:
        logger.error("PyQt6 not installed. Run: pip install PyQt6 PyQt6-WebEngine")
        sys.exit(1)

    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("Seed")
    app.setApplicationVersion("1.6.0")
    app.setOrganizationName("Seed")
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)

    # 设置应用图标
    icon_path = _find_brand_icon()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    # 加载设置
    settings = load_settings()

    # 启动后端
    backend = BackendManager()
    backend.start()

    # 启动 WebSocket 核心服务器
    ws_server = WebSocketManager()
    ws_server.start()

    class _RestartWorker(QThread):
        """在工作线程中重启后端 / WebSocket 服务。

        start() 内的健康探测是 urlopen + sleep 轮询（最长 30s，
        frozen 120s），若在 GUI 主线程执行会造成界面假死；
        这里只做子进程管理，不触碰任何 QWidget，可安全放入 QThread。
        """

        def __init__(self, restart_backend: bool, restart_ws: bool, parent=None):
            super().__init__(parent)
            self._restart_backend = restart_backend
            self._restart_ws = restart_ws

        def run(self):
            if self._restart_backend:
                backend.start()
            if self._restart_ws:
                ws_server.start()

    class _EdgeResizeFilter(QObject):
        """无边框窗口的边缘缩放：应用级事件过滤器。

        Web 视图铺满窗口会吞掉鼠标事件，因此用应用级过滤器捕获：
        光标进入窗口边缘热区并按下左键时，交给系统处理缩放（startSystemResize）。
        """

        BORDER = 6

        def __init__(self, window):
            super().__init__()
            self._window = window

        def eventFilter(self, obj, event):
            w = self._window
            try:
                if w.isMaximized() or not w.isVisible():
                    return False
                if (
                    event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton
                ):
                    edges = self._edges_at(self._global_pos(event))
                    if edges is not None:
                        w.windowHandle().startSystemResize(edges)
                        return True
            except Exception:
                return False
            return False

        @staticmethod
        def _global_pos(event):
            gp = getattr(event, "globalPosition", None)
            return gp().toPoint() if callable(gp) else event.globalPos()

        def _edges_at(self, gpos):
            rect = self._window.frameGeometry()
            B = self.BORDER
            edges = None

            def add(edge):
                nonlocal edges
                edges = edge if edges is None else edges | edge

            if abs(gpos.x() - rect.left()) <= B:
                add(Qt.Edge.LeftEdge)
            if abs(gpos.x() - rect.right()) <= B:
                add(Qt.Edge.RightEdge)
            if abs(gpos.y() - rect.top()) <= B:
                add(Qt.Edge.TopEdge)
            if abs(gpos.y() - rect.bottom()) <= B:
                add(Qt.Edge.BottomEdge)
            return edges

    # 创建主窗口
    class SeedWindow(QMainWindow):
        WINDOW_RADIUS = 18

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Seed - AI 生命体")
            brand_icon = _find_brand_icon()
            if brand_icon is not None:
                # 显式设置窗口图标，确保 Windows 任务栏使用 Seed logo，
                # 不依赖 QApplication 默认图标或 exe 的回退图标。
                self.setWindowIcon(QIcon(str(brand_icon)))
            self.setMinimumSize(QSize(1024, 700))
            self.menuBar().hide()

            # 无边框窗口：去掉系统原生标题栏，改用自绘极简标题栏（见 _build_titlebar）。
            self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

            # 恢复窗口大小和位置
            geo = settings.get("geometry", {})
            if geo:
                self.setGeometry(
                    geo.get("x", 100),
                    geo.get("y", 100),
                    geo.get("width", 1280),
                    geo.get("height", 800),
                )
            else:
                self.resize(1280, 800)
                # 居中显示
                screen = app.primaryScreen().geometry()
                x = (screen.width() - 1280) // 2
                y = (screen.height() - 800) // 2
                self.move(x, y)

            # 创建 Web 视图
            self.web_view = QWebEngineView()
            self._frontend_loaded = False

            # 配置 Web 设置
            web_settings = self.web_view.settings()
            web_settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
            )
            web_settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            web_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
            web_settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, False)

            # 设置用户代理
            profile = self.web_view.page().profile()
            profile.setHttpUserAgent("SeedDesktop/1.6.0")

            # 监听加载完成
            self.web_view.loadFinished.connect(self._on_load_finished)

            # 直接加载前端：后端已在 main() 中就绪（start() 内含健康探测），
            # 无需启动载入动画。预渲染背景用亮色（默认主题为亮色），避免闪烁。
            self.web_view.page().setBackgroundColor(QColor("#f6f7f9"))

            # 中央区域 = 自绘标题栏 + Web 视图（无边框窗口）
            central = QWidget()
            central.setObjectName("seedWindowFrame")
            central.setStyleSheet(self._window_frame_qss(dark=False))
            central_layout = QVBoxLayout(central)
            central_layout.setContentsMargins(1, 1, 1, 1)
            central_layout.setSpacing(0)
            self._titlebar = self._build_titlebar()
            self.web_view.setStyleSheet(
                "QWebEngineView { border: none; background: transparent; "
                "border-bottom-left-radius: 17px; border-bottom-right-radius: 17px; }"
            )
            central_layout.addWidget(self._titlebar)
            central_layout.addWidget(self.web_view, 1)
            self._window_frame = central
            self.setCentralWidget(central)
            self._apply_window_shape()
            QTimer.singleShot(0, self._load_frontend)

            # 创建系统托盘
            self._create_tray()

            # 状态检查定时器
            self._restart_worker = None
            self.status_timer = QTimer()
            self.status_timer.timeout.connect(self._check_backend)
            self.status_timer.start(10000)

        def _load_frontend(self):
            """加载前端（由后端静态文件服务提供）"""
            if self._frontend_loaded:
                return
            if not backend.is_running():
                QTimer.singleShot(1000, self._load_frontend)
                return

            frontend_url = f"http://127.0.0.1:{backend.port}"
            logger.info(f"Loading frontend: {frontend_url}")
            self.web_view.load(QUrl(frontend_url))

        def _on_load_finished(self, ok):
            """前端加载完成"""
            current_url = self.web_view.url().toString()
            logger.info(f"Page loaded: {current_url} (ok={ok})")

            if ok:
                # 如果是主界面加载完成
                if f"127.0.0.1:{backend.port}" in current_url:
                    if self._frontend_loaded:
                        return
                    self._frontend_loaded = True
                    logger.info("Frontend loaded successfully")
                    # 注入错误捕获
                    self.web_view.page().runJavaScript("""
                        window.onerror = function(msg, url, line, col, error) {
                            console.error('JS Error:', msg, 'at', url, ':', line);
                            return false;
                        };
                        // 检查 Vue app 是否挂载
                        setTimeout(() => {
                            const app = document.getElementById('app');
                            if (app && app.children.length === 0) {
                                document.body.innerHTML = '<div style=\"display:flex;align-items:center;justify-content:center;height:100vh;background:#0d1117;color:#e2e8f0;font-family:sans-serif;\"><div style=\"text-align:center;\"><h1 style=\"font-size:48px;margin-bottom:16px;\">🧠</h1><h2>Seed</h2><p style=\"color:#94a3b8;margin-top:8px;\">界面加载中，请稍候...</p><p style=\"color:#64748b;font-size:12px;margin-top:16px;\">如果长时间无响应，请刷新页面</p></div></div>';
                            }
                        }, 3000);
                    """)
                    # 前端就绪后同步标题栏主题（读取 data-theme）
                    QTimer.singleShot(800, self._sync_titlebar_theme)
                else:
                    # 非主界面页面（兜底）：继续加载前端
                    QTimer.singleShot(500, self._load_frontend)
            else:
                logger.warning(f"Page load failed: {current_url}")
                # 显示错误页面
                self.web_view.setHtml(f"""
                    <html><body style="background:#0d1117;color:#e2e8f0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;">
                    <div style="text-align:center;">
                        <h1 style="font-size:48px;">⚠️</h1>
                        <h2>加载失败</h2>
                        <p style="color:#94a3b8;">无法加载页面: {current_url}</p>
                        <p style="color:#64748b;font-size:12px;margin-top:16px;">请检查后端服务是否运行</p>
                    </div></body></html>
                """)
                QTimer.singleShot(5000, self._load_frontend)

        @staticmethod
        def _titlebar_qss(dark: bool) -> str:
            """标题栏样式（跟随前端主题）。默认亮色，暗色主题切为深色。"""
            if dark:
                return """
                    #seedTitlebar { background: #0f141b; border-bottom: 1px solid #1c2530; border-top-left-radius: 17px; border-top-right-radius: 17px; }
                    #seedTitlebar QLabel { color: #cbd5e1; font-size: 12px; font-weight: 600; }
                    QPushButton.titlebarBtn { background: transparent; border: none; color: #94a3b8; font-size: 14px; border-radius: 6px; }
                    QPushButton.titlebarBtn:hover { background: #1f2937; color: #e2e8f0; }
                    QPushButton.titlebarClose:hover { background: #dc2626; color: #ffffff; }
                """
            return """
                #seedTitlebar { background: #f6f7f9; border-bottom: 1px solid #e2e5ea; border-top-left-radius: 17px; border-top-right-radius: 17px; }
                #seedTitlebar QLabel { color: #334155; font-size: 12px; font-weight: 600; }
                QPushButton.titlebarBtn { background: transparent; border: none; color: #64748b; font-size: 14px; border-radius: 6px; }
                QPushButton.titlebarBtn:hover { background: #e6e9ee; color: #0f172a; }
                QPushButton.titlebarClose:hover { background: #dc2626; color: #ffffff; }
            """

        @staticmethod
        def _window_frame_qss(dark: bool) -> str:
            border = "#283442" if dark else "#d8dde5"
            return f"""
                #seedWindowFrame {{
                    background: transparent;
                    border: 1px solid {border};
                    border-radius: 18px;
                }}
            """

        def _apply_window_shape(self):
            """用圆角 mask 裁掉 QWebEngineView 的直角，避免白角露出。"""
            if self.isMaximized():
                self.clearMask()
                return
            path = QPainterPath()
            path.addRoundedRect(
                0.0,
                0.0,
                float(self.width()),
                float(self.height()),
                float(self.WINDOW_RADIUS),
                float(self.WINDOW_RADIUS),
            )
            self.setMask(QRegion(path.toFillPolygon().toPolygon()))

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._apply_window_shape()

        def _apply_titlebar_theme(self, theme):
            """根据前端 data-theme 同步标题栏配色（'dark' → 深色，其余 → 亮色）。"""
            dark = str(theme).strip().lower() == "dark"
            if hasattr(self, "_titlebar") and self._titlebar is not None:
                self._titlebar.setStyleSheet(self._titlebar_qss(dark))
            if hasattr(self, "_window_frame") and self._window_frame is not None:
                self._window_frame.setStyleSheet(self._window_frame_qss(dark))

        def _sync_titlebar_theme(self):
            """前端就绪后读取其主题，同步标题栏配色。"""
            try:
                self.web_view.page().runJavaScript(
                    "document.documentElement.getAttribute('data-theme') || ''",
                    self._apply_titlebar_theme,
                )
            except Exception as e:
                logger.debug("【_sync_titlebar_theme】处理失败（非致命）: %s", e)

        def _build_titlebar(self):
            """自绘极简标题栏：标题 + 最小化/最大化/关闭，可拖拽、双击切换最大化。"""
            bar = QWidget()
            bar.setFixedHeight(36)
            bar.setObjectName("seedTitlebar")
            bar.setStyleSheet(self._titlebar_qss(dark=False))

            layout = QHBoxLayout(bar)
            layout.setContentsMargins(12, 0, 8, 0)
            layout.setSpacing(2)

            title = QLabel("Seed")
            layout.addWidget(title)
            layout.addStretch(1)

            min_btn = QPushButton("–")
            min_btn.setProperty("class", "titlebarBtn")
            min_btn.setFixedSize(40, 26)
            min_btn.setToolTip("最小化")
            min_btn.clicked.connect(self.showMinimized)
            layout.addWidget(min_btn)

            max_btn = QPushButton("□")
            max_btn.setProperty("class", "titlebarBtn")
            max_btn.setFixedSize(40, 26)
            max_btn.setToolTip("最大化 / 还原")
            max_btn.clicked.connect(self._toggle_max_restore)
            layout.addWidget(max_btn)

            close_btn = QPushButton("×")
            close_btn.setProperty("class", "titlebarBtn titlebarClose")
            close_btn.setFixedSize(40, 26)
            close_btn.setToolTip("关闭")
            close_btn.clicked.connect(self.close)
            layout.addWidget(close_btn)

            bar.mousePressEvent = self._titlebar_mouse_press
            bar.mouseDoubleClickEvent = self._titlebar_double_click
            return bar

        def _titlebar_mouse_press(self, event):
            """拖拽标题栏移动窗口（交给系统处理，等同原生标题栏）。"""
            if event.button() == Qt.MouseButton.LeftButton:
                handle = self.windowHandle()
                if handle is not None:
                    handle.startSystemMove()
                event.accept()

        def _titlebar_double_click(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self._toggle_max_restore()

        def _toggle_max_restore(self):
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

        def _create_tray(self):
            """创建系统托盘"""
            if not QSystemTrayIcon.isSystemTrayAvailable():
                return

            self.tray = QSystemTrayIcon(self)

            # 使用应用 logo 作为托盘图标（与窗口图标一致）
            tray_icon_path = _find_brand_icon()
            if tray_icon_path is not None:
                self.tray.setIcon(QIcon(str(tray_icon_path)))

            # 托盘菜单
            tray_menu = QMenu()

            show_action = tray_menu.addAction("显示窗口")
            show_action.triggered.connect(self._show_window)

            tray_menu.addSeparator()

            life_action = tray_menu.addAction("生命状态")
            life_action.triggered.connect(lambda: self._run_js("location.hash='/life'"))

            tray_menu.addSeparator()

            quit_action = tray_menu.addAction("退出")
            quit_action.triggered.connect(self._quit)

            self.tray.setContextMenu(tray_menu)
            self.tray.activated.connect(self._tray_activated)
            self.tray.show()

        def _tray_activated(self, reason):
            """托盘图标被点击"""
            if reason == QSystemTrayIcon.ActivationReason.Trigger:
                self._show_window()

        def _show_window(self):
            """显示窗口"""
            self.show()
            self.raise_()
            self.activateWindow()

        def _check_backend(self):
            """检查后端和 WebSocket 服务状态（主线程只做 poll 判断，
            实际重启交给 _RestartWorker，避免健康探测阻塞 UI）"""
            if self._restart_worker is not None and self._restart_worker.isRunning():
                return  # 上一次重启仍在进行，跳过本次 tick
            need_backend = not backend.is_running()
            need_ws = not ws_server.is_running()
            if not need_backend and not need_ws:
                return
            if need_backend:
                logger.warning("Backend stopped, restarting...")
            if need_ws:
                logger.warning("WebSocket server stopped, restarting...")
            self._restart_worker = _RestartWorker(need_backend, need_ws, parent=self)
            self._restart_worker.start()

        def _run_js(self, code):
            """执行 JavaScript"""
            self.web_view.page().runJavaScript(code)

        def closeEvent(self, event):
            """窗口关闭事件"""
            # 保存窗口位置
            geo = self.geometry()
            settings["geometry"] = {
                "x": geo.x(),
                "y": geo.y(),
                "width": geo.width(),
                "height": geo.height(),
            }
            save_settings(settings)

            # 最小化到托盘而不是退出
            if hasattr(self, "tray") and self.tray.isVisible():
                self.hide()
                self.tray.showMessage(
                    "Seed",
                    "已最小化到系统托盘，双击图标恢复窗口",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000,
                )
                event.ignore()
            else:
                self._quit()
                event.accept()

        def _quit(self):
            """真正退出"""
            backend.stop()
            ws_server.stop()
            app.quit()

    # 创建并显示窗口
    window = SeedWindow()
    window.show()

    # 无边框窗口的边缘缩放（拖动窗口边缘调整大小）
    edge_resize_filter = _EdgeResizeFilter(window)
    app.installEventFilter(edge_resize_filter)

    # 启动事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
