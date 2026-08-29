"""
[唯一产品入口] Seed 桌面客户端 — 开发与打包共用
==============================================

原生桌面应用，嵌入 Web 前端，通过子进程管理后端生命周期。

功能：
1. 嵌入 Vue 前端（QWebEngineView），标题栏亦由前端 DOM 承载
2. 系统托盘（最小化到托盘；不发气泡通知）
3. 窗口管理（记住大小、位置）
4. subprocess 启动 uvicorn（端口 8000），进程内启动 WebSocket 服务器（端口 8765）
5. 子进程崩溃自动重启，并以 job object 保证随主进程退出

启动方式：
- 开发：python desktop/main.py
- 打包：desktop/seed.spec 的 a_main 就以本文件为唯一入口，产物 dist/Seed/Seed.exe

与 api/run_app.py 的关系：本文件是**开发与打包共用的唯一入口**；run_app.py 是仍可
独立运行的历史入口，既不被 seed.spec 打包，也不在 scripts/sync_version.py 的版本
同步清单内（因此它没有版本声明）。修改桌面行为时以本文件为准，run_app.py 仅在需要
避免留下第二套实现时同步收敛。
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# 设置日志：开发模式输出到控制台；打包（windowed）模式无 stderr，
# 增加文件 handler 便于冒烟排查（含 frozen 启动卡死定位）。
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("SeedDesktop")

# 项目根目录：打包（frozen）模式下以 exe 所在目录为准，
# 开发模式下为仓库根（本文件的上级目录）。
FROZEN = getattr(sys, "frozen", False)
ROOT_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).parent.parent
SETTINGS_FILE = ROOT_DIR / "desktop" / "settings.json"
LOG_DIR = ROOT_DIR / "logs"


# QWebEngine 在没有可用 GPU 驱动、远程桌面或受限桌面权限环境中，普通的
# 多进程 renderer 可能卡在根 HTML，既不触发 loadFinished，也不继续请求
# 前端模块。当前桌面客户端以 CPU-only 为基线，单进程是稳定的 shell 降级；
# 这些默认值必须位于任何 PyQt6 导入之前，显式环境变量仍可覆盖，便于在
# 具备稳定 GPU 的机器上恢复多进程性能路径。
def _configure_qt_runtime() -> None:
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --single-process")
    os.environ.setdefault("QT_OPENGL", "software")


_configure_qt_runtime()

# PyInstaller's PyQt6 runtime hook configures Qt plugins, but on some Windows
# builds it does not expose the wheel's nested ``Qt6/bin`` directory to the
# DLL loader before the first ``PyQt6.QtCore`` import.  Keep the handle alive
# for the process lifetime so QtCore.pyd can resolve Qt6Core.dll reliably.
_QT_DLL_DIRECTORY_HANDLES: list[object] = []
_QT_PRELOADED_LIBRARIES: list[object] = []


def _prepare_frozen_qt_dll_path() -> None:
    """Make the bundled Qt6 DLL directory visible before importing PyQt6."""

    global _QT_DLL_DIRECTORY_HANDLES, _QT_PRELOADED_LIBRARIES
    if not FROZEN or sys.platform != "win32":
        return
    internal_root = Path(getattr(sys, "_MEIPASS", ROOT_DIR / "_internal"))
    qt_bin = internal_root / "PyQt6" / "Qt6" / "bin"
    if not qt_bin.is_dir():
        return
    # PyInstaller keeps QtWebEngineProcess.exe inside the nested Qt bin
    # directory. Without an explicit path, the frozen browser can create its
    # top-level window and request the document HTML, but the renderer never
    # continues with module/CSS loading; the packaged client then appears as a
    # blank window while the backend health check remains green.
    qt_webengine_process = qt_bin / "QtWebEngineProcess.exe"
    if qt_webengine_process.is_file():
        os.environ.setdefault("QTWEBENGINEPROCESS_PATH", str(qt_webengine_process))
    search_directories = (internal_root, qt_bin)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    for directory in reversed(search_directories):
        directory_text = str(directory)
        if directory_text not in path_entries:
            path_entries.insert(0, directory_text)
    os.environ["PATH"] = os.pathsep.join(path_entries)
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        _QT_DLL_DIRECTORY_HANDLES = [
            add_dll_directory(str(directory)) for directory in search_directories
        ]
    # Resolve the two libraries that are commonly missed by frozen Qt loads
    # by absolute path.  Keeping the WinDLL objects alive prevents the loader
    # from unloading them before QtCore.pyd imports.
    try:
        from ctypes import WinDLL

        for library in (
            internal_root / "python3.dll",
            qt_bin / "Qt6Core.dll",
        ):
            if library.is_file():
                _QT_PRELOADED_LIBRARIES.append(WinDLL(str(library)))
    except OSError as exc:
        logger.warning("bundled Qt dependency preload failed: %s", exc)


_prepare_frozen_qt_dll_path()

# 端口 / API 路径契约（后端与 WebSocket 服务器共用，改动只此一处）
BACKEND_PORT = 8000
WS_PORT = 8765
HEALTH_PATH = "/api/health"
SWITCH_MODEL_PATH = "/api/runtime/activate"


def build_frontend_url(port: int) -> str:
    """Build a desktop URL that keeps the backend port and client mode."""

    return f"http://127.0.0.1:{int(port)}/#/?taiji_client=desktop"


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
        ROOT_DIR / "icon.ico",
        internal_root / "icon.ico",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


APP_USER_MODEL_ID = "Seed.Desktop.Shell"


def _set_windows_app_identity() -> None:
    """声明 Windows AppUserModelID。

    系统通知（含托盘气泡）左上角的归属图标与应用名不取 QIcon，而是取
    进程的 AppUserModelID；未声明时 Windows 回退到 exe 身份，于是显示
    占位方块加 "Seed.exe"。必须在任何窗口创建之前调用。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception as exc:  # pragma: no cover - 仅 Windows 运行时路径
        logger.warning(f"AppUserModelID 设置失败: {exc}")


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


# ---------------------------------------------------------------------------
# 子进程生命周期兜底（Windows）
# ---------------------------------------------------------------------------
# GUI 进程被强杀或崩溃时 _quit() → backend.stop() 这条清理路径根本不会执行，
# 后端 worker 会独活并继续监听 8000；而就绪探测只看「端口是否响应 /api/health」，
# 于是下一次启动静默接管了上一轮的陈旧后端 —— 表现为代码改了但行为像旧的。
# 不靠 Python 层再加一层 try/finally（强杀时同样不执行），改用内核级 Job Object：
# JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 使 Job 句柄随主进程消亡时，内核自动终止
# Job 内全部子进程，与主进程如何死亡无关。

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
ORPHAN_BACKEND_IMAGE = "SeedBackend.exe"

_CHILD_JOB = None
_CHILD_JOB_FAILED = False


def _child_job_handle():
    """惰性创建「主进程一消失就杀光子进程」的 Job Object（仅 Windows）。"""
    global _CHILD_JOB, _CHILD_JOB_FAILED
    if sys.platform != "win32" or _CHILD_JOB_FAILED:
        return None
    if _CHILD_JOB is not None:
        return _CHILD_JOB
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(job)
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

        _CHILD_JOB = job
        logger.info("Child job object armed (kill-on-close)")
        return _CHILD_JOB
    except Exception as e:
        # 老系统或受限 Job 环境下可能失败；退化为仅依赖 stop()，不影响启动。
        _CHILD_JOB_FAILED = True
        logger.warning(f"Child job object unavailable, orphan cleanup degraded: {e}")
        return None


def adopt_child(process) -> bool:
    """把子进程纳入 kill-on-close Job，主进程无论怎么死它都会被回收。"""
    job = _child_job_handle()
    if job is None or process is None:
        return False
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = int(process._handle)
        if not kernel32.AssignProcessToJobObject(job, handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        return True
    except Exception as e:
        logger.warning(f"Failed to adopt child PID {getattr(process, 'pid', '?')} into job: {e}")
        return False


def snapshot_processes() -> dict:
    """进程快照：pid -> (ppid, image_name)。非 Windows 或失败时返回空字典。"""
    if sys.platform != "win32":
        return {}
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
            return {}
        result = {}
        try:
            entry = PROCESSENTRY32()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
            if not kernel32.Process32First(snap, ctypes.byref(entry)):
                return {}
            while True:
                result[int(entry.th32ProcessID)] = (
                    int(entry.th32ParentProcessID),
                    entry.szExeFile.decode("latin-1", "replace"),
                )
                if not kernel32.Process32Next(snap, ctypes.byref(entry)):
                    break
        finally:
            kernel32.CloseHandle(snap)
        return result
    except Exception as e:
        logger.debug("【snapshot_processes】处理失败（非致命）: %s", e)
        return {}


def tcp_listener_pid(port: int):
    """返回 IPv4 下监听 ``port`` 的进程 PID；查不到返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
        AF_INET = 2
        TCP_TABLE_OWNER_PID_LISTENER = 3

        class MIB_TCPROW_OWNER_PID(ctypes.Structure):
            _fields_ = [
                ("dwState", wintypes.DWORD),
                ("dwLocalAddr", wintypes.DWORD),
                ("dwLocalPort", wintypes.DWORD),
                ("dwRemoteAddr", wintypes.DWORD),
                ("dwRemotePort", wintypes.DWORD),
                ("dwOwningPid", wintypes.DWORD),
            ]

        size = wintypes.DWORD(0)
        iphlpapi.GetExtendedTcpTable(
            None, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0
        )
        buf = ctypes.create_string_buffer(size.value)
        if (
            iphlpapi.GetExtendedTcpTable(
                buf, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0
            )
            != 0
        ):
            return None

        count = ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD))[0]
        rows = ctypes.cast(
            ctypes.byref(buf, ctypes.sizeof(wintypes.DWORD)),
            ctypes.POINTER(MIB_TCPROW_OWNER_PID * count),
        ).contents
        for row in rows:
            raw = row.dwLocalPort
            # dwLocalPort 是网络字节序
            if (((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)) == port:
                return int(row.dwOwningPid)
        return None
    except Exception as e:
        logger.debug("【tcp_listener_pid】处理失败（非致命）: %s", e)
        return None


def should_reap_listener(owner_pid, snapshot: dict) -> bool:
    """判定端口占用者是否为「本产品遗留的孤儿后端」，可安全回收。

    两个条件同时成立才回收，避免误杀用户自己起的服务或第二个客户端实例：
    1. 映像名恰为打包后端入口（该名字只存在于本产品的包里）；
    2. 其父进程已不在系统里（真孤儿）—— 父进程还活着说明另有实例在正常持有它。
    """
    if not owner_pid or owner_pid not in snapshot:
        return False
    ppid, image = snapshot[owner_pid]
    if image.lower() != ORPHAN_BACKEND_IMAGE.lower():
        return False
    return ppid not in snapshot


def load_settings() -> dict:
    """加载窗口设置"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
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
        self._reap_orphan_listener()

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
            adopt_child(self.process)

            # 等待后端就绪
            self._wait_for_ready()

            # Seed 模式：环境变量 SEED_RUNTIME=1 时，后端就绪后自动切换到
            # taiji 原生运行时（加载 checkpoints/seed_corpus.pt）。
            if os.environ.get("SEED_RUNTIME") == "1":
                self._activate_seed()

        except Exception as e:
            logger.error(f"Failed to start backend: {e}")

    def _reap_orphan_listener(self):
        """启动前回收上一轮遗留的孤儿后端。

        Job Object 已覆盖「本次运行期间主进程怎么死都回收」，但客户端在本改动
        之前留下的孤儿、以及 Job 不可用的降级场景仍需处理。就绪探测只看
        /api/health 有无响应，孤儿会被静默当成「后端已就绪」接管，导致前端加载
        的其实是上一轮的陈旧 dist —— 必须在启动前显式清掉。
        """
        owner = tcp_listener_pid(self.port)
        if owner is None or owner == os.getpid():
            return
        snapshot = snapshot_processes()
        if not should_reap_listener(owner, snapshot):
            return
        logger.warning(f"Reaping orphaned backend on port {self.port} (PID: {owner})")
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            PROCESS_TERMINATE = 0x0001
            handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, owner)
            if not handle:
                logger.warning(f"OpenProcess failed for orphan PID {owner}")
                return
            try:
                kernel32.TerminateProcess(handle, 1)
            finally:
                kernel32.CloseHandle(handle)
            # 等端口真正释放，否则新后端 bind 会失败
            for _ in range(20):
                if tcp_listener_pid(self.port) != owner:
                    return
                time.sleep(0.2)
            logger.warning(f"Port {self.port} still held after reaping PID {owner}")
        except Exception as e:
            logger.warning(f"Failed to reap orphaned backend PID {owner}: {e}")

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
            adopt_child(self.process)

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
                data=json.dumps({"checkpoint_id": ""}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=120)
            logger.info("Seed native runtime activated")
        except Exception as e:
            logger.warning(f"Seed runtime activation failed: {e}")

    def _wait_for_ready(self, timeout: int = 30):
        """等待后端就绪"""
        import urllib.error
        import urllib.request

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
            adopt_child(self.process)

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
        from PyQt6.QtCore import (  # noqa: F401
            QEvent,
            QFile,
            QIODevice,
            QObject,
            QPoint,
            QSize,
            Qt,
            QThread,
            QTimer,
            QUrl,
            pyqtSlot,
        )
        from PyQt6.QtGui import (  # noqa: F401
            QAction,
            QColor,
            QFont,
            QIcon,
            QPainter,
            QPainterPath,
            QPixmap,
            QRegion,
        )
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtWebEngineCore import (  # noqa: F401
            QWebEngineProfile,
            QWebEngineScript,
            QWebEngineSettings,
        )
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWidgets import (
            QApplication,
            QMainWindow,
            QMenu,
            QMessageBox,  # noqa: F401
            QSplashScreen,  # noqa: F401
            QSystemTrayIcon,
        )
    except ImportError as exc:
        # Keep the original exception in the frozen-client log.  The grouped
        # imports include QtWebEngine, and a generic "PyQt6 not installed"
        # message otherwise hides DLL/resource packaging failures.
        logger.exception("PyQt6 desktop imports failed: %r", exc)
        logger.error("PyQt6 not installed. Run: pip install PyQt6 PyQt6-WebEngine")
        sys.exit(1)

    # 创建应用
    _set_windows_app_identity()
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

    def _inject_webchannel_client(profile) -> None:
        """把 qwebchannel.js 注入为文档级用户脚本。

        前端由后端以 http:// 提供，无法用 <script src="qrc:..."> 引用 Qt 资源，
        因此把 Qt 自带的客户端库在 DocumentCreation 阶段注入主世界，
        前端即可直接使用 window.QWebChannel。
        """
        scripts = profile.scripts()
        if scripts.find("seed_qwebchannel"):
            return
        f = QFile(":/qtwebchannel/qwebchannel.js")
        if not f.open(QIODevice.OpenModeFlag.ReadOnly):
            logger.warning("qwebchannel.js 资源读取失败，前端标题栏将退化为无窗口控制")
            return
        try:
            source = bytes(f.readAll()).decode("utf-8")
        finally:
            f.close()
        script = QWebEngineScript()
        script.setName("seed_qwebchannel")
        script.setSourceCode(source)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setRunsOnSubFrames(False)
        scripts.insert(script)

    class _WindowBridge(QObject):
        """前端标题栏 → 窗口控制的 QWebChannel 桥。

        标题栏移入 Web 层后，最小化 / 最大化 / 关闭 / 拖拽 / 双击等原本由
        QPushButton 承担的行为改由前端调用这些槽完成。窗口状态回写到
        ``document.documentElement`` 的 data-maximized，供 CSS 调整圆角。
        """

        def __init__(self, window):
            super().__init__(window)
            self._window = window

        @pyqtSlot()
        def minimize(self):
            self._window.showMinimized()

        @pyqtSlot()
        def toggleMaximize(self):
            self._window._toggle_max_restore()

        @pyqtSlot()
        def close(self):
            self._window.close()

        @pyqtSlot()
        def startDrag(self):
            handle = self._window.windowHandle()
            if handle is not None:
                handle.startSystemMove()

        @pyqtSlot(result=bool)
        def isMaximized(self):
            return self._window.isMaximized()

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

            # 无边框窗口：去掉系统原生标题栏，标题栏由前端 AppTitlebar.vue 绘制，
            # 窗口控制经 _WindowBridge 走 QWebChannel 回调。
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

            # 窗口控制桥：标题栏由前端绘制，通过 QWebChannel 回调窗口操作
            self._bridge = _WindowBridge(self)
            self._channel = QWebChannel(self.web_view.page())
            self._channel.registerObject("seedWindow", self._bridge)
            self.web_view.page().setWebChannel(self._channel)
            _inject_webchannel_client(profile)

            # 直接加载前端：后端已在 main() 中就绪（start() 内含健康探测），
            # 无需启动载入动画。预渲染背景用亮色（默认主题为亮色），避免闪烁。
            self.web_view.page().setBackgroundColor(QColor("#f6f7f9"))

            # 中央区域 = 纯 Web 视图。标题栏已移入前端，Qt 侧不再叠加控件层，
            # 否则两套渲染引擎各自绘制背景，必然出现接缝。窗口圆角由
            # _apply_window_shape 的 QRegion 遮罩统一负责。
            self.web_view.setStyleSheet("QWebEngineView { border: none; background: transparent; }")
            self.setCentralWidget(self.web_view)
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

            frontend_url = build_frontend_url(backend.port)
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
                    # 前端就绪后回写窗口状态，供 Web 标题栏按钮与圆角联动
                    QTimer.singleShot(0, self._sync_window_state)
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
            self._sync_window_state()

        def changeEvent(self, event):
            super().changeEvent(event)
            if event.type() == QEvent.Type.WindowStateChange:
                self._apply_window_shape()
                self._sync_window_state()

        def _sync_window_state(self):
            """把最大化状态写回 DOM，供前端标题栏按钮与窗口圆角联动。"""
            if not getattr(self, "_frontend_loaded", False):
                return
            flag = "true" if self.isMaximized() else "false"
            try:
                self.web_view.page().runJavaScript(
                    f"document.documentElement.setAttribute('data-maximized','{flag}')"
                )
            except Exception as e:
                logger.debug("【_sync_window_state】处理失败（非致命）: %s", e)

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

            # 关闭窗口后不再弹系统气泡，恢复方式改由 tooltip 承载
            self.tray.setToolTip("Seed — 双击图标恢复窗口")

            # 托盘菜单
            tray_menu = QMenu()

            show_action = tray_menu.addAction("显示窗口")
            show_action.triggered.connect(self._show_window)

            tray_menu.addSeparator()

            life_action = tray_menu.addAction("生命状态")
            life_action.triggered.connect(self._show_life_status)

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

        def _show_life_status(self):
            """从托盘恢复窗口并切换到唯一的生命状态页面。"""
            self._show_window()
            # 先让隐藏窗口完成恢复，再让 WebView 执行 hash 路由切换；否则只改 hash
            # 时页面仍处于隐藏态，用户会误以为托盘菜单无效。
            QTimer.singleShot(0, lambda: self._run_js("location.hash='/life'"))

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
                # 不弹系统气泡：Windows 通知的归属图标由 AppUserModelID 决定，
                # 且托盘提示对已知行为属于噪音。托盘 tooltip 已说明恢复方式。
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
