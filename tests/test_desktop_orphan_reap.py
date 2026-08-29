"""孤儿后端回收判定的契约测试。

背景：客户端被强杀时 ``_quit() -> backend.stop()`` 不会执行，打包后端
（SeedBackend.exe）会独活并继续监听 8000。而 ``_wait_for_ready`` 只探测
``/api/health`` 是否响应，孤儿会被静默当成「后端已就绪」接管——用户看到的
就是「代码改了但客户端行为像旧的」。已经踩过两次（PID 20488、4944）。

真正的兜底是内核级 Job Object（kill-on-close），但回收旧孤儿的判定逻辑是
纯函数，必须锁住它的边界：只杀真孤儿，绝不误杀用户自己的服务或第二个实例。

``desktop/main.py`` 顶层只导入标准库（PyQt 在 main() 内部才导入），
因此这里可以直接 import 而不需要 GUI 环境。
"""

from __future__ import annotations

import os

from desktop.main import (
    ORPHAN_BACKEND_IMAGE,
    _configure_qt_runtime,
    build_frontend_url,
    should_reap_listener,
)


def test_desktop_qt_runtime_has_safe_software_defaults(monkeypatch):
    """打包客户端在导入 Qt 前应提供无 GPU 驱动时的稳定退化路径。"""
    monkeypatch.delenv("QTWEBENGINE_CHROMIUM_FLAGS", raising=False)
    monkeypatch.delenv("QT_OPENGL", raising=False)

    _configure_qt_runtime()

    assert "--disable-gpu" in os.environ["QTWEBENGINE_CHROMIUM_FLAGS"]
    assert os.environ["QT_OPENGL"] == "software"


def test_desktop_frontend_url_keeps_backend_port():
    """前端必须显式进入 desktop 模式，并复用实际后端端口。"""
    assert build_frontend_url(8137) == (
        "http://127.0.0.1:8137/#/?taiji_client=desktop"
    )


def test_reaps_backend_whose_parent_is_gone():
    """映像名匹配且父进程已消失 → 真孤儿，回收。"""
    snapshot = {4944: (20488, ORPHAN_BACKEND_IMAGE)}  # 20488 不在快照里
    assert should_reap_listener(4944, snapshot) is True


def test_keeps_backend_owned_by_a_live_client():
    """父进程还活着说明另有客户端实例正常持有它，不能杀。"""
    snapshot = {
        4944: (11560, ORPHAN_BACKEND_IMAGE),
        11560: (1, "Seed.exe"),
    }
    assert should_reap_listener(4944, snapshot) is False


def test_keeps_foreign_process_on_the_port():
    """端口被用户自己的服务占用时必须原样放行，否则是破坏性行为。"""
    snapshot = {7000: (1, "python.exe")}
    assert should_reap_listener(7000, snapshot) is False


def test_image_name_match_is_case_insensitive():
    """Windows 进程映像名大小写不稳定，判定不能因大小写漏掉孤儿。"""
    snapshot = {4944: (20488, ORPHAN_BACKEND_IMAGE.upper())}
    assert should_reap_listener(4944, snapshot) is True


def test_no_listener_or_unknown_pid_is_a_noop():
    """查不到占用者 / 快照拿不到（非 Windows 或 API 失败）时不做任何回收。"""
    assert should_reap_listener(None, {4944: (1, ORPHAN_BACKEND_IMAGE)}) is False
    assert should_reap_listener(0, {}) is False
    assert should_reap_listener(4944, {}) is False
