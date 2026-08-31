"""
聊天 & 健康检查 API 路由
提供：
- POST /api/chat/stream     → 流式聊天（支持本地/云端/Agent/Seed引擎）
- POST /api/chat/history/{session_id} → 保存会话历史
- POST /api/chat/upload     → 聊天文件上传
- GET  /api/health          → 健康检查
"""

import json
import logging
import os
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from api.chat_strategies import create_event_generator
from api.legacy_bridge import legacy_available
from api.models import (
    ChatRequest,
    ChatWorkbenchRequest,
    LanguagePlanningRequest,
    NaturalLanguageWorkbenchTaskRequest,
    SemanticProviderEvidenceRequest,
    TaskDecompositionRequest,
    TaskInterpretationRequest,
    TaskPlanningRequest,
    TaskSequencePlanningRequest,
)
from api.seed_runtime import get_seed_runtime, is_seed_active
from seed_platform.app_state import app_state
from seed_platform.paths import get_external_path

logger = logging.getLogger("ApiServer.Chat")
router = APIRouter()


# ======================== 流式聊天 ========================

# 替换字符/控制字符占比超过该阈值时，判定回复为不可读的调试输出。
_UNREADABLE_BAD_CHAR_RATIO = 0.02


def _answer_readable(text: str) -> bool:
    """启发式判定 Seed 原生输出是否为人类可读文本。"""
    if not text or not text.strip():
        return False
    bad = 0
    for ch in text:
        code = ord(ch)
        if ch == "\ufffd" or (code < 32 and ch not in "\n\r\t"):
            bad += 1
    return bad / len(text) < _UNREADABLE_BAD_CHAR_RATIO


def _intent_from_chat_request(request, seed_runtime):
    """Convert a structured chat task request into a Taiji-owned intent."""

    from taiji import ActionIntent

    item = request.intent
    return ActionIntent(
        intent_id=item.intent_id,
        kind=item.kind,
        parameters=item.parameters,
        source_goal_id=item.source_goal_id,
        expected_outcome=item.expected_outcome,
        confidence=item.confidence,
        tick=item.tick,
    )


def _workbench_event(event):
    return {
        "type": "workbench",
        "data": event.to_payload(),
    }


def _seed_event_generator(request, seed_runtime, *, workbench=False):
    """Seed 原生分支：用户消息转为 byte 流喂入基底，generate 产出回复。

    多轮上下文由 taiji 持久状态天然承担（无需 KV cache 拼装）；回复同时
    作为清醒持续学习写回基底。事件格式与前端统一解析协议一致。
    readable=False 时前端渲染为调试输出卡片；正常路径由 native-readable
    表层保证返回可读文本。
    """

    import asyncio

    async def event_generator():
        workbench_result = None
        try:
            if workbench:
                event_queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def push_event(event):
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)

                def execute_intent():
                    intent = _intent_from_chat_request(request, seed_runtime)
                    return seed_runtime.execute_workbench_intent(
                        intent,
                        snapshot_id=request.intent.snapshot_id,
                        approval_token=request.intent.approval_token,
                        mcp_registry_snapshot_id=request.intent.mcp_registry_snapshot_id,
                        event_sink=push_event,
                    )

                task = asyncio.create_task(asyncio.to_thread(execute_intent))
                while not task.done() or not event_queue.empty():
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue
                    yield f"data: {json.dumps(_workbench_event(event), ensure_ascii=False)}\n\n"
                workbench_result = await task

            answer = await asyncio.to_thread(
                seed_runtime.chat,
                request.prompt,
                history=request.history or None,
            )
            event = {
                "type": "final",
                "data": {
                    "answer": answer,
                    "step": 1,
                    "readable": _answer_readable(answer),
                    "language_backend": seed_runtime.chat_language_backend,
                    "runtime": "seed",
                    "workbench": (
                        None
                        if workbench_result is None
                        else {
                            "outcome": workbench_result.get("outcome"),
                            "tool_call": workbench_result.get("tool_call"),
                        }
                    ),
                },
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Seed 推理出错: {e}")
            yield f"data: {json.dumps(f'生成出错: {e}', ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return event_generator


@router.post("/api/chat/workbench/stream")
async def chat_workbench_stream(request: ChatWorkbenchRequest):
    """Run one structured Taiji intent and stream the shared workbench audit.

    This endpoint is an execution transport, not a prose-to-tool selector:
    the caller must provide a Taiji-owned ``ActionIntent`` binding. The same
    SeedRuntime audit events are emitted to this chat stream and remain
    available to the IDE through ``/api/workbench/events``.
    """

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        _intent_from_chat_request(request, seed_runtime)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        _seed_event_generator(request, seed_runtime, workbench=True)(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/api/chat/workbench/interpret")
async def chat_workbench_interpret(request: TaskInterpretationRequest):
    """Admit prose as Taiji Goal evidence without selecting or executing tools."""

    import asyncio

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        interpretation = await asyncio.to_thread(
            seed_runtime.interpret_task,
            request.prompt,
            history=request.history,
            constraints=request.constraints,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "format": interpretation.to_payload()["format"],
        "interpretation": interpretation.to_payload(),
        "goal": interpretation.to_goal().to_payload(),
        "execution": {
            "status": "not_planned",
            "action_intent": None,
            "tool_call": None,
            "side_effects": False,
            "next": "taiji_planner",
        },
    }


@router.post("/api/chat/workbench/plan")
async def chat_workbench_plan(request: TaskPlanningRequest):
    """Interpret prose and ask Taiji to plan against supplied affordances only."""

    import asyncio

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        return await asyncio.to_thread(
            seed_runtime.plan_task,
            request.prompt,
            snapshot_id=request.snapshot_id,
            parameter_bindings=request.parameter_bindings,
            history=request.history,
            constraints=request.constraints,
            novelty=request.novelty,
            resource_budget=request.resource_budget,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/chat/workbench/decompose")
async def chat_workbench_decompose(request: TaskDecompositionRequest):
    """Admit semantic task steps without selecting or executing a capability."""

    import asyncio

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        return await asyncio.to_thread(
            seed_runtime.decompose_task,
            request.steps,
            confidence=request.confidence,
            ambiguity=request.ambiguity,
            status=request.status,
            provenance=request.provenance,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/chat/workbench/semantic-evidence")
async def chat_workbench_semantic_evidence(request: SemanticProviderEvidenceRequest):
    """Validate provider semantic evidence without selecting or executing tools."""

    import asyncio

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        return await asyncio.to_thread(
            seed_runtime.admit_semantic_provider_evidence,
            request.prompt,
            request.evidence,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/chat/workbench/sequence-plan")
async def chat_workbench_sequence_plan(request: TaskSequencePlanningRequest):
    """Ground admitted semantic steps against Workbench without executing them."""

    import asyncio

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        return await asyncio.to_thread(
            seed_runtime.plan_task_sequence,
            snapshot_id=request.snapshot_id,
            parameter_bindings=request.parameter_bindings,
            novelty=request.novelty,
            resource_budget=request.resource_budget,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/chat/workbench/execute-natural-language")
async def chat_workbench_execute_natural_language(
    request: NaturalLanguageWorkbenchTaskRequest,
):
    """Execute a bounded Taiji-owned task without a caller-supplied ActionIntent."""

    import asyncio

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        return await asyncio.to_thread(
            seed_runtime.execute_natural_language_workbench_task,
            request.prompt,
            request.semantic_evidence,
            snapshot_id=request.snapshot_id,
            parameter_bindings=request.parameter_bindings,
            loop_id=request.loop_id,
            max_steps=request.max_steps,
            max_budget_units=request.max_budget_units,
            novelty=request.novelty,
            resource_budget=request.resource_budget,
            learn=request.learn,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/chat/workbench/language-plan")
async def chat_workbench_language_plan(request: LanguagePlanningRequest):
    """Plan a language switch from Taiji and Workbench evidence, without applying it."""

    import asyncio

    seed_runtime = get_seed_runtime()
    if seed_runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    try:
        return await asyncio.to_thread(
            seed_runtime.plan_language_selection,
            snapshot_id=request.snapshot_id,
            path=request.path,
            lsp_language_id=request.lsp_language_id,
            novelty=request.novelty,
            resource_budget=request.resource_budget,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天端点，支持 SSE 推送"""
    # Seed 原生运行时激活时优先走原生分支（与 Cortex 互斥）
    seed_runtime = get_seed_runtime()
    if seed_runtime is not None:
        return StreamingResponse(
            _seed_event_generator(request, seed_runtime)(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    legacy_enabled = legacy_available()

    # 触发用户指令，中断当前生命活动
    if legacy_enabled:
        try:
            from neuroplex.life.life_scheduler import get_life_scheduler

            get_life_scheduler().handle_user_directive()
        except Exception as e:
            logger.warning(f"Failed to trigger user directive: {e}")

    # 根据引擎类型选择数据收集器
    def collector_factory():
        if not legacy_enabled:
            return None
        try:
            from neuroplex.agent_ext.data_collector import DataCollector

            return DataCollector(  # type: ignore[call-arg]  # 冻结存根无显式 __init__，运行时由 except 兜底
                save_path=get_external_path(
                    os.path.join("agent", "conversations", f"{int(time.time())}.jsonl")
                )
            )
        except Exception:
            return None

    event_generator = create_event_generator(request, app_state, collector_factory)
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/chat/stream")
async def chat_stream_get():
    """明确处理对 stream 的 GET 请求，返回 405（方法不允许）以满足客户端预期"""
    raise HTTPException(status_code=405, detail="Method Not Allowed")


# ======================== 会话历史管理 ========================

_history_dir = get_external_path(os.path.join("user_data", "chat_history"))
os.makedirs(_history_dir, exist_ok=True)


def _safe_session_id(session_id: str) -> str:
    """验证 session_id，防止路径穿越"""
    import re

    if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        raise HTTPException(status_code=400, detail="无效的会话 ID")
    return session_id


@router.post("/api/chat/history/{session_id}")
async def save_chat_history(session_id: str, request: Request):
    """保存或更新指定会话的名称和消息历史"""
    session_id = _safe_session_id(session_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 请求体") from None

    session_file = os.path.join(_history_dir, f"{session_id}.json")

    # 读取已有数据（如存在）
    existing = {}
    if os.path.exists(session_file):
        try:
            with open(session_file, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as e:
            logger.debug("【save_chat_history】处理失败（非致命）: %s", e)

    # 合并更新
    if "name" in body:
        existing["name"] = body["name"]
    if "messages" in body:
        existing["messages"] = body["messages"]
    existing["session_id"] = session_id
    existing["updated_at"] = time.time()

    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存会话历史失败: {e}")
        raise HTTPException(status_code=500, detail="保存会话历史失败") from e

    return {"status": "ok", "session_id": session_id}


@router.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """获取指定会话的历史记录"""
    session_id = _safe_session_id(session_id)
    session_file = os.path.join(_history_dir, f"{session_id}.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="会话不存在")

    try:
        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"读取会话历史失败: {e}")
        raise HTTPException(status_code=500, detail="读取会话历史失败") from e


@router.post("/api/chat/sessions")
async def create_chat_session(request: Request):
    """创建会话（兼容测试中使用的简单 JSON body）"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 请求体") from None
    sid = str(body.get("id") or int(time.time()))
    name = body.get("name", "")
    session_file = os.path.join(_history_dir, f"{sid}.json")
    data = {
        "session_id": sid,
        "name": name,
        "messages": body.get("messages", []),
        "updated_at": time.time(),
    }
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        raise HTTPException(status_code=500, detail="创建会话失败") from e
    return {"status": "ok", "session_id": sid}


@router.get("/api/chat/sessions")
async def list_chat_sessions():
    """列出所有已保存的会话"""
    sessions = []
    if os.path.exists(_history_dir):
        for fname in sorted(os.listdir(_history_dir), reverse=True):
            if fname.endswith(".json"):
                fpath = os.path.join(_history_dir, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append(
                        {
                            "session_id": data.get("session_id", fname.replace(".json", "")),
                            "name": data.get("name", ""),
                            "updated_at": data.get("updated_at", 0),
                        }
                    )
                except Exception:
                    continue
    return sessions


@router.delete("/api/chat/history/{session_id}")
async def delete_chat_history(session_id: str):
    """删除指定会话的历史记录"""
    session_id = _safe_session_id(session_id)
    session_file = os.path.join(_history_dir, f"{session_id}.json")
    if os.path.exists(session_file):
        try:
            os.remove(session_file)
        except Exception as e:
            logger.error(f"删除会话历史失败: {e}")
            raise HTTPException(status_code=500, detail="删除会话历史失败") from e
    return {"status": "ok"}


# ======================== 文件上传 ========================

# 支持的文本文件扩展名
_TEXT_EXTENSIONS = {
    "txt",
    "md",
    "py",
    "js",
    "ts",
    "jsx",
    "tsx",
    "vue",
    "html",
    "htm",
    "css",
    "json",
    "xml",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "conf",
    "sh",
    "bash",
    "bat",
    "cmd",
    "ps1",
    "sql",
    "java",
    "c",
    "cpp",
    "h",
    "hpp",
    "go",
    "rs",
    "rb",
    "php",
    "swift",
    "kt",
    "scala",
    "r",
    "m",
    "lua",
    "pl",
    "ex",
    "exs",
    "hs",
    "ml",
    "clj",
    "lisp",
    "el",
    "vim",
    "tex",
    "csv",
    "log",
    "gitignore",
    "dockerfile",
    "makefile",
    "cmake",
    "gradle",
}

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff", "tif", "svg"}

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/api/chat/upload")
async def upload_chat_file(file: UploadFile = File(...)):
    """上传文件并解析内容（用于聊天上下文）"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    # 读取文件内容
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取文件失败: {e}") from e

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="文件过大，最大支持 20MB")

    # 图片文件：仅返回元信息
    if ext in _IMAGE_EXTENSIONS:
        return {
            "status": "ok",
            "filename": file.filename,
            "type": "image",
            "size": len(content),
            "parsed_text": f"[图片: {file.filename} ({len(content)} bytes)]",
        }

    # 文本文件：尝试解析内容
    if ext in _TEXT_EXTENSIONS or ext == "":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("gbk")
            except UnicodeDecodeError:
                text = content.decode("latin-1")

        # 截断过长内容
        max_chars = 50000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [截断，共 {len(content)} 字节]"

        return {
            "status": "ok",
            "filename": file.filename,
            "type": "text",
            "size": len(content),
            "parsed_text": text,
        }

    # PDF 文件
    if ext == "pdf":
        try:
            import io

            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(io.BytesIO(content))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                text = "\n\n".join(text_parts)
            except ImportError:
                text = f"[PDF 文件: {file.filename} - 需要安装 PyPDF2 才能解析 PDF 内容]"
        except Exception as e:
            text = f"[PDF 解析失败: {e}]"

        return {
            "status": "ok",
            "filename": file.filename,
            "type": "pdf",
            "size": len(content),
            "parsed_text": text,
        }

    # DOCX 文件
    if ext == "docx":
        try:
            import io

            try:
                from docx import Document

                doc = Document(io.BytesIO(content))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                text = f"[DOCX 文件: {file.filename} - 需要安装 python-docx 才能解析 DOCX 内容]"
        except Exception as e:
            text = f"[DOCX 解析失败: {e}]"

        return {
            "status": "ok",
            "filename": file.filename,
            "type": "docx",
            "size": len(content),
            "parsed_text": text,
        }

    # 其他未知类型
    return {
        "status": "ok",
        "filename": file.filename,
        "type": "unknown",
        "size": len(content),
        "parsed_text": f"[不支持的文件类型: .{ext}]",
    }


# ======================== 健康检查 ========================


@router.get("/api/health")
async def health_check():
    """健康检查端点"""
    # 启动未完成时返回 loading / downloading 状态
    if not app_state.startup_complete:
        from api.legacy_bridge import legacy_startup_download_progress

        dl = legacy_startup_download_progress()
        if dl["active"]:
            return {
                "status": "downloading",
                "message": dl["message"],
                "percent": dl["percent"],
                "total_mb": dl["total_mb"],
                "downloaded_mb": dl["downloaded_mb"],
            }
        return {"status": "loading", "message": "模型正在加载中..."}

    if app_state.startup_error:
        return {"status": "error", "message": app_state.startup_error}

    seed_runtime = get_seed_runtime()
    payload = {
        "status": "ok",
        "service": "Taiji API",
        "timestamp": time.time(),
        "model_loaded": app_state.model is not None or is_seed_active(),
        # Seed is the product's Taiji-native runtime.  The app_state flag
        # only describes the optional Legacy/Cortex object and must not make
        # the public health contract contradict runtime/status.
        "taiji_available": is_seed_active() or app_state.is_taiji(),
        "seed_active": is_seed_active(),
    }
    if seed_runtime is not None:
        payload["language_provider"] = seed_runtime.language_provider_status

    # Expose security middleware status so callers can detect silent degradation
    try:
        from api.app import SECURITY_MIDDLEWARE_AVAILABLE

        payload["security_middleware"] = SECURITY_MIDDLEWARE_AVAILABLE
    except ImportError:
        payload["security_middleware"] = False

    return payload
