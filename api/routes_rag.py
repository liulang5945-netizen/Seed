"""
RAG 知识库 API 路由
"""

import logging
import os
import shutil
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from neuroplex.tools.rag import RAGConfig, RAGKnowledgeBase
from seed_platform.app_state import app_state
from seed_platform.paths import get_external_path

from .models import RAGSearchRequest

logger = logging.getLogger("ApiServer.RAG")
router = APIRouter()


def _ensure_rag_kb() -> Any | None:
    """Create the optional Legacy RAG adapter without coupling platform state to it."""

    if app_state.rag_kb is None:
        try:
            app_state.update_rag_kb(RAGKnowledgeBase(persist_dir=get_external_path("rag_data")))
        except Exception as exc:
            logger.warning("RAG knowledge base unavailable: %s", exc)
    return app_state.rag_kb


_ensure_rag_kb()


def _process_rag_file_background(file_path: str):
    """后台处理：嵌入模型向量化"""
    try:
        kb = _ensure_rag_kb()
        if kb is None:
            raise RuntimeError("RAG knowledge base is unavailable")
        kb.add_file(file_path)
        kb.rebuild_index()
        logger.info(f"✅ 后台 RAG 向量化建库完成: {file_path}")
    except Exception as e:
        logger.error(f"❌ 后台 RAG 向量化失败: {e}")


@router.post("/api/rag/upload")
def upload_rag_document(
    file: UploadFile = File(...), bg_tasks: BackgroundTasks = BackgroundTasks()
):
    """接收前端上传的文档，加入 RAG 知识库"""
    try:
        doc_dir = get_external_path("docs")
        os.makedirs(doc_dir, exist_ok=True)
        upload_name = file.filename or "document"
        file_path = os.path.join(doc_dir, os.path.basename(upload_name))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        bg_tasks.add_task(_process_rag_file_background, file_path)
        return {
            "status": "success",
            "message": f"文件 {file.filename} 已上传，正在后台向量化建库，请稍后查看！",
        }
    except Exception as e:
        logger.error(f"RAG 添加文件失败: {e}")
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="内部错误，请查看日志") from e


@router.post("/api/rag/clear")
async def clear_rag_documents():
    """清空 RAG 知识库索引及本地文档（不可恢复，前端需二次确认）"""
    try:
        doc_dir = get_external_path("docs")
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir, ignore_errors=True)
        removed = 0
        kb: Any = app_state.rag_kb
        if kb is not None:
            names = list(kb.get_doc_names())
            removed = len(names)
            for name in names:
                kb.remove_file(name)
            kb.rebuild_index()
        return {"status": "success", "removed": removed, "message": "知识库已清空！"}
    except Exception as e:
        logger.error(f"RAG 清空失败: {e}")
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="内部错误，请查看日志") from e


@router.get("/api/rag/files")
def list_rag_files():
    """获取已挂载的 RAG 文件列表（含大小/修改时间/索引状态）

    返回 {"files": [{"name", "size"?, "mtime"?, "status"?}]}：
    - size/mtime 来自本地文档目录的文件系统信息，不可得时省略；
    - status 为 "indexed"（已建入索引）或 "pending"（仅落盘待索引）。
    """
    kb = _ensure_rag_kb()
    indexed = set(kb.get_doc_names()) if kb is not None else set()
    doc_dir = get_external_path("docs")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        names = sorted(os.listdir(doc_dir)) if os.path.isdir(doc_dir) else []
    except OSError:
        names = []
    for name in names:
        file_path = os.path.join(doc_dir, name)
        if not os.path.isfile(file_path):
            continue
        seen.add(name)
        entry: dict[str, Any] = {"name": name}
        try:
            stat = os.stat(file_path)
            entry["size"] = stat.st_size
            entry["mtime"] = stat.st_mtime
        except OSError:
            pass
        entry["status"] = "indexed" if name in indexed else "pending"
        files.append(entry)
    # 索引中存在但本地文档目录已缺失的记录（如持久化残留）
    for name in sorted(indexed - seen):
        files.append({"name": name, "status": "indexed"})
    return {"files": files}


@router.delete("/api/rag/file/{filename:path}")
def delete_rag_file(filename: str):
    """删除指定的 RAG 文件"""
    try:
        doc_dir = os.path.abspath(get_external_path("docs"))
        doc_path = os.path.abspath(os.path.join(doc_dir, filename))
        if not (doc_path == doc_dir or doc_path.startswith(doc_dir + os.sep)):
            raise HTTPException(status_code=403, detail="路径不安全")
        if os.path.exists(doc_path):
            os.remove(doc_path)
        kb = _ensure_rag_kb()
        if kb is not None:
            kb.remove_file(filename)
            kb.rebuild_index()
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="内部错误，请查看日志") from e


# ======================== RAG 检索策略配置 API ========================


@router.get("/api/rag/config")
async def get_rag_config():
    """获取当前 RAG 检索策略配置"""
    try:
        config = RAGConfig()
        return {"status": "success", "config": config.to_dict()}
    except Exception as e:
        logger.error(f"获取 RAG 配置失败: {e}")
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="内部错误，请查看日志") from e


@router.put("/api/rag/config")
async def update_rag_config(updates: dict):
    """更新 RAG 检索策略配置

    可更新字段:
    - enable_hybrid: bool  是否启用混合检索 (Dense + BM25)
    - enable_reranker: bool  是否启用 Cross-Encoder 重排序
    - enable_query_rewrite: bool  是否启用查询改写
    - candidate_k: int  混合检索候选数
    - reranker_model: str  重排序模型名称
    """
    try:
        config = RAGConfig()
        valid_keys = set(RAGConfig.DEFAULTS.keys())
        filtered = {k: v for k, v in updates.items() if k in valid_keys}
        if not filtered:
            raise HTTPException(status_code=400, detail="无有效的配置字段")
        config.update(filtered)
        return {"status": "success", "updated": list(filtered.keys()), "config": config.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新 RAG 配置失败: {e}")
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="内部错误，请查看日志") from e


@router.get("/api/rag/status")
async def get_rag_status():
    """获取 RAG 知识库状态信息"""
    try:
        kb: Any = app_state.rag_kb
        if not kb:
            return {"status": "not_initialized"}
        return {
            "status": "ok",
            "doc_count": len(kb.documents),
            "chunk_count": len(kb.chunks),
            "has_embeddings": kb.embeddings is not None,
            "has_bm25": kb._bm25_index is not None,
            "bm25_doc_count": kb._bm25_index.doc_count if kb._bm25_index else 0,
            "embed_dim": kb._embed_dim,
        }
    except Exception as e:
        logger.error(f"获取 RAG 状态失败: {e}")
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="内部错误，请查看日志") from e


@router.get("/api/rag/stats")
async def get_rag_stats():
    """知识库统计（前端 KB 页工具栏用，形状与 /api/rag/status 对齐）。"""
    try:
        kb: Any = app_state.rag_kb
        if not kb:
            return {"status": "ok", "doc_count": 0, "chunk_count": 0}
        return {
            "status": "ok",
            "doc_count": len(kb.documents),
            "chunk_count": len(kb.chunks),
            "has_embeddings": kb.embeddings is not None,
        }
    except Exception as e:
        logger.error(f"获取 RAG 统计失败: {e}")
        return {"status": "error", "doc_count": 0, "chunk_count": 0}


@router.post("/api/rag/search")
async def rag_search(req: RAGSearchRequest):
    """在知识库中进行语义搜索"""
    try:
        kb: Any = app_state.rag_kb
        if not kb or not kb.chunks:
            return {"results": []}
        results = kb.search(req.query, top_k=req.top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"RAG 搜索失败: {e}")
        return {"results": []}


@router.get("/api/rag/preview/{filename:path}")
def rag_preview(filename: str):
    """预览 RAG 文档内容"""
    try:
        doc_dir = os.path.abspath(get_external_path("docs"))
        doc_path = os.path.abspath(os.path.join(doc_dir, filename))
        if not (doc_path == doc_dir or doc_path.startswith(doc_dir + os.sep)):
            raise HTTPException(status_code=403, detail="路径不安全")
        if os.path.exists(doc_path):
            with open(doc_path, encoding="utf-8", errors="replace") as f:
                content = f.read(10000)
            return {"content": content}
        return {"content": "(文件不存在)"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="内部错误，请查看日志") from e
