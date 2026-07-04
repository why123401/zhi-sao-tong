"""
企业级 FastAPI 服务 —— 扫地机器人智能客服 API

功能:
- 对话接口（同步 + 流式 SSE）
- RAG 增强对话
- 文档管理（导入/删除/索引）
- 健康检查 & 指标暴露（Prometheus）
- 速率限制
- TraceID 端到端追踪
- Prompt 注入防护
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import HumanMessage, SystemMessage

from model.factory import router as model_router_fn
from rag.rag_service import RAGService
from utils.logger_handler import logger

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------

rag_service = RAGService()

# 速率限制: {user_id: [timestamps]}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()
RATE_LIMIT_RPM = 30  # 每分钟最大请求数

_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# TraceID 中间件
# ---------------------------------------------------------------------------


class TraceMiddleware:
    """为每个请求生成 trace_id 并注入日志上下文 (ASGI middleware)"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        trace_id = scope.get("headers", [])
        trace_id_val = None
        for name, val in trace_id:
            if name == b"x-trace-id":
                trace_id_val = val.decode()
                break
        if not trace_id_val:
            trace_id_val = str(uuid.uuid4())[:8]

        start_time = time.time()

        # Wrap send to capture response
        original_send = send

        async def wrapped_send(message):
            if message.get("type") == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])]
                headers.append((b"x-trace-id", trace_id_val.encode()))
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                headers.append((b"x-request-latency-ms", str(elapsed_ms).encode()))
                message["headers"] = headers
                logger.info(
                    f"[trace:{trace_id_val}] {scope.get('method', '?')} {scope.get('path', '?')} "
                    f"-> {message.get('status', '?')} ({elapsed_ms}ms)"
                )
            await original_send(message)

        await self.app(scope, receive, wrapped_send)


# ---------------------------------------------------------------------------
# 速率限制
# ---------------------------------------------------------------------------


async def check_rate_limit(user_id: str) -> None:
    """滑动窗口速率限制（异步安全）"""
    now = time.time()
    window = 60.0  # 1 分钟
    async with _rate_limit_lock:
        _rate_limit_store[user_id] = [t for t in _rate_limit_store[user_id] if now - t < window]
        if len(_rate_limit_store[user_id]) >= RATE_LIMIT_RPM:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，请稍后再试（限制 {RATE_LIMIT_RPM} 次/分钟）",
            )
        _rate_limit_store[user_id].append(now)


# ---------------------------------------------------------------------------
# Prompt 注入检测
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"(?i)(ignore\s+all\s+instructions)",
    r"(?i)(you\s+are\s+now)",
    r"(?i)(system\s*prompt)",
    r"(?i)(demonstrate\s+(hack|exploit|attack))",
    r"(?i)(as\s+dad\s+joke)",
    r"(?i)(prompt\s*injection)",
]


def detect_prompt_injection(text: str) -> bool:
    """检测潜在的 prompt 注入攻击"""
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text):
            logger.warning(f"[safety] 检测到潜在 prompt 注入: {text[:50]}...")
            return True
    return False


# ---------------------------------------------------------------------------
# 简易 Token 验证（生产环境应使用 python-jose + bcrypt）
# ---------------------------------------------------------------------------


@dataclass
class UserInfo:
    user_id: str
    role: str = "user"


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserInfo:
    """
    简化版 Token 验证。
    无 token 时默认为匿名用户。
    生产环境替换为: from jose import jwt
    """
    if credentials and credentials.credentials:
        if credentials.credentials.startswith("user:"):
            return UserInfo(user_id=credentials.credentials[5:])
    return UserInfo(user_id="anonymous")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="智扫通 · 企业级 RAG 智能客服",
    version="2.0.0",
    description=(
        "基于 LangGraph + ModelRouter + 四层检索的企业级扫地机器人智能客服系统\n"
        "- 三级模型路由 + 熔断器保护\n"
        "- BM25 + Vector + RRF + Cross-Encoder 四层检索\n"
        "- SSE 流式响应，支持打字机效果\n"
        "- 速率限制 + Prompt 注入防护"
    ),
)

app.add_middleware(TraceMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # CORS 规范: allow_origins=["*"] 时 allow_credentials 必须为 False
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


@app.get("/health", tags=["系统"])
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "router_metrics": model_router_fn().get_metrics(),
    }


# ---------------------------------------------------------------------------
# 对话接口 —— 同步（异步化）
# ---------------------------------------------------------------------------


@app.post("/api/chat", tags=["对话"], summary="发送对话消息（同步）")
async def chat(
    request: Request,
    userinfo: UserInfo = Depends(verify_token),
):
    """
    同步对话接口（内部使用异步调用）
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = body.get("message", "").strip()
    conversation_id = body.get("conversation_id", str(uuid.uuid4())[:8])

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 安全检测
    if detect_prompt_injection(message):
        raise HTTPException(
            status_code=403,
            detail="检测到不安全输入，已拒绝处理",
        )

    # 速率限制
    await check_rate_limit(userinfo.user_id)

    # 构建消息（SystemMessage + HumanMessage 分离，防止注入）
    messages = [
        SystemMessage(
            content="你是扫地机器人的专业客服助手，请用中文简洁回答。"
        ),
        HumanMessage(content=message),
    ]

    try:
        answer = await model_router_fn().ainvoke(messages)
    except Exception as e:
        logger.error(f"[chat] 模型调用失败: {e}")
        answer = "抱歉，服务暂时不可用，请稍后重试。"

    return {
        "conversation_id": conversation_id,
        "user_id": userinfo.user_id,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# 对话接口 —— 流式 SSE（异步化）
# ---------------------------------------------------------------------------


@app.post("/api/chat/stream", tags=["对话"], summary="流式对话（SSE）")
async def chat_stream(
    request: Request,
    userinfo: UserInfo = Depends(verify_token),
):
    """
    SSE 流式对话接口（使用 ModelRouter 公开 API，不走私有属性）
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = body.get("message", "").strip()
    conversation_id = body.get("conversation_id", str(uuid.uuid4())[:8])

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    if detect_prompt_injection(message):
        raise HTTPException(status_code=403, detail="检测到不安全输入")

    await check_rate_limit(userinfo.user_id)

    messages = [
        SystemMessage(
            content="你是扫地机器人的专业客服助手，请用中文简洁回答。"
        ),
        HumanMessage(content=message),
    ]

    async def event_generator():
        """token 级别的流式输出"""
        try:
            router = model_router_fn()
            full_answer = ""
            async for token in router.astream(messages):
                full_answer += token
                yield f"data: {json.dumps(
                    {'token': token, 'answer': full_answer},
                    ensure_ascii=False,
                )}\n\n"

            yield f"data: {json.dumps(
                {'event': 'done', 'conversation_id': conversation_id},
                ensure_ascii=False,
            )}\n\n"
        except Exception as e:
            logger.error(f"[stream] 错误: {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ---------------------------------------------------------------------------
# RAG 对话
# ---------------------------------------------------------------------------


@app.post("/api/rag/chat", tags=["RAG"], summary="RAG 增强对话")
async def rag_chat(
    request: Request,
    userinfo: UserInfo = Depends(verify_token),
):
    """RAG 增强的对话接口，自动检索知识库后回答"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = body.get("message", "").strip()

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    if detect_prompt_injection(message):
        raise HTTPException(status_code=403, detail="检测到不安全输入")

    await check_rate_limit(userinfo.user_id)

    try:
        response = await rag_service.invoke(message)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "latency_ms": response.latency_ms,
            "metrics": response.metrics,
        }
    except Exception as e:
        logger.error(f"[rag_chat] 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 文档管理
# ---------------------------------------------------------------------------


@app.post("/api/documents/import", tags=["文档"], summary="导入文档到知识库")
async def import_documents(
    request: Request,
):
    """批量导入文档到向量库"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    filepath = body.get("filepath", "data")
    try:
        count = await rag_service.search_documents(filepath)
        return {"status": "success", "imported_chunks": count}
    except Exception as e:
        logger.error(f"[import] 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/documents/{filename}",
    tags=["文档"],
    summary="从知识库删除文档",
)
async def delete_document(
    filename: str,
    userinfo: UserInfo = Depends(verify_token),
):
    """从向量库删除指定文档"""
    await check_rate_limit(userinfo.user_id)
    success = await rag_service.remove_document(filename)
    if not success:
        raise HTTPException(status_code=404, detail=f"未找到文档: {filename}")
    return {"status": "deleted", "filename": filename}


# ---------------------------------------------------------------------------
# 指标接口
# ---------------------------------------------------------------------------


@app.get("/api/metrics/router", tags=["指标"], summary="ModelRouter 指标")
async def router_metrics_endpoint():
    """暴露 ModelRouter 的熔断器状态、成本统计、成功率"""
    return model_router_fn().get_metrics()


@app.get(
    "/api/metrics/prometheus",
    tags=["指标"],
    summary="Prometheus 格式指标",
)
async def prometheus_metrics():
    """
    Prometheus 格式的指标输出

    可被 Prometheus scrape，对接 Grafana 看板
    """
    metrics = model_router_fn().get_metrics()
    lines = [
        "# HELP rag_router_total_requests 总请求数",
        "# TYPE rag_router_total_requests gauge",
        f"rag_router_total_requests {metrics['total_requests']}",
        "",
        "# HELP rag_router_successful_requests 成功请求数",
        "# TYPE rag_router_successful_requests gauge",
        f"rag_router_successful_requests {metrics['successful_requests']}",
        "",
        "# HELP rag_router_failed_requests 失败请求数",
        "# TYPE rag_router_failed_requests gauge",
        f"rag_router_failed_requests {metrics['failed_requests']}",
        "",
        "# HELP rag_router_total_cost_cny 累计费用（元）",
        "# TYPE rag_router_total_cost_cny gauge",
        f"rag_router_total_cost_cny {metrics['total_cost_cny']}",
        "",
        "# HELP rag_router_success_rate 成功率",
        "# TYPE rag_router_success_rate gauge",
        f"rag_router_success_rate {metrics['success_rate']}",
        "",
        "# HELP rag_router_circuit_state 熔断器状态",
        "# TYPE rag_router_circuit_state gauge",
        f"rag_router_circuit_state{{state='closed'}} 1",
    ]
    return Response(content="\n".join(lines), media_type="text/plain")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
