"""
模型工厂 —— 统一管理 ChatModel / Embedding 的创建与路由

改造说明:
- 使用 @lru_cache 懒加载，避免模块导入时网络不可用导致崩溃
- 每个模型注册独立熔断器，实现真正的三级降级
- 所有模型创建包裹 try/except，失败时记录日志而非直接崩溃
- 使用 StreamingChatTongyi 替代 ChatTongyi，实现真正的 token 级流式输出
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi

from model.router import ModelRouter, Priority, CircuitBreaker
from model.streaming_tongyi import StreamingChatTongyi
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | ChatTongyi]:
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂 —— 创建通义千问实例（支持 token 级流式）"""
    def generator(self) -> Optional[StreamingChatTongyi]:
        model_name = rag_conf.get("chat_model_name", "qwen3-max")
        logger.info(f"[ChatModelFactory] 创建模型: {model_name} (StreamingChatTongyi)")
        return StreamingChatTongyi(model=model_name)


class EmbeddingsFactory(BaseModelFactory):
    """嵌入模型工厂 —— 创建 DashScope 嵌入实例"""
    def generator(self) -> Optional[Embeddings]:
        model_name = rag_conf.get("embedding_model_name", "text-embedding-v4")
        logger.info(f"[EmbeddingsFactory] 创建模型: {model_name}")
        return DashScopeEmbeddings(model=model_name)


# ── 懒加载全局实例 ──────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_chat_model() -> StreamingChatTongyi | None:
    """懒加载聊天模型（流式版），创建失败时返回 None"""
    try:
        factory = ChatModelFactory()
        return factory.generator()
    except Exception as e:
        logger.error(f"[ChatModelFactory] 模型创建失败: {e}")
        return None


@lru_cache(maxsize=1)
def _get_embed_model() -> Embeddings | None:
    """懒加载嵌入模型，创建失败时返回 None"""
    try:
        factory = EmbeddingsFactory()
        return factory.generator()
    except Exception as e:
        logger.error(f"[EmbeddingsFactory] 模型创建失败: {e}")
        return None


@lru_cache(maxsize=1)
def _build_router() -> ModelRouter:
    """构建 ModelRouter，注册三级降级模型"""
    router_obj = ModelRouter()

    primary = _get_chat_model()
    if primary is not None:
        primary_name = rag_conf.get("chat_model_name", "qwen3-max")
        router_obj.register_model(
            primary, Priority.HIGH, primary_name,
            failure_threshold=5, recovery_timeout=30.0,
        )
        logger.info(f"[Router] 注册主模型: {primary_name} (HIGH, Streaming)")

    # 降级模型 1：qwen-plus
    fallback_name = "qwen-plus"
    try:
        fallback_model = StreamingChatTongyi(model=fallback_name)
        router_obj.register_model(
            fallback_model, Priority.MEDIUM, fallback_name,
            failure_threshold=3, recovery_timeout=60.0,
        )
        logger.info(f"[Router] 注册降级模型: {fallback_name} (MEDIUM, Streaming)")
    except Exception as e:
        logger.warning(f"[Router] 降级模型 {fallback_name} 创建失败: {e}")

    # 兜底模型：qwen-turbo
    turbo_name = "qwen-turbo"
    try:
        turbo_model = StreamingChatTongyi(model=turbo_name)
        router_obj.register_model(
            turbo_model, Priority.LOW, turbo_name,
            failure_threshold=2, recovery_timeout=120.0,
        )
        logger.info(f"[Router] 注册兜底模型: {turbo_name} (LOW, Streaming)")
    except Exception as e:
        logger.warning(f"[Router] 兜底模型 {turbo_name} 创建失败: {e}")

    return router_obj


# 公开 API —— 调用方通过函数调用获取实例
chat_model = _get_chat_model
embed_model = _get_embed_model
router = _build_router
