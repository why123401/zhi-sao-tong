"""DashScope 原生 Token 级流式适配器

问题:
LangChain 的 ChatTongyi.astream() 将 DashScope 的 token 流合并为单次返回，
导致无法实现真正的打字机效果。
同时 ChatTongyi._generate() 走旧版同步 SDK，不支持 qwen3-max 等新模型。

解决方案:
继承 ChatTongyi，重写 _generate / _stream / astream 方法，
全部走 DashScope 原生 AioGeneration.call API。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


# Models known to be incompatible with DashScope native APIs
# Fall back to qwen3-max when these are configured
_UNSUPPORTED_MODELS = {
    "qwen3.7-plus",
}

_DEFAULT_FALLBACK = "qwen3-max"
_MSG_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "assistant",
    "function": "assistant",
    "assistant": "assistant",
}


def _build_ds_messages(messages: list) -> list[dict[str, str]]:
    """将 LangChain messages 转换为 DashScope 格式"""
    ds_messages = []
    for msg in messages:
        msg_type = getattr(msg, "type", "human")
        ds_role = _MSG_TYPE_TO_ROLE.get(msg_type, "user")

        content = getattr(msg, "content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)

        ds_messages.append({"role": ds_role, "content": content})
    return ds_messages


def _resolve_model_name(raw_name: str) -> str:
    """解析模型名，不兼容的模型名 fallback 到 qwen3-max"""
    if raw_name in _UNSUPPORTED_MODELS:
        logger = __import__('utils.logger_handler', fromlist=['logger']).logger
        logger.warning(f"[StreamingChatTongyi] 模型 {raw_name} 不被 DashScope SDK 支持，fallback 到 {_DEFAULT_FALLBACK}")
        return _DEFAULT_FALLBACK
    return raw_name


def _get_api_key(model: ChatTongyi) -> str:
    """安全获取 API key 字符串"""
    key = model.dashscope_api_key
    if hasattr(key, "get_secret_value"):
        return key.get_secret_value()
    return key


class StreamingChatTongyi(ChatTongyi):
    """
    支持真正 token 级流式输出的通义千问模型。

    继承 ChatTongyi 保留所有工具绑定、参数配置能力，
    重写 _generate / _stream / astream 全部走 DashScope 原生 API。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── 同步调用（invoke） ─────────────────────────────────────

    def _generate(
        self,
        messages: list,
        *,
        stop: Optional[list[str]] = None,
        run_manager: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        同步调用 —— 走 DashScope 新版 AioGeneration API（同步包装）。

        旧版 Generation.call 不支持 qwen3.7-plus 等新模型，
        统一使用 AioGeneration.call(stream=False) 作为同步调用入口。
        """
        import asyncio

        from dashscope import AioGeneration

        ds_messages = _build_ds_messages(messages)
        api_key = _get_api_key(self)

        # 在同步上下文中运行异步调用
        def _async_call():
            loop = asyncio.new_event_loop()
            try:
                # AioGeneration.call(stream=False) 返回协程
                coro = AioGeneration.call(
                    model=_resolve_model_name(self.model_name or _DEFAULT_FALLBACK),
                    messages=ds_messages,
                    stream=False,
                    api_key=api_key,
                    **(stop and {"stop": stop} or {}),
                    **{k: v for k, v in kwargs.items()
                       if k in ("temperature", "top_p", "max_tokens",
                                "result_format", "enable_search", "top_k",
                                "repetition_penalty", "seed")},
                )
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        response = _async_call()

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: {response.code} - {response.message}"
            )

        content = response.output.choices[0].message.content
        generation = ChatGeneration(
            message=AIMessageChunk(content=content),
        )
        return ChatResult(generations=[generation])

    # ── 同步流式（stream） ─────────────────────────────────────

    def _stream(
        self,
        messages: list,
        *,
        stop: Optional[list[str]] = None,
        run_manager: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        同步流式 —— 走 DashScope 新版 AioGeneration 流式 API。
        旧版 Generation.call(stream=True) 不支持新模型，
        统一使用 AioGeneration.call(stream=True)。
        """
        import asyncio

        from dashscope import AioGeneration

        ds_messages = _build_ds_messages(messages)
        api_key = _get_api_key(self)

        def _async_stream():
            loop = asyncio.new_event_loop()
            try:
                coro = AioGeneration.call(
                    model=_resolve_model_name(self.model_name or _DEFAULT_FALLBACK),
                    messages=ds_messages,
                    stream=True,
                    api_key=api_key,
                    **(stop and {"stop": stop} or {}),
                    **{k: v for k, v in kwargs.items()
                       if k in ("temperature", "top_p", "max_tokens",
                                "result_format", "enable_search", "top_k",
                                "repetition_penalty", "seed")},
                )
                gen = loop.run_until_complete(coro)
                return gen
            finally:
                loop.close()

        responses = _async_stream()

        for response in responses:
            delta = ""
            if hasattr(response, "output") and response.output:
                choices = response.output.choices
                if choices and len(choices) > 0:
                    delta = choices[0].message.content or ""

            if delta:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=delta),
                )

    # ── 异步流式（astream） ─────────────────────────────────────

    async def _astream(
        self,
        messages: list,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Token 级异步流式输出 —— 直接调用 DashScope 原生 AioGeneration.call。

        每次 yield 一个 ChatGenerationChunk，包含当前 token 增量。
        """
        from dashscope import AioGeneration

        ds_messages = _build_ds_messages(messages)
        api_key = _get_api_key(self)

        # AioGeneration.call(stream=True) 返回协程，需先 await 得到 async generator
        ds_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("model", "messages", "api_key", "stream", "stop",
                     "temperature", "top_p", "max_tokens", "result_format",
                     "enable_search", "top_k", "repetition_penalty",
                     "meta", "baselines", "seed", "hash", "prompt_type",
                     "dataset_id", "model_request_id", "model_client")
        }
        responses = await AioGeneration.call(
            model=self.model_name or "qwen3-max",
            messages=ds_messages,
            stream=True,
            api_key=api_key,
            **(stop and {"stop": stop} or {}),
            **ds_kwargs,
        )

        async for response in responses:
            delta = ""
            if hasattr(response, "output") and response.output:
                choices = response.output.choices
                if choices and len(choices) > 0:
                    delta = choices[0].message.content or ""

            if delta:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=delta),
                )

    async def astream(
        self,
        messages: list,
        *,
        stop: Optional[list[str]] = None,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        """重写 astream，返回 AIMessageChunk 增量"""
        async for gen_chunk in self._astream(messages, stop=stop, **kwargs):
            if gen_chunk.message.content:
                yield gen_chunk.message
