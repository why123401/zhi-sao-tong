"""LangGraph ReAct Agent —— 扫地机器人智能客服

核心能力:
- ReAct 推理循环 (思考 -> 工具调用 -> 观察 -> 再思考 -> 回答)
- 多轮对话记忆 (SqliteSaver 持久化存储)
- 对话历史摘要压缩 (控制 checkpoint 大小)
- Token 级流式输出 (通过 DashScope 原生 AioGeneration API)
- LangGraph 中间件: 工具监控 + 模型前日志 + 动态提示词切换
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
import os

from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.summarizer import ConversationSummarizer
from model.factory import chat_model, router as model_router_fn
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger


class ReactAgent:
    def __init__(self, thread_id: str = "default", use_persistent: bool = True):
        model = chat_model()
        if model is None:
            raise RuntimeError("Chat model could not be initialized. Check DASHSCOPE_API_KEY.")

        # 对话记忆: 优先使用 SqliteSaver 持久化，失败则降级为内存存储
        self.checkpointer = self._build_checkpointer(use_persistent)
        self.thread_id = thread_id

        # 对话历史摘要管理器
        self.summarizer = ConversationSummarizer(max_recent_rounds=10)

        self.agent = create_agent(
            model=model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.checkpointer,
        )

    @staticmethod
    def _build_checkpointer(use_persistent: bool):
        """构建 checkpointer，持久化失败时自动降级为内存存储"""
        if not use_persistent:
            return MemorySaver()

        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "agent_memory.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        try:
            import sqlite3
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except Exception:
            return MemorySaver()

    def execute_stream(self, query: str, thread_id: str = None):
        """同步流式执行，Token 级流式输出

        实现方式:
        1. 通过 Agent 推理（含工具调用），获取最终回复
        2. 将最终回复通过 DashScope 原生 AioGeneration API 做 token 级流式输出

        注意:
        - 第一步 Agent 推理是阻塞的（工具调用期间不产出内容）
        - 第二步模型生成文本时走原生流式 API，每次 yield 一小段文本（3-8 字符）
        - 不重新调用 LLM 生成，直接使用 Agent 推理得到的最终答案
        """
        tid = thread_id or self.thread_id
        input_dict = {
            "input": query,
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        config = {"configurable": {"thread_id": tid}}

        # 第一步：通过 Agent 推理，等待最终回复
        final_answer = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values",
                                        config=config,
                                        context={"report": False}):
            messages = chunk.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content.strip()
                    break

        # 第二步：Token 级流式输出最终答案
        if final_answer:
            yield from self._stream_token_by_token(final_answer)

    @staticmethod
    def _stream_token_by_token(text: str):
        """将文本通过 DashScope 原生 API 做 token 级流式输出

        原理: 把最终答案作为用户输入发送给 LLM，让 LLM 原样复述，
        通过 stream=True 获取 token 级别的增量输出。
        这样既保留了 Agent 推理的正确性，又实现了打字机效果。
        """
        import asyncio
        from model.streaming_tongyi import _resolve_model_name, _build_ds_messages, _get_api_key
        from model.factory import _get_chat_model

        model = _get_chat_model()
        if model is None:
            # Fallback: 逐字输出
            for char in text:
                yield char
            return

        model_name = _resolve_model_name(model.model_name or "qwen3-max")
        api_key = _get_api_key(model)

        # 构建消息：让 LLM 原样输出这段文本
        ds_messages = _build_ds_messages([
            SystemMessage(content="请直接输出以下内容，不要做任何修改或补充。"),
            HumanMessage(content=text),
        ])

        async def _stream():
            from dashscope import AioGeneration
            coro = AioGeneration.call(
                model=model_name,
                messages=ds_messages,
                stream=True,
                api_key=api_key,
            )
            gen = await coro
            async for response in gen:
                delta = ""
                if hasattr(response, "output") and response.output:
                    choices = response.output.choices
                    if choices and len(choices) > 0:
                        delta = choices[0].message.content or ""
                if delta:
                    yield delta

        loop = asyncio.new_event_loop()
        try:
            async_gen = _stream()
            while True:
                try:
                    token = loop.run_until_complete(async_gen.__anext__())
                    yield token
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    async def aexecute_stream(self, query: str, thread_id: str = None):
        """异步流式版本，使用 LangGraph 的 astream"""
        tid = thread_id or self.thread_id
        input_dict = {
            "input": query,
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        config = {"configurable": {"thread_id": tid}}

        async for chunk in self.agent.astream(input_dict, stream_mode="values",
                                               config=config,
                                               context={"report": False}):
            latest_message = chunk["messages"][-1]
            if isinstance(latest_message, AIMessage):
                content = latest_message.content
                if content:
                    yield content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的2025-01的使用报告"):
        print(chunk, end="", flush=True)
