"""LangGraph ReAct Agent —— 扫地机器人智能客服

核心能力:
- ReAct 推理循环 (思考 -> 工具调用 -> 观察 -> 再思考 -> 回答)
- 多轮对话记忆 (SqliteSaver 持久化存储)
- 对话历史摘要压缩 (控制 checkpoint 大小)
- Token 级流式输出 (通过 StreamingChatTongyi 原生 DashScope API)
- LangGraph 中间件: 工具监控 + 模型前日志 + 动态提示词切换
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
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
        1. 使用 stream_mode="values" 等待 Agent 推理完成（含工具调用）
        2. 提取最终 AIMessage 的完整内容
        3. 通过 StreamingChatTongyi.astream() 重新生成，逐 token yield

        注意:
        - 步骤 1 中 Agent 推理是阻塞的（工具调用期间不产出）
        - 步骤 3 中模型生成文本时走 DashScope 原生流式 API，
          每次 yield 一小段文本（3-8 个字符），实现打字机效果
        - 步骤 3 会额外调用一次 LLM，但换来真正的 token 级流式体验
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
            # 取最后一条 AIMessage 作为最终回复
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content.strip()
                    break

        # 第二步：通过 DashScope 原生 AioGeneration 做 token 级流式输出
        if final_answer:
            try:
                import asyncio
                from model.streaming_tongyi import _resolve_model_name
                from model.factory import _get_chat_model

                model = _get_chat_model()
                if model is None:
                    raise RuntimeError("Model not available")

                # 获取系统提示词和模型名
                system_prompt = load_system_prompts()
                model_name = _resolve_model_name(model.model_name or "qwen3-max")

                # 构建 DashScope 格式的消息
                from model.streaming_tongyi import _build_ds_messages, _get_api_key
                ds_messages = _build_ds_messages([
                    type('SystemMessage', (), {'type': 'system', 'content': system_prompt})(),
                    type('HumanMessage', (), {'type': 'human', 'content': final_answer})(),
                ])

                async def _stream_tokens():
                    from dashscope import AioGeneration
                    api_key = _get_api_key(model)
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

                # 在同步函数中运行异步生成器
                loop = asyncio.new_event_loop()
                try:
                    async_generator = _stream_tokens()
                    # 逐 token yield
                    import asyncio
                    while True:
                        try:
                            token = loop.run_until_complete(async_generator.__anext__())
                            yield token
                        except StopAsyncIteration:
                            break
                finally:
                    loop.close()
            except Exception:
                # Fallback: 逐字输出
                for char in final_answer:
                    yield char

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
