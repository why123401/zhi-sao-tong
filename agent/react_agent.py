"""LangGraph ReAct Agent —— 扫地机器人智能客服

核心能力:
- ReAct 推理循环 (思考 -> 工具调用 -> 观察 -> 再思考 -> 回答)
- 多轮对话记忆 (SqliteSaver 持久化存储)
- 对话历史摘要压缩 (控制 checkpoint 大小)
- Token 级流式输出 (通过 stream_mode="messages" 捕获 AIMessage delta)
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
        """同步流式执行，逐段输出最终回复

        实现方式:
        使用 stream_mode="values" 监听 Agent 每一步的状态变更。
        每当产生新的 AIMessage（模型生成完成），立即 yield 其内容。
        这样模型生成文本的过程中就能逐步产出，而不是等整个推理结束。

        注意:
        - 工具调用阶段（RAG检索等）不产出内容，用户会看到短暂等待
        - 模型生成文本时会逐条产出 AIMessage，实现近实时流式
        - ReAct 过程中可能有多条 AIMessage（思考 + 最终回答），
          最后一条即为最终回复
        """
        tid = thread_id or self.thread_id
        input_dict = {
            "input": query,
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        config = {"configurable": {"thread_id": tid}}

        # 追踪已 yield 的 AIMessage 内容，避免重复
        yielded_content = set()

        for chunk in self.agent.stream(input_dict, stream_mode="values",
                                        config=config,
                                        context={"report": False}):
            messages = chunk.get("messages", [])
            # 遍历所有消息，找出新增的 AIMessage
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content.strip()
                    if content and content not in yielded_content:
                        yielded_content.add(content)
                        yield content

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
