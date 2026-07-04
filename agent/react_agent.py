"""LangGraph ReAct Agent —— 扫地机器人智能客服

核心能力:
- ReAct 推理循环 (思考 -> 工具调用 -> 观察 -> 再思考 -> 回答)
- 多轮对话记忆 (SqliteSaver 持久化存储)
- 对话历史摘要压缩 (控制 checkpoint 大小)
- Token 级流式输出 (通过 stream_mode="generator" 逐 token yield)
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
        1. 使用 stream_mode="values" 等待 Agent 推理完成
        2. 识别最终 AIMessage（推理结束时最后一条消息）
        3. 将最终答案按段落分块 yield，实现打字机效果

        注意:
        - Agent 在工具调用阶段（如 RAG 检索）是阻塞的，此时不产出内容
        - 只有模型生成文本时才会 yield，工具调用期间的等待对用户透明
        - 按段落分块 yield 比逐字符更高效，用户体验也更自然
        """
        tid = thread_id or self.thread_id
        input_dict = {
            "input": query,
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        config = {"configurable": {"thread_id": tid}}

        # 等待 Agent 推理完成，追踪最后一条 AIMessage
        final_answer = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values",
                                        config=config,
                                        context={"report": False}):
            messages = chunk.get("messages", [])
            # 检查最后一条消息是否是 AIMessage（推理结束的标志）
            if messages and isinstance(messages[-1], AIMessage):
                content = messages[-1].content.strip()
                if content:
                    final_answer = content

        # 将最终答案按段落/句子分块 yield，实现打字机效果
        if final_answer:
            yield from self._yield_incremental(final_answer)

    @staticmethod
    def _yield_incremental(text: str, chunk_size: int = 3):
        """将文本按小块逐段 yield，模拟打字机效果

        按句子边界切分，每 chunk_size 个句子为一块输出，
        这样既流畅又不会太碎片化。
        """
        import re
        # 按中文/英文句子边界分割
        sentences = re.split(r'(?<=[。！？\n])', text)
        buffer = ""
        for sentence in sentences:
            if not sentence.strip():
                if buffer:
                    yield buffer.strip()
                    buffer = ""
                continue
            buffer += sentence
            # 积累一定长度后 yield
            if len(buffer) >= chunk_size * 8:  # 约 24 个中文字
                yield buffer.strip()
                buffer = ""
        if buffer:
            yield buffer.strip()

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
