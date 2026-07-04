"""对话历史摘要管理

核心思路：当对话轮数超过阈值时，用 LLM 将早期对话压缩为摘要，
保留摘要 + 最近 N 轮完整消息，控制 checkpoint 大小。

实现方式：
- 不修改 LangGraph 内部逻辑
- 在每次 execute_stream 前检查历史长度
- 如果超限，调用 LLM 摘要旧消息，注入 system prompt
- 清除旧消息，只保留摘要和最近 N 轮
"""

from __future__ import annotations

import os
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from model.factory import chat_model
from utils.logger_handler import logger


# 摘要模板：告诉 LLM 如何压缩对话
SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个对话摘要助手。
请将以下对话历史压缩为简洁的摘要，保留关键信息和决策要点。

要求:
1. 用中文总结，不超过 500 字
2. 保留用户的明确需求和关键问题
3. 保留 AI 给出的重要建议和结论
4. 保留工具调用的关键结果（如天气、数据查询）
5. 不要保留闲聊内容
6. 按时间顺序组织

对话历史:
{history}

请生成摘要:"""),
    ("placeholder", "{messages}"),
])


class ConversationSummarizer:
    """对话历史摘要管理器

    工作原理:
    1. 维护一个滑动窗口：最近 MAX_RECENT_ROUNDS 轮完整消息
    2. 超出窗口的消息被 LLM 摘要为一段文本
    3. 摘要存储在 session_state 中，作为 system prompt 的一部分传给下一轮
    """

    def __init__(
        self,
        max_recent_rounds: int = 10,
        summary_ttl_seconds: int = 3600,  # 摘要有效期 1 小时
    ):
        self.max_recent_rounds = max_recent_rounds
        self.summary_ttl = summary_ttl_seconds
        self._model = None

    def _get_model(self):
        """懒加载聊天模型"""
        if self._model is None:
            self._model = chat_model()
        return self._model

    def summarize_history(self, messages: list[BaseMessage]) -> str:
        """
        将消息列表压缩为摘要字符串

        Args:
            messages: 完整的历史消息列表

        Returns:
            摘要文本
        """
        # 提取人类可读的对话历史
        history_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history_lines.append(f"[用户]: {content}")
            elif isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history_lines.append(f"[AI]: {content}")
            elif isinstance(msg, SystemMessage):
                # 跳过 system message（不包含在摘要中）
                continue

        history_text = "\n".join(history_lines[-60:])  # 最多取最后 60 条消息参与摘要

        model = self._get_model()
        if model is None:
            logger.warning("[摘要] 模型不可用，跳过摘要")
            return ""

        try:
            # 调用 LLM 生成摘要
            response = model.invoke([
                SystemMessage(content=SUMMARIZE_PROMPT.system_message.content),
                HumanMessage(content=history_text),
            ])
            summary = response.content if isinstance(response.content, str) else str(response.content)
            logger.info(f"[摘要] 对话压缩完成，摘要长度: {len(summary)} 字符")
            return summary
        except Exception as e:
            logger.error(f"[摘要] 摘要生成失败: {e}")
            return ""

    def trim_messages(
        self,
        messages: list[BaseMessage],
        max_recent_rounds: int | None = None,
        summary: str = "",
    ) -> list[BaseMessage]:
        """
        裁剪消息列表，只保留最近 N 轮 + 摘要

        Args:
            messages: 完整消息列表
            max_recent_rounds: 保留最近的轮数（每轮 = 1 HumanMessage + 1 AIMessage）
            summary: 早期对话的摘要文本

        Returns:
            裁剪后的消息列表（system prompt + 摘要 + 最近 N 轮）
        """
        max_recent = max_recent_rounds or self.max_recent_rounds

        # 过滤掉 system message（稍后重新注入）
        recent_messages = [
            msg for msg in messages
            if not isinstance(msg, SystemMessage)
        ]

        # 按轮数计算：每轮 = 1 Human + 1 AI
        # 保留最近的 max_recent_rounds * 2 条消息
        max_recent_msgs = max_recent * 2
        if len(recent_messages) > max_recent_msgs:
            trimmed = recent_messages[-max_recent_msgs:]
            logger.info(f"[摘要] 裁剪消息: {len(recent_messages)} -> {len(trimmed)} 条")
        else:
            trimmed = recent_messages

        # 如果有摘要，注入到消息列表开头
        if summary:
            summary_msg = HumanMessage(
                content=f"[对话摘要]\n之前的对话已被压缩为摘要:\n{summary}\n\n以下是最近的对话，请基于摘要和近期对话继续回答用户问题。"
            )
            return [summary_msg] + trimmed

        return trimmed

    def build_messages_with_summary(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
        summary: str = "",
    ) -> list[BaseMessage]:
        """
        构建包含摘要的最终消息列表

        Args:
            messages: 裁剪后的消息列表
            system_prompt: 系统提示词
            summary: 早期对话摘要

        Returns:
            最终的完整消息列表
        """
        result: list[BaseMessage] = [SystemMessage(content=system_prompt)]

        if summary:
            summary_msg = HumanMessage(
                content=f"[对话摘要]\n之前的对话已被压缩为摘要:\n{summary}\n\n以下是最近的对话，请基于摘要和近期对话继续回答用户问题。"
            )
            result.append(summary_msg)

        result.extend(messages)
        return result

    def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """判断是否需要触发摘要"""
        # 统计非 system 消息数量
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        # 超过 3 轮（6 条消息）就考虑摘要
        return len(non_system) > self.max_recent_rounds * 2 + 4
