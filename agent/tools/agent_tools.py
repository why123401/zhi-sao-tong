"""Agent 工具集 —— 所有数据查询逻辑收敛到此层

Agent 层只负责意图判断和路由决策，不做数据查询。
所有工具函数通过 @tool 装饰器注册为 LangChain 工具。
"""

import json
import os
import random

from langchain_core.tools import tool

from rag.rag_service import RAGService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# 延迟初始化，避免模块导入时阻塞
_rag_instance: RAGService | None = None


def _get_rag() -> RAGService:
    """懒加载 RAG 服务实例"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGService()
    return _rag_instance


# ── 模拟数据 ──────────────────────────────────────────────

user_ids = [f"{i:04d}" for i in range(1001, 1011)]
month_arr = [f"2025-{m:02d}" for m in range(1, 13)]

external_data: dict[str, dict[str, dict[str, str]]] = {}


@tool
def rag_summarize(query: str) -> str:
    """从向量存储中检索参考资料并生成总结回答。

    Args:
        query: 检索查询词，应为贴合用户问题的核心关键词。

    Returns:
        字符串类型的专业资料内容。
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    rag = _get_rag()

    def _invoke():
        try:
            return rag.rag_summarize(query)
        except RuntimeError:
            # 在运行中的事件循环内，用全新事件循环调用异步版本
            return asyncio.new_event_loop().run_until_complete(
                rag.rag_summarize_async(query)
            )

    with ThreadPoolExecutor() as executor:
        future = executor.submit(_invoke)
        return future.result()


@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。

    通过 wttr.in 免费天气 API 获取真实天气数据。
    如果 API 不可用，回退到模拟数据。

    Args:
        city: 城市名称，如 "深圳"、"杭州"、"Beijing"。

    Returns:
        天气描述字符串，包含温度、湿度、风力、能见度等信息。
    """
    try:
        import urllib.request
        import urllib.parse

        req = urllib.request.Request(
            f"https://wttr.in/{urllib.parse.quote(city)}?format=j1",
            headers={"User-Agent": "SmartCustomerService/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}]).get("value", city)
        temp_c = current.get("temp_C", "N/A")
        feels_like = current.get("FeelsLikeC", "N/A")
        wind_mph = current.get("windspeedMiles", "N/A")
        humidity = current.get("humidity", "N/A")
        desc = current.get("weatherDesc", [{}])[0].get("value", "未知")
        visibility_km = current.get("visibility", "N/A")
        pressure_mb = current.get("pressure", "N/A")

        return (
            f"{area_name} 当前天气: {desc} | "
            f"温度: {temp_c}°C (体感 {feels_like}°C) | "
            f"湿度: {humidity}% | "
            f"风速: {wind_mph} mph | "
            f"能见度: {visibility_km} km | "
            f"气压: {pressure_mb} mb"
        )
    except Exception:
        # 降级：返回模拟数据，保证 Agent 不会因天气工具失败而中断
        return (
            f"城市 {city} 天气为晴天，气温26摄氏度，"
            f"空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"
        )


@tool
def get_user_location() -> str:
    """获取用户所在的城市名称（模拟数据）。

    Returns:
        城市名字符串。
    """
    return random.choice(["深圳", "合肥", "杭州"])


@tool
def get_user_id() -> str:
    """获取当前用户的唯一标识（模拟数据）。

    Returns:
        用户ID字符串，格式为4位数字。
    """
    return random.choice(user_ids)


@tool
def get_current_month() -> str:
    """获取当前月份（模拟数据）。

    Returns:
        月份字符串，格式为 YYYY-MM。
    """
    return random.choice(month_arr)


def _generate_external_data() -> dict[str, dict[str, dict[str, str]]]:
    """从 CSV 文件加载外部使用记录数据。"""
    global external_data
    if external_data:
        return external_data

    external_data_path = get_abs_path(agent_conf["external_data_path"])

    if not os.path.exists(external_data_path):
        logger.warning(f"外部数据文件不存在: {external_data_path}")
        return external_data

    try:
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f.readlines()[1:], start=2):
                arr = line.strip().split(",")
                if len(arr) < 5:
                    logger.warning(f"CSV 第 {line_num} 行字段不足，跳过")
                    continue

                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time_key = arr[5].replace('"', "") if len(arr) > 5 else "unknown"

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time_key] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }
    except Exception as e:
        logger.error(f"加载外部数据失败: {e}")

    return external_data


@tool
def fetch_external_data(user_id: str, month: str) -> str:
    """从外部系统中获取指定用户在指定月份的使用记录。

    Args:
        user_id: 用户ID（4位数字字符串）。
        month: 月份，格式 YYYY-MM。

    Returns:
        结构化使用记录字符串；未检索到则返回空字符串。
    """
    _generate_external_data()

    try:
        record = external_data[user_id][month]
        return str(record)
    except KeyError:
        logger.warning(f"[fetch_external_data] 未找到用户 {user_id} 在 {month} 的记录")
        return ""


@tool
def fill_context_for_report() -> str:
    """无入参，调用后触发中间件自动为报告生成的场景动态注入上下文信息。

    仅当用户意图为生成/查询个人使用报告时调用。
    """
    return "fill_context_for_report 已调用"
