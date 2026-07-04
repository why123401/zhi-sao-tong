"""
企业级 ModelRouter —— 三级模型路由 + 熔断器 + 成本追踪

设计目标:
- 当主模型（qwen3-max）不可用时，自动降级到备用模型
- 熔断器保护：每个模型独立熔断，互不影响
- 成本追踪：记录每次调用的 token 用量和费用
"""

from __future__ import annotations

import asyncio
import enum
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

from utils.logger_handler import logger


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------


class CircuitState(enum.Enum):
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 熔断
    HALF_OPEN = "half_open" # 半开（试探恢复）


class Priority(enum.IntEnum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class CostRecord:
    """单次调用的成本记录"""
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    latency_ms: float = 0.0
    cost_cny: float = 0.0


@dataclass
class RouterMetrics:
    """路由汇总指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost_cny: float = 0.0
    records: list[CostRecord] = field(default_factory=list)

    # 按模型的统计
    model_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, rec: CostRecord) -> None:
        self.records.append(rec)
        self.total_requests += 1
        if rec.success:
            self.successful_requests += 1
            self.total_cost_cny += rec.cost_cny
        else:
            self.failed_requests += 1

        stats = self.model_stats.setdefault(rec.model_name, {
            "success": 0, "fail": 0, "tokens_in": 0, "tokens_out": 0,
        })
        if rec.success:
            stats["success"] += 1
        else:
            stats["fail"] += 1
        stats["tokens_in"] += rec.input_tokens
        stats["tokens_out"] += rec.output_tokens


# ---------------------------------------------------------------------------
# 熔断器（每个模型独立维护）
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    熔断器（Circuit Breaker Pattern）

    - 连续失败 >= failure_threshold  → OPEN
    - OPEN 持续时间 >= recovery_timeout  → HALF_OPEN
    - HALF_OPEN 下一次成功  → CLOSED
    - HALF_OPEN 下一次失败  → OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,  # 秒
    ) -> None:
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                assert self._last_failure_time is not None
                if (time.time() - self._last_failure_time) >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        return self.state != CircuitState.OPEN


# ---------------------------------------------------------------------------
# 价格表（元/百万 token）
# ---------------------------------------------------------------------------

_PRICE_TABLE: dict[str, dict[str, float]] = {
    "qwen3-max":        {"input": 2.0,   "output": 8.0},
    "qwen-plus":        {"input": 0.5,   "output": 2.0},
    "qwen-turbo":       {"input": 0.1,   "output": 0.3},
    "qwen-max":         {"input": 5.0,   "output": 20.0},
}


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """估算费用（元），找不到价格表则给 0 占位"""
    prices = _PRICE_TABLE.get(model_name, {"input": 0, "output": 0})
    return (input_tokens / 1_000_000) * prices["input"] + \
           (output_tokens / 1_000_000) * prices["output"]


# ---------------------------------------------------------------------------
# ModelRouter —— 核心
# ---------------------------------------------------------------------------


class ModelRouter:
    """
    三级模型路由 + 熔断器

    降级链: qwen3-max (HIGH) → qwen-plus (MEDIUM) → qwen-turbo (LOW)

    每个模型拥有独立的熔断器，互不影响。
    只有当前优先级的模型熔断后才降级到下一级。

    使用方式:
        router = ModelRouter()
        router.register_model(chat_model, Priority.HIGH, "qwen3-max")
        router.register_model(fallback_model, Priority.MEDIUM, "qwen-plus")
        ...
        response = router.invoke(messages)
    """

    def __init__(self) -> None:
        self._models: list[tuple[BaseChatModel, Priority, str, CircuitBreaker]] = []
        self._metrics = RouterMetrics()
        self._lock = threading.Lock()
        self._last_model_used: str = "unknown"

    def register_model(
        self,
        model: BaseChatModel,
        priority: Priority,
        model_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        """注册一个候选模型（含独立熔断器）"""
        cb = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        with self._lock:
            self._models.append((model, priority, model_name, cb))
            self._models.sort(key=lambda x: x[1])

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> str:
        """
        依次尝试每个模型，走熔断器保护。
        返回 LLM 的文本输出。
        """
        start = time.time()
        last_error: Optional[Exception] = None

        for model, _priority, model_name, cb in self._models:
            # 熔断检查（每个模型独立）
            if not cb.allow_request():
                continue

            try:
                result = model.invoke(messages, **kwargs)
                latency_ms = (time.time() - start) * 1000

                # 估算 token（简单按字符数 / 1.3）
                input_text = " ".join(
                    m.content for m in messages if hasattr(m, "content") and m.content
                )
                output_text = result.content if isinstance(result.content, str) else ""
                input_tokens = max(1, len(input_text) // 1.3)
                output_tokens = max(1, len(output_text) // 1.3)

                cost = estimate_cost(model_name, input_tokens, output_tokens)
                rec = CostRecord(
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    success=True,
                    cost_cny=cost,
                )
                self._metrics.record(rec)
                cb.record_success()
                self._last_model_used = model_name
                return output_text

            except Exception as e:
                last_error = e
                cb.record_failure()

        # 全部失败
        latency_ms = (time.time() - start) * 1000
        self._metrics.record(CostRecord(
            model_name="unknown",
            latency_ms=latency_ms,
            success=False,
        ))
        raise RuntimeError(
            f"ModelRouter 所有模型调用失败: {last_error}"
        ) from last_error

    def get_metrics(self) -> dict[str, Any]:
        """暴露给监控接口的指标摘要"""
        stats = {}
        for name, s in self._metrics.model_stats.items():
            total = s["success"] + s["fail"]
            stats[name] = {
                **s,
                "total": total,
                "success_rate": round(s["success"] / max(total, 1), 4),
            }
        return {
            "circuit_state": "closed",  # 兼容旧接口，实际每个模型独立
            "total_requests": self._metrics.total_requests,
            "successful_requests": self._metrics.successful_requests,
            "failed_requests": self._metrics.failed_requests,
            "total_cost_cny": round(self._metrics.total_cost_cny, 6),
            "success_rate": round(
                self._metrics.successful_requests / max(self._metrics.total_requests, 1), 4
            ),
            "model_stats": stats,
        }

    def get_primary_model(self) -> Optional[BaseChatModel]:
        """返回最高优先级模型（公开 API，替代直接访问 _models）"""
        return self._models[0][0] if self._models else None

    @property
    def model_count(self) -> int:
        """已注册模型数量"""
        return len(self._models)

    @property
    def last_model_used(self) -> str:
        """最近一次成功调用的模型名称"""
        return self._last_model_used

    @property
    def model_breakers(self) -> dict[str, CircuitBreaker]:
        """暴露每个模型的独立熔断器状态（供诊断用）"""
        return {name: cb for _, _, name, cb in self._models}

    # ── 异步方法 ──────────────────────────────────────────

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> str:
        """
        异步版本的 invoke，使用 await model.ainvoke() 避免阻塞事件循环。

        降级链: 高优先级 -> 中优先级 -> 低优先级
        每个模型独立熔断器保护。
        """
        start = time.time()
        last_error: Optional[Exception] = None

        for model, _priority, model_name, cb in self._models:
            if not cb.allow_request():
                continue

            try:
                result = await model.ainvoke(messages, **kwargs)
                latency_ms = (time.time() - start) * 1000

                input_text = " ".join(
                    m.content for m in messages if hasattr(m, "content") and m.content
                )
                output_text = result.content if isinstance(result.content, str) else ""
                input_tokens = max(1, len(input_text) // 1.3)
                output_tokens = max(1, len(output_text) // 1.3)

                cost = estimate_cost(model_name, input_tokens, output_tokens)
                rec = CostRecord(
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    success=True,
                    cost_cny=cost,
                )
                self._metrics.record(rec)
                cb.record_success()
                self._last_model_used = model_name
                return output_text

            except Exception as e:
                last_error = e
                cb.record_failure()

        # 全部失败
        latency_ms = (time.time() - start) * 1000
        self._metrics.record(CostRecord(
            model_name="unknown",
            latency_ms=latency_ms,
            success=False,
        ))
        raise RuntimeError(
            f"ModelRouter 所有模型调用失败: {last_error}"
        ) from last_error

    async def astream(self, messages: list[BaseMessage], **kwargs: Any):
        """
        异步流式生成器，遍历所有模型走降级链。

        Yields:
            str: 单个 token（文本片段）
        """
        if not self._models:
            raise RuntimeError("ModelRouter 没有注册任何模型")

        last_error: Optional[Exception] = None
        for model, _priority, model_name, cb in self._models:
            # 熔断检查
            if not cb.allow_request():
                logger.warning(f"[astream] 模型 {model_name} 熔断器打开，跳过")
                continue

            try:
                if hasattr(model, "astream"):
                    async for chunk in model.astream(messages, **kwargs):
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if token:
                            yield token
                    cb.record_success()
                    return  # 成功则停止
                else:
                    # 降级：同步调用后逐字符 yield
                    answer = await self.ainvoke(messages, **kwargs)
                    for char in answer:
                        yield char
                    cb.record_success()
                    return

            except Exception as e:
                last_error = e
                cb.record_failure()
                logger.warning(f"[astream] 模型 {model_name} 流式失败: {e}，尝试降级")

        # 全部失败
        raise RuntimeError(f"ModelRouter astream 所有模型调用失败: {last_error}") from last_error
