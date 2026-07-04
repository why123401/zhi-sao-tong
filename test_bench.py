"""API 压力测试脚本

用法:
    python test_bench.py                  # 默认测试 /api/chat
    python test_bench.py --endpoint rag   # 测试 /api/rag/chat
    python test_bench.py --concurrency 10 --requests 100

输出:
    平均延迟、P50/P95/P99 延迟、吞吐量(QPS)、成功率
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class LatencyStats:
    """延迟统计"""
    count: int = 0
    total_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(self.count, 1)

    @property
    def p50_ms(self) -> float:
        return sorted(self.latencies)[len(self.latencies) // 2] if self.latencies else 0

    @property
    def p95_ms(self) -> float:
        idx = int(len(self.latencies) * 0.95)
        return sorted(self.latencies)[idx] if self.latencies else 0

    @property
    def p99_ms(self) -> float:
        idx = int(len(self.latencies) * 0.99)
        return sorted(self.latencies)[idx] if self.latencies else 0

    @property
    def qps(self) -> float:
        if self.count == 0:
            return 0
        return self.count / (self.total_ms / 1000)


BASE_URL = "http://localhost:8000"

ENDPOINTS = {
    "chat": {
        "path": "/api/chat",
        "method": "POST",
        "body": {"message": "小户型适合什么扫地机器人"},
    },
    "rag": {
        "path": "/api/rag/chat",
        "method": "POST",
        "body": {"message": "扫地机器人怎么保养"},
    },
    "health": {
        "path": "/health",
        "method": "GET",
        "body": None,
    },
}


async def single_request(client: httpx.AsyncClient, endpoint: dict) -> float | None:
    """发送单次请求，返回延迟(ms)，失败返回 None"""
    try:
        start = time.perf_counter()
        if endpoint["method"] == "GET":
            resp = await client.get(BASE_URL + endpoint["path"])
        else:
            resp = await client.post(
                BASE_URL + endpoint["path"],
                json=endpoint["body"],
                timeout=30.0,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            return elapsed_ms
        else:
            print(f"  [WARN] 状态码 {resp.status_code}: {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


async def benchmark(
    client: httpx.AsyncClient,
    endpoint: dict,
    num_requests: int,
    concurrency: int,
) -> LatencyStats:
    """并发压测"""
    stats = LatencyStats()
    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0

    async def worker(idx: int):
        nonlocal success_count
        async with semaphore:
            latency = await single_request(client, endpoint)
            if latency is not None:
                stats.count += 1
                stats.total_ms += latency
                stats.latencies.append(latency)
                success_count += 1
                if (idx + 1) % 10 == 0 or idx == 0:
                    print(f"  [{idx + 1}/{num_requests}] 成功", end="\r")

    print(f"  开始压测: {num_requests} 请求, 并发度 {concurrency}")
    tasks = [worker(i) for i in range(num_requests)]
    await asyncio.gather(*tasks)

    stats.count = success_count
    return stats


def print_report(stats: LatencyStats, endpoint_name: str):
    """打印测试报告"""
    print("\n" + "=" * 60)
    print(f"  压测报告: {endpoint_name}")
    print("=" * 60)
    print(f"  请求数:      {stats.count}")
    print(f"  平均延迟:    {stats.avg_ms:.2f} ms")
    print(f"  P50 延迟:    {stats.p50_ms:.2f} ms")
    print(f"  P95 延迟:    {stats.p95_ms:.2f} ms")
    print(f"  P99 延迟:    {stats.p99_ms:.2f} ms")
    print(f"  吞吐量:      {stats.qps:.2f} QPS")
    print("=" * 60)


async def main(endpoint_name: str, num_requests: int, concurrency: int):
    if endpoint_name not in ENDPOINTS:
        print(f"未知端点: {endpoint_name}。可选: {list(ENDPOINTS.keys())}")
        return

    ep = ENDPOINTS[endpoint_name]
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # 先检查服务是否在线
        try:
            await client.get(BASE_URL + "/health", timeout=5.0)
        except Exception:
            print(f"无法连接到 {BASE_URL}，请先启动服务: python api/server.py")
            return

        stats = await benchmark(client, ep, num_requests, concurrency)
        print_report(stats, ep["path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API 压力测试")
    parser.add_argument(
        "--endpoint", default="chat",
        choices=list(ENDPOINTS.keys()),
        help="要测试的端点",
    )
    parser.add_argument("--requests", type=int, default=20, help="总请求数")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数")
    args = parser.parse_args()

    asyncio.run(main(args.endpoint, args.requests, args.concurrency))
