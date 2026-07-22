"""Measure what the MCP boundary costs per tool call.

Calls the same stubbed tool body two ways — in-process, and through the
real WebResearchMCPClient over stdio — and compares. The stub keeps the
figure to transport + JSON-RPC + dispatch; live Tavily latency (~100-500 ms)
would bury a single-digit-millisecond effect.

Subprocess spawn and handshake are reported separately, not per call: the
client is a process-wide singleton.

    python -m scripts.bench_mcp_overhead [--calls N]
"""
import argparse
import asyncio
import os
import statistics
import time

# The server subprocess inherits this process's stderr and logs every
# CallToolRequest at INFO. Quiet it before the client spawns anything.
os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")

from rag.mcp_client import WebResearchMCPClient  # noqa: E402
from scripts.bench_mcp_stub_server import _stub_tavily_search  # noqa: E402

QUERY = "how does chain-of-thought prompting work?"


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(int(round(pct / 100 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return ordered[index]


def _summarize(label: str, samples_ms: list[float]) -> dict:
    return {
        "label": label,
        "p50": statistics.median(samples_ms),
        "p95": _percentile(samples_ms, 95),
        "mean": statistics.fmean(samples_ms),
    }


async def _time_in_process(calls: int) -> list[float]:
    samples = []
    for _ in range(calls):
        start = time.perf_counter()
        await _stub_tavily_search(QUERY, max_results=5)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


async def _time_over_mcp(client: WebResearchMCPClient, calls: int) -> list[float]:
    samples = []
    for _ in range(calls):
        start = time.perf_counter()
        await client.call("tavily_search", {"query": QUERY, "max_results": 5})
        samples.append((time.perf_counter() - start) * 1000)
    return samples


async def main(calls: int, warmup: int = 5) -> None:
    client = WebResearchMCPClient(server_module="scripts.bench_mcp_stub_server")
    try:
        connect_start = time.perf_counter()
        await client.tool_names()  # spawn + handshake + list_tools
        connect_ms = (time.perf_counter() - connect_start) * 1000

        await _time_in_process(warmup)
        await _time_over_mcp(client, warmup)

        direct = await _time_in_process(calls)
        over_mcp = await _time_over_mcp(client, calls)
    finally:
        await client.aclose()

    rows = [_summarize("in-process", direct), _summarize("over MCP", over_mcp)]

    print(f"\nweb-research tool call — {calls} calls per path, stubbed tool body\n")
    print(f"{'path':<12} {'p50 (ms)':>10} {'p95 (ms)':>10} {'mean (ms)':>10}")
    print("-" * 46)
    for row in rows:
        print(
            f"{row['label']:<12} {row['p50']:>10.3f} {row['p95']:>10.3f} {row['mean']:>10.3f}"
        )

    print("-" * 46)
    print(
        f"{'overhead':<12} {rows[1]['p50'] - rows[0]['p50']:>10.3f} "
        f"{rows[1]['p95'] - rows[0]['p95']:>10.3f} "
        f"{rows[1]['mean'] - rows[0]['mean']:>10.3f}"
    )
    print(
        f"\none-time connect (spawn + handshake + discovery): {connect_ms:.1f} ms, "
        "paid once per process\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=50, help="calls per path (default 50)")
    args = parser.parse_args()
    asyncio.run(main(args.calls))
