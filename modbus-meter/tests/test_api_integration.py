"""API 端到端集成测试。

在同一事件循环内启动 pymodbus 内存模拟表（随机本地端口），
通过 httpx ASGI 传输调用 FastAPI，不依赖 Docker 或固定端口。
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

import httpx
import pytest
from pymodbus.server import StartAsyncTcpServer

from app.main import ReadRequest, app
from simulator.meter_sim import device


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def meter_port() -> int:
    port = _free_port()
    task = asyncio.create_task(
        StartAsyncTcpServer(context=device, address=("127.0.0.1", port))
    )
    for _ in range(50):
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), timeout=0.2
            )
            writer.close()
            break
        except OSError:
            await asyncio.sleep(0.1)
    else:
        task.cancel()
        pytest.fail("模拟 Modbus 表启动超时")
    try:
        yield port
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_read_meter_happy_path(meter_port: int) -> None:
    payload = {
        "points": [
            {"address": 0, "words": 2},
            {"address": 2, "words": 2},
            {"address": 10, "words": 1},
        ],
        "range_start": 0,
        "range_end": 99,
        "max_read": 8,
        "host": "127.0.0.1",
        "port": meter_port,
    }
    async with await _client() as client:
        resp = await client.post("/read", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # 0..3 一段、10 一段 -> 2 次；读 4 + 1 = 5 个寄存器
    assert data["request_count"] == 2
    assert data["total_registers"] == 5
    values = {p["index"]: p["value"] for p in data["points"]}
    assert values == {0: 100000, 1: 100, 2: 1234}
    assert data["segments"][0]["registers"] == [1, 34464, 0, 100]


async def test_infeasible_plan_returns_422_without_device_call(meter_port: int) -> None:
    payload = {
        "points": [{"address": 200, "words": 1}],
        "range_start": 0,
        "range_end": 99,
        "max_read": 8,
        "host": "127.0.0.1",
        "port": meter_port,
    }
    async with await _client() as client:
        resp = await client.post("/read", json=payload)
    assert resp.status_code == 422
    assert "无可行方案" in resp.json()["detail"]


async def test_connection_failure_returns_502_no_partial_reads() -> None:
    dead_port = _free_port()  # 无人监听
    payload = {
        "points": [
            {"address": 0, "words": 1},
            {"address": 20, "words": 1},
        ],
        "range_start": 0,
        "range_end": 99,
        "max_read": 8,
        "host": "127.0.0.1",
        "port": dead_port,
        "timeout": 0.5,
    }
    async with await _client() as client:
        resp = await client.post("/read", json=payload)
    assert resp.status_code == 502
    body = resp.json()
    assert "连接" in body["detail"]
    assert "segments" not in body  # 不返回部分读数


async def test_duplicate_single_words_and_same_address_double_word(meter_port: int) -> None:
    # 与 Swagger 示例一致的场景：地址 10 两条单字 + 一条双字（同址不同字数，
    # 各自按自己的字数解码），再加 12、20 的单字
    payload = {
        "points": [
            {"address": 10, "words": 1},
            {"address": 10, "words": 1},
            {"address": 10, "words": 2},
            {"address": 12, "words": 1},
            {"address": 20, "words": 1},
        ],
        "range_start": 0,
        "range_end": 99,
        "max_read": 8,
        "host": "127.0.0.1",
        "port": meter_port,
    }
    async with await _client() as client:
        resp = await client.post("/read", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # 10..12 一段（含同址单字/双字与地址 12），20 一段 -> 2 次、3 + 1 = 4 字
    assert data["request_count"] == 2
    assert data["total_registers"] == 4
    first, second = data["segments"]
    assert (first["start"], first["end"], first["count"]) == (10, 12, 3)
    assert first["point_indexes"] == [0, 1, 2, 3]  # 每条重复测点的原下标都在
    assert (second["start"], second["end"], second["point_indexes"]) == (20, 20, [4])

    # 模拟表：10..13 = 1234, 5678, 65535, 42；地址 20 = (20*37+11)&0xFFFF
    values = {p["index"]: p["value"] for p in data["points"]}
    assert values == {
        0: 1234,                 # 单字
        1: 1234,                 # 重复单字，读数一致
        2: (1234 << 16) | 5678,  # 同址双字，高字在前拼接
        3: 65535,
        4: (20 * 37 + 11) & 0xFFFF,
    }
    # points 按提交顺序（原下标升序）逐条返回
    assert [p["index"] for p in data["points"]] == [0, 1, 2, 3, 4]
    assert [p["words"] for p in data["points"]] == [1, 1, 2, 1, 1]


async def test_heavy_planning_runs_off_event_loop(meter_port: int, monkeypatch) -> None:
    # 大批重复测点的规划必须放到工作线程（asyncio.to_thread）：规划期间事件循环
    # 仍能调度其他协程，其他电表的小批请求不会被拖住。
    # 注：httpx 的 ASGITransport 在同一事件循环内串行化请求，无法用来压并发，
    # 因此直接验证“规划未跑在事件循环线程上”这一实现约定。
    import threading

    from app import main as main_module

    real_plan = main_module.plan_reads
    entered = threading.Event()
    release = threading.Event()
    loop_thread_id = threading.get_ident()
    planner_thread_id: int | None = None

    def slow_plan(*args, **kwargs):
        nonlocal planner_thread_id
        planner_thread_id = threading.get_ident()
        entered.set()
        if not release.wait(timeout=30):  # 挂住工作线程；若在 loop 线程上会卡死整个循环
            raise RuntimeError("测试同步事件超时")
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(main_module, "plan_reads", slow_plan)

    heavy_payload = ReadRequest(
        points=[{"address": 50, "words": 1}] * 5000,
        range_start=0,
        range_end=99,
        max_read=8,
        host="127.0.0.1",
        port=meter_port,
    )

    heavy_task = asyncio.create_task(main_module.read_meter(heavy_payload))
    # 轮询而不是阻塞等待：必须不断 await 让事件循环调度 read_meter，
    # to_thread 的工作线程才会拿到规划任务
    deadline = asyncio.get_running_loop().time() + 10
    while not entered.is_set() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    if not entered.is_set():
        pytest.fail("大批规划任务未在预期时间内启动")

    # 规划挂在工作线程上时，事件循环依然可调度
    loop_alive = await asyncio.wait_for(asyncio.sleep(0.05), timeout=2)
    assert loop_alive is None
    assert planner_thread_id is not None
    assert planner_thread_id != loop_thread_id

    release.set()
    response = await asyncio.wait_for(heavy_task, timeout=30)
    assert response.request_count == 1
    assert sum(len(s.point_indexes) for s in response.segments) == 5000


def test_swagger_schema_carries_duplicate_points_example() -> None:
    schema = app.openapi()
    # Pydantic v2 / OpenAPI 3.1 下 examples 为示例对象列表
    examples = schema["components"]["schemas"]["ReadRequest"]["examples"]
    body = next(iter(examples))["value"]
    words_at_10 = [p["words"] for p in body["points"] if p["address"] == 10]
    assert words_at_10.count(1) >= 2  # 重复单字
    assert 2 in words_at_10            # 同址双字
