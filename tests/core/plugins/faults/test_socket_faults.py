import asyncio
import socket
import time

import pytest

from testsuite import asyncio_socket
from testsuite.mockserver import faults


def _socketpair(*, scheduler_a=None, scheduler_b=None):
    sock_a, sock_b = socket.socketpair()
    return (
        asyncio_socket.from_socket(sock_a, fault_scheduler=scheduler_a),
        asyncio_socket.from_socket(sock_b, fault_scheduler=scheduler_b),
    )


async def test_plain_socket_no_faults():
    writer, reader = _socketpair()
    try:
        await writer.sendall(b'hello')
        assert await reader.recv(1024) == b'hello'
    finally:
        writer.close()
        reader.close()


async def test_short_write():
    scheduler = faults.FaultScheduler(
        seed=1,
        rules=[
            {'seq': 1, 'action': {'type': 'short_write', 'bytes': 4}},
        ],
    )
    writer, reader = _socketpair(scheduler_a=scheduler)
    try:
        sent = await writer.sendall(b'0123456789')
        assert sent == 4
        data = await reader.recv(1024)
        assert data == b'0123'
    finally:
        writer.close()
        reader.close()


async def test_short_write_only_on_second_op():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 2, 'action': {'type': 'short_write', 'bytes': 1}},
        ],
    )
    writer, reader = _socketpair(scheduler_a=scheduler)
    try:
        # Full sends keep sock_sendall semantics (returns None).
        assert await writer.sendall(b'first') is None
        assert await writer.sendall(b'second') == 1
    finally:
        writer.close()
        reader.close()


async def test_half_packet_recv():
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'seq': 1,
                'action': {
                    'type': 'half_packet',
                    'bytes': 3,
                    'delay': 0.05,
                },
            },
        ],
    )
    writer, reader = _socketpair(scheduler_b=scheduler)
    try:
        await writer.sendall(b'0123456789')
        first = await reader.recv(1024)
        assert first == b'012'
        second = await reader.recv(1024)
        assert second == b'3456789'
    finally:
        writer.close()
        reader.close()


async def test_half_packet_send():
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'seq': 1,
                'action': {
                    'type': 'half_packet',
                    'bytes': 4,
                    'delay': 0.1,
                },
            },
        ],
    )
    writer, reader = _socketpair(scheduler_a=scheduler)
    try:
        send_task = asyncio.ensure_future(writer.sendall(b'0123456789'))
        await asyncio.sleep(0.02)
        first = await reader.recv(1024)
        assert first == b'0123'
        second = await reader.recv(1024)
        assert second == b'456789'
        await send_task
    finally:
        writer.close()
        reader.close()


async def test_delay_send():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 1, 'action': {'type': 'delay', 'delay': 0.1}},
        ],
    )
    writer, reader = _socketpair(scheduler_a=scheduler)
    try:
        started = time.perf_counter()
        await writer.sendall(b'hello')
        assert time.perf_counter() - started >= 0.09
        assert await reader.recv(1024) == b'hello'
    finally:
        writer.close()
        reader.close()


async def test_reset_send():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 1, 'action': {'type': 'reset'}},
        ],
    )
    writer, reader = _socketpair(scheduler_a=scheduler)
    try:
        with pytest.raises(faults.FaultConnectionReset):
            await writer.sendall(b'hello')
        assert writer.socket.fileno() == -1
    finally:
        writer.close()
        reader.close()


async def test_reset_recv():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 1, 'direction': faults.RECV,
             'action': {'type': 'reset'}},
        ],
    )
    writer, reader = _socketpair(scheduler_b=scheduler)
    try:
        with pytest.raises(faults.FaultConnectionReset):
            await reader.recv(1024)
    finally:
        writer.close()
        reader.close()


async def test_timeout_send():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 1, 'action': {'type': 'timeout', 'timeout': 0.05}},
        ],
    )
    writer, reader = _socketpair(scheduler_a=scheduler)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await writer.sendall(b'hello')
    finally:
        writer.close()
        reader.close()


async def test_direction_isolation():
    # Reset rule applies to recv, sends pass through.
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'direction': faults.RECV,
                'action': {'type': 'reset'},
            },
        ],
    )
    writer, reader = _socketpair(scheduler_b=scheduler)
    try:
        await writer.sendall(b'x')
        with pytest.raises(faults.FaultConnectionReset):
            await reader.recv(1024)
    finally:
        writer.close()
        reader.close()


async def test_loop_auto_discovery():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 1, 'action': {'type': 'short_write', 'bytes': 2}},
        ],
    )
    scheduler.arm()
    writer, reader = _socketpair()
    try:
        assert await writer.sendall(b'hello') == 2
    finally:
        writer.close()
        reader.close()
        scheduler.disarm()


async def test_disarmed_loop_has_no_scheduler():
    scheduler = faults.FaultScheduler()
    scheduler.arm()
    loop = asyncio.get_running_loop()
    assert faults.scheduler_for_loop(loop) is scheduler
    scheduler.disarm()
    assert faults.scheduler_for_loop(loop) is None
