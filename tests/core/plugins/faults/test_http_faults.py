import asyncio
import json
import pathlib
import tempfile
import time

import aiohttp
import pytest

import testsuite
from testsuite.mockserver import faults

_REPLAY_LOG = {
    'version': 1,
    'seed': None,
    'armed_at': 0.0,
    'rules': [],
    'decisions': [
        {
            'index': 1,
            'seq': 1,
            'connection': 1,
            'direction': 'request',
            'rule_id': None,
            'action': None,
            'elapsed': 0.0,
            'note': None,
        },
        {
            'index': 2,
            'seq': 1,
            'connection': 1,
            'direction': 'response',
            'rule_id': None,
            'action': {'type': 'delay', 'delay': 0.15},
            'elapsed': 0.0,
            'note': None,
        },
    ],
}

_REPLAY_DIR = tempfile.mkdtemp(prefix='faults-replay-')
_REPLAY_LOG_PATH = pathlib.Path(_REPLAY_DIR) / 'fault-log.json'
_REPLAY_LOG_PATH.write_text(json.dumps(_REPLAY_LOG))


@pytest.fixture
async def http_session():
    async with aiohttp.ClientSession() as session:
        yield session


async def test_no_faults_without_fixture(
    mockserver: testsuite.MockserverFixture,
    http_session: aiohttp.ClientSession,
):
    @mockserver.json_handler('/foo')
    def _handler(request):
        return {'ok': True}

    response = await http_session.get(mockserver.url('/foo'))
    assert response.status == 200
    assert await response.json() == {'ok': True}


async def test_http_delay(
    mockserver: testsuite.MockserverFixture,
    faults: faults.FaultScheduler,
    http_session: aiohttp.ClientSession,
):
    @mockserver.json_handler('/foo')
    def _handler(request):
        return {'ok': True}

    faults.configure(
        seed=1,
        rules=[
            {
                'seq': 1,
                'direction': 'response',
                'action': {'type': 'delay', 'delay': 0.2},
            },
        ],
    )
    started = time.perf_counter()
    response = await http_session.get(mockserver.url('/foo'))
    elapsed = time.perf_counter() - started
    assert response.status == 200
    assert elapsed >= 0.18


async def test_http_error_response_short_circuits_handler(
    mockserver: testsuite.MockserverFixture,
    faults: faults.FaultScheduler,
    http_session: aiohttp.ClientSession,
):
    handler_calls = []

    @mockserver.json_handler('/foo')
    def _handler(request):
        handler_calls.append(1)
        return mockserver.make_response('ok')

    faults.configure(
        rules=[
            {
                'seq': 1,
                'direction': 'request',
                'action': {
                    'type': 'error_response',
                    'status': 503,
                    'body': 'service unavailable',
                },
            },
        ],
    )
    response = await http_session.get(mockserver.url('/foo'))
    assert response.status == 503
    assert await response.text() == 'service unavailable'
    assert not handler_calls


async def test_http_error_response_replaces_response(
    mockserver: testsuite.MockserverFixture,
    faults: faults.FaultScheduler,
    http_session: aiohttp.ClientSession,
):
    @mockserver.json_handler('/foo')
    def _handler(request):
        return {'ok': True}

    faults.configure(
        rules=[
            {
                'seq': 1,
                'direction': 'response',
                'action': {
                    'type': 'error_response',
                    'status': 500,
                    'body': 'injected failure',
                    'headers': {'X-Injected': '1'},
                },
            },
        ],
    )
    response = await http_session.get(mockserver.url('/foo'))
    assert response.status == 500
    assert await response.text() == 'injected failure'
    assert response.headers['X-Injected'] == '1'


async def test_http_reset(
    mockserver: testsuite.MockserverFixture,
    faults: faults.FaultScheduler,
    http_session: aiohttp.ClientSession,
):
    @mockserver.json_handler('/foo')
    def _handler(request):
        return {'ok': True}

    faults.configure(
        rules=[
            {
                'seq': 1,
                'direction': 'response',
                'action': {'type': 'reset'},
            },
        ],
    )
    # POST is not idempotent: per RFC 9112 aiohttp does not retry it,
    # so the reset surfaces as ServerDisconnectedError.
    with pytest.raises(aiohttp.ServerDisconnectedError):
        await http_session.post(mockserver.url('/foo'), json={})


async def test_http_timeout(
    mockserver: testsuite.MockserverFixture,
    faults: faults.FaultScheduler,
    http_session: aiohttp.ClientSession,
):
    @mockserver.json_handler('/foo')
    def _handler(request):
        return {'ok': True}

    faults.configure(
        rules=[
            {
                'seq': 1,
                'direction': 'request',
                'action': {'type': 'timeout', 'timeout': 1.0},
            },
        ],
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            http_session.get(mockserver.url('/foo')),
            timeout=0.2,
        )


@pytest.mark.faults_replay_file(str(_REPLAY_LOG_PATH))
async def test_http_fault_replay(
    faults: faults.FaultScheduler,
    mockserver: testsuite.MockserverFixture,
    http_session: aiohttp.ClientSession,
):
    @mockserver.json_handler('/foo')
    def _handler(request):
        return {'ok': True}

    assert faults.replay_mode
    started = time.perf_counter()
    response = await http_session.get(mockserver.url('/foo'))
    elapsed = time.perf_counter() - started
    assert response.status == 200
    assert elapsed >= 0.13
