# pylint: disable=protected-access
import asyncio
import time

import aiohttp
import pytest

import testsuite
from testsuite.mockserver import scenarios

CLIENT_HEADER = scenarios.DEFAULT_CLIENT_ID_HEADER


class Client:
    def __init__(self, *, base_url, session):
        self._session = session
        self._base_url = base_url

    def request(self, method, path, **kwargs):
        url = '{}/{}'.format(self._base_url.rstrip('/'), path.lstrip('/'))
        kwargs.setdefault('raise_for_status', False)
        return self._session.request(method, url, **kwargs)

    def get(self, path, **kwargs):
        return self.request('GET', path, **kwargs)

    def post(self, path, **kwargs):
        return self.request('POST', path, **kwargs)


@pytest.fixture
async def mockserver_client(mockserver: testsuite.MockserverFixture):
    async with aiohttp.ClientSession() as session:
        yield Client(base_url=mockserver.base_url, session=session)


def _ordered_doc():
    return {
        'states': {
            'initial': [
                {
                    'match': {'path': '/foo'},
                    'times': 2,
                    'response': {'json': {'step': 1}},
                },
                {
                    'match': {'path': '/foo'},
                    'response': {'json': {'step': 2}},
                },
            ],
        },
    }


async def test_scenario_ordered_steps(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    handle = mockserver.scenarios.load(_ordered_doc())

    for expected in (1, 1, 2, 2):
        response = await mockserver_client.get('/foo')
        assert response.status == 200
        assert await response.json() == {'step': expected}

    assert handle.times_called == 4
    assert handle.client_state('127.0.0.1').state == 'initial'


async def test_scenario_client_isolation(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    handle = mockserver.scenarios.load(_ordered_doc())

    async def get_as(client_id):
        response = await mockserver_client.get(
            '/foo', headers={CLIENT_HEADER: client_id}
        )
        return await response.json()

    assert await get_as('client-a') == {'step': 1}
    assert await get_as('client-b') == {'step': 1}
    assert await get_as('client-a') == {'step': 1}
    assert await get_as('client-a') == {'step': 2}
    assert await get_as('client-b') == {'step': 1}

    assert set(handle.client_ids) == {'client-a', 'client-b'}
    assert handle.client_state('client-a').step_index == 1
    assert handle.client_state('client-b').step_index == 1
    assert handle.client_state('client-b').step_count == 0
    assert handle.client_state('client-b').total_calls == 2


async def test_scenario_state_transition(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    handle = mockserver.scenarios.load(
        {
            'initial_state': 'a',
            'states': {
                'a': [
                    {
                        'match': {'path': '/foo'},
                        'response': {'json': {'state': 'a'}},
                        'next_state': 'b',
                    },
                ],
                'b': [
                    {
                        'match': {'path': '/foo'},
                        'times': 2,
                        'response': {'json': {'state': 'b'}},
                        'next_state': 'a',
                    },
                ],
            },
        },
    )

    for expected in ('a', 'b', 'b', 'a', 'b'):
        response = await mockserver_client.get(
            '/foo', headers={CLIENT_HEADER: 'c'}
        )
        assert await response.json() == {'state': expected}

    assert handle.client_state('c').state == 'b'
    assert handle.client_state('c').step_count == 1


async def test_scenario_unmatched_fallback(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {'method': 'POST', 'path': '/scn'},
                        'response': {'json': {'from': 'scenario'}},
                    },
                ],
            },
        },
    )

    @mockserver.json_handler('/other')
    def _other(request):
        return {'from': 'handler'}

    # Same path, method not matching current step -> regular handler wins.
    @mockserver.json_handler('/scn')
    def _scn(request):
        return {'from': 'handler'}

    response = await mockserver_client.post('/scn')
    assert await response.json() == {'from': 'scenario'}

    response = await mockserver_client.get('/scn')
    assert await response.json() == {'from': 'handler'}

    response = await mockserver_client.get('/other')
    assert await response.json() == {'from': 'handler'}


async def test_scenario_delay(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/slow'},
                        'delay_ms': 50,
                        'response': {'json': {'ok': True}},
                    },
                ],
            },
        },
    )

    started = time.perf_counter()
    response = await mockserver_client.get('/slow')
    elapsed = time.perf_counter() - started
    assert response.status == 200
    assert elapsed >= 0.045


async def test_scenario_template(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {
                            'method': 'POST',
                            'path_regex': r'/order/(?P<order_id>\w+)',
                        },
                        'response': {
                            'json': {
                                'order_id': '{{order_id}}',
                                'name': '{{json.name}}',
                                'method': '{{method}}',
                                'nested': '{{json.items[0]}}',
                            },
                            'headers': {'X-Echo-Kind': '{{query.kind}}'},
                        },
                    },
                ],
            },
        },
    )

    response = await mockserver_client.post(
        '/order/abc123',
        json={'name': 'widget', 'items': ['first']},
        params={'kind': 'fast'},
    )
    assert response.status == 200
    assert await response.json() == {
        'order_id': 'abc123',
        'name': 'widget',
        'method': 'POST',
        'nested': 'first',
    }
    assert response.headers['X-Echo-Kind'] == 'fast'


async def test_scenario_close_connection(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/close'},
                        'close_connection': True,
                        'response': {'json': {'ok': True}},
                    },
                ],
            },
        },
    )

    response = await mockserver_client.get('/close')
    assert response.status == 200
    assert await response.json() == {'ok': True}
    assert response.headers.get('Connection', '').lower() == 'close'

    # Server is still operational; terminal step keeps serving.
    response = await mockserver_client.get('/close')
    assert response.status == 200


async def test_scenario_abort_connection(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/abort'},
                        'close_connection': 'abort',
                        'response': {'json': {'ok': True}},
                    },
                ],
            },
        },
    )

    @mockserver.json_handler('/alive')
    def _alive(request):
        return {'alive': True}

    with pytest.raises(
        (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError),
    ):
        response = await mockserver_client.get('/abort')
        await response.read()

    response = await mockserver_client.get('/alive')
    assert await response.json() == {'alive': True}


@pytest.mark.parametrize(
    'doc',
    [
        'not-a-mapping',
        {'states': {}},
        {'states': {'initial': []}},
        {'states': {'initial': [{'match': {'path': 1}}]}},
        {
            'states': {
                'initial': [{'match': {'path': '/f', 'path_prefix': '/g'}}]
            }
        },
        {'states': {'initial': [{'match': {'path': '/f'}, 'times': 0}]}},
        {'states': {'initial': [{'match': {'path': '/f'}, 'times': 'x'}]}},
        {
            'states': {
                'initial': [
                    {'match': {'path': '/f'}, 'delay': 1, 'delay_ms': 1},
                ],
            },
        },
        {
            'states': {
                'initial': [
                    {'match': {'path': '/f'}, 'delay': 10000},
                ],
            },
        },
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/f'},
                        'response': {'status': 99},
                    },
                ],
            },
        },
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/f'},
                        'response': {'json': {}, 'text': 'x'},
                    },
                ],
            },
        },
        {
            'states': {
                'initial': [
                    {'match': {'path': '/f'}, 'unknown_key': 1},
                ],
            },
        },
        {
            'initial_state': 'missing',
            'states': {
                'initial': [{'match': {'path': '/f'}}],
            },
        },
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/f'},
                        'next_state': 'nowhere',
                    },
                ],
            },
        },
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/f', 'path_regex': '['},
                    },
                ],
            },
        },
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/f', 'headers': {'X': 1}},
                    },
                ],
            },
        },
    ],
)
async def test_invalid_scenario_schema(
    mockserver: testsuite.MockserverFixture,
    doc,
):
    with pytest.raises(scenarios.ScenarioSpecError):
        mockserver.scenarios.load(doc)


async def test_scenario_script_error_rollback(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
    mockserver_errors_pop,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/broken'},
                        'response': {'text': 'hello {{missing_var}}'},
                    },
                    {
                        'match': {'path': '/broken'},
                        'response': {'json': {'ok': True}},
                    },
                ],
            },
        },
    )

    @mockserver.json_handler('/alive')
    def _alive(request):
        return {'alive': True}

    for _ in range(2):
        response = await mockserver_client.get('/broken')
        assert response.status == 500
        error = mockserver_errors_pop()
        assert isinstance(error, scenarios.ScenarioError)

    # State was rolled back: scenario still on its first step,
    # but the server itself is fully operational.
    response = await mockserver_client.get('/alive')
    assert await response.json() == {'alive': True}


async def test_scenario_concurrency(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/foo'},
                        'times': 3,
                        'response': {'json': {'from': 'scenario'}},
                    },
                    # After step is exhausted, requests to /foo no longer
                    # match the scenario and fall through to the handler.
                    {
                        'match': {'path': '/bar'},
                        'response': {'json': {}},
                    },
                ],
            },
        },
    )

    @mockserver.json_handler('/foo')
    def _foo(request):
        return {'from': 'handler'}

    async def request():
        response = await mockserver_client.get('/foo')
        assert response.status == 200
        return await response.json()

    results = await asyncio.gather(*(request() for _ in range(10)))

    assert results.count({'from': 'scenario'}) == 3
    assert results.count({'from': 'handler'}) == 7


async def test_scenario_client_timeout(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
    mockserver_errors_pop,
):
    mockserver.scenarios.load(
        {
            'states': {
                'initial': [
                    {
                        'match': {'path': '/slow'},
                        'delay': 1.0,
                        'times': 1,
                        'response': {'json': {'step': 1}},
                    },
                    {
                        'match': {'path': '/slow'},
                        'response': {'json': {'step': 2}},
                    },
                ],
            },
        },
    )

    with pytest.raises(asyncio.TimeoutError):
        await mockserver_client.get(
            '/slow',
            timeout=aiohttp.ClientTimeout(total=0.05),
        )

    # Cancelled request consumed its claim: state machine moved forward,
    # server keeps serving.
    response = await mockserver_client.get('/slow')
    assert await response.json() == {'step': 2}


@pytest.mark.parametrize('client_id_from', ['shared', 'peer'])
async def test_scenario_shared_client(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
    client_id_from,
):
    handle = mockserver.scenarios.load(
        {
            'client_id_from': client_id_from,
            'states': {
                'initial': [
                    {
                        'match': {'path': '/s'},
                        'times': 1,
                        'response': {'json': {'step': 1}},
                    },
                    {
                        'match': {'path': '/s'},
                        'response': {'json': {'step': 2}},
                    },
                ],
            },
        },
    )

    response = await mockserver_client.get('/s', headers={CLIENT_HEADER: 'one'})
    assert await response.json() == {'step': 1}

    # Different header does not isolate: all requests share one machine.
    response = await mockserver_client.get('/s', headers={CLIENT_HEADER: 'two'})
    assert await response.json() == {'step': 2}

    expected_ids = [''] if client_id_from == 'shared' else ['127.0.0.1']
    assert handle.client_ids == expected_ids


async def test_scenario_header_required_fallback(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'client_id_from': 'header',
            'states': {
                'initial': [
                    {
                        'match': {'path': '/h'},
                        'response': {'json': {'from': 'scenario'}},
                    },
                ],
            },
        },
    )

    @mockserver.json_handler('/h')
    def _h(request):
        return {'from': 'handler'}

    response = await mockserver_client.get(
        '/h', headers={CLIENT_HEADER: 'present'}
    )
    assert await response.json() == {'from': 'scenario'}

    # Header missing -> scenario does not claim the request.
    response = await mockserver_client.get('/h')
    assert await response.json() == {'from': 'handler'}


async def test_scenario_load_file(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
    tmp_path,
):
    path = tmp_path / 'scenario.yaml'
    path.write_text(
        'id: from-file\n'
        'states:\n'
        '  initial:\n'
        '    - match: {path: /f}\n'
        '      response: {json: {x: 1}}\n',
    )

    handle = mockserver.scenarios.load_file(str(path))
    assert handle.id == 'from-file'

    response = await mockserver_client.get('/f')
    assert await response.json() == {'x': 1}


async def test_scenario_cleanup_part1(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    mockserver.scenarios.load(
        {
            'id': 'leak-check',
            'states': {
                'initial': [
                    {
                        'match': {'path': '/cleanup'},
                        'response': {'json': {'from': 'scenario'}},
                    },
                ],
            },
        },
    )

    response = await mockserver_client.get('/cleanup')
    assert await response.json() == {'from': 'scenario'}


async def test_scenario_cleanup_part2(
    mockserver: testsuite.MockserverFixture,
    mockserver_client: Client,
):
    @mockserver.json_handler('/cleanup')
    def _cleanup(request):
        return {'from': 'handler'}

    response = await mockserver_client.get('/cleanup')
    assert await response.json() == {'from': 'handler'}
