# pylint: disable=protected-access
import json

import pytest

from testsuite.mockserver import faults


def test_unconfigured_no_faults():
    scheduler = faults.FaultScheduler()
    for seq in range(1, 4):
        decision = scheduler.evaluate(
            seq=seq,
            connection=1,
            direction=faults.REQUEST,
        )
        assert decision.action is None
        assert decision.rule_id is None
    # Fault-free evaluations are still recorded.
    assert len(scheduler.decisions) == 3


def test_configure_by_seq():
    scheduler = faults.FaultScheduler(
        seed=1,
        rules=[
            {'seq': 2, 'action': {'type': 'delay', 'delay': 0.1}},
        ],
    )
    first = scheduler.evaluate(seq=1, connection=1, direction=faults.REQUEST)
    second = scheduler.evaluate(seq=2, connection=1, direction=faults.REQUEST)
    assert first.action is None
    assert second.action == {'type': 'delay', 'delay': 0.1}


def test_seq_list_and_range():
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'seq': {'gte': 2, 'lte': 3},
                'action': {'type': 'reset'},
            },
        ],
    )

    def evaluate(seq):
        return scheduler.evaluate(
            seq=seq,
            connection=1,
            direction=faults.REQUEST,
        ).action

    assert evaluate(1) is None
    assert evaluate(2) == {'type': 'reset'}
    assert evaluate(3) == {'type': 'reset'}
    assert evaluate(4) is None


def test_seed_determinism():
    rules = [
        {
            'direction': faults.RESPONSE,
            'action': {
                'type': 'delay',
                'delay': {'min': 0.0, 'max': 1.0},
            },
        },
    ]
    values = []
    for _ in range(2):
        scheduler = faults.FaultScheduler(seed=42, rules=rules)
        decision = scheduler.evaluate(
            seq=1,
            connection=1,
            direction=faults.RESPONSE,
        )
        values.append(decision.action['delay'])
    assert values[0] == values[1]

    other = faults.FaultScheduler(seed=43, rules=rules)
    decision = other.evaluate(
        seq=1,
        connection=1,
        direction=faults.RESPONSE,
    )
    assert decision.action['delay'] != values[0]


def test_request_and_op_counters():
    scheduler = faults.FaultScheduler()
    assert scheduler.next_request_seq('c1') == 1
    assert scheduler.next_request_seq('c2') == 2
    assert scheduler.next_op_index('c1', faults.SEND) == 1
    assert scheduler.next_op_index('c1', faults.SEND) == 2
    assert scheduler.next_op_index('c1', faults.RECV) == 1
    assert scheduler.on_connection('c1') == 1
    assert scheduler.on_connection('c2') == 2


def test_connection_and_direction_matchers():
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'connection': 2,
                'direction': faults.SEND,
                'action': {'type': 'reset'},
            },
        ],
    )
    assert (
        scheduler.evaluate(
            seq=1,
            connection=1,
            direction=faults.SEND,
        ).action
        is None
    )
    assert (
        scheduler.evaluate(
            seq=1,
            connection=2,
            direction=faults.RECV,
        ).action
        is None
    )
    decision = scheduler.evaluate(
        seq=1,
        connection=2,
        direction=faults.SEND,
    )
    assert decision.action == {'type': 'reset'}


async def test_time_window_matcher():
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'time_window': {'start': 0.05, 'end': 0.25},
                'action': {'type': 'delay', 'delay': 0},
            },
        ],
    )
    scheduler.arm()

    def evaluate(seq):
        return scheduler.evaluate(
            seq=seq,
            connection=1,
            direction=faults.REQUEST,
        ).action

    assert evaluate(1) is None
    await scheduler.asleep(0.1)
    assert evaluate(2) is not None
    await scheduler.asleep(0.2)
    assert evaluate(3) is None


def test_first_rule_wins():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 1, 'action': {'type': 'reset'}},
            {'seq': 1, 'action': {'type': 'delay', 'delay': 1}},
        ],
    )
    decision = scheduler.evaluate(
        seq=1,
        connection=1,
        direction=faults.REQUEST,
    )
    assert decision.action['type'] == 'reset'
    assert decision.rule_id == 'rule0'


def test_short_write_resolution():
    scheduler = faults.FaultScheduler(
        seed=5,
        rules=[
            {'action': {'type': 'short_write', 'bytes': 4}},
        ],
    )
    decision = scheduler.evaluate(
        seq=1,
        connection=1,
        direction=faults.SEND,
    )
    assert 'bytes' not in decision.action
    action = scheduler.resolve_action(decision, data_len=100)
    assert action['bytes'] == 4
    # Resolution is idempotent and rng-free.
    assert scheduler.resolve_action(decision, data_len=100)['bytes'] == 4


def test_half_packet_fraction():
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'action': {
                    'type': 'half_packet',
                    'bytes': {'fraction': 0.3},
                    'delay': 0,
                },
            },
        ],
    )
    decision = scheduler.evaluate(
        seq=1,
        connection=1,
        direction=faults.SEND,
    )
    action = scheduler.resolve_action(decision, data_len=10)
    assert action['bytes'] == 3


def test_error_response_config():
    scheduler = faults.FaultScheduler(
        rules=[
            {
                'action': {
                    'type': 'error_response',
                    'status': 503,
                    'body': 'oops',
                    'headers': {'X-Foo': 'bar'},
                },
            },
        ],
    )
    decision = scheduler.evaluate(
        seq=1,
        connection=1,
        direction=faults.REQUEST,
    )
    assert decision.action['status'] == 503
    assert decision.action['body'] == 'oops'
    assert decision.action['headers'] == {'X-Foo': 'bar'}
    assert decision.action['content_type'] == 'text/plain'


def test_export_import_roundtrip():
    scheduler = faults.FaultScheduler(
        seed=7,
        rules=[
            {
                'seq': 1,
                'action': {
                    'type': 'delay',
                    'delay': {'min': 0.1, 'max': 0.5},
                },
            },
        ],
    )
    scheduler.evaluate(seq=1, connection=1, direction=faults.REQUEST)
    scheduler.evaluate(seq=2, connection=1, direction=faults.REQUEST)

    data = json.loads(json.dumps(scheduler.export_log()))
    replay = faults.FaultScheduler.from_log(data)
    assert replay.replay_mode

    first = replay.evaluate(seq=1, connection=1, direction=faults.REQUEST)
    second = replay.evaluate(seq=2, connection=1, direction=faults.REQUEST)
    assert first.action == scheduler.decisions[0].action
    assert second.action is None


def test_replay_mismatch():
    scheduler = faults.FaultScheduler(
        rules=[
            {'seq': 1, 'action': {'type': 'reset'}},
        ],
    )
    scheduler.evaluate(seq=1, connection=1, direction=faults.REQUEST)

    replay = faults.FaultScheduler.from_log(scheduler.export_log())
    with pytest.raises(faults.FaultReplayMismatch):
        replay.evaluate(seq=1, connection=2, direction=faults.REQUEST)

    replay = faults.FaultScheduler.from_log(scheduler.export_log())
    replay.evaluate(seq=1, connection=1, direction=faults.REQUEST)
    with pytest.raises(faults.FaultReplayMismatch):
        replay.evaluate(seq=2, connection=1, direction=faults.REQUEST)


def test_replay_keeps_concrete_bytes():
    scheduler = faults.FaultScheduler(
        seed=9,
        rules=[
            {
                'action': {
                    'type': 'short_write',
                    'bytes': {'min': 1, 'max': 50},
                },
            },
        ],
    )
    decision = scheduler.evaluate(
        seq=1,
        connection=1,
        direction=faults.SEND,
    )
    scheduler.resolve_action(decision, data_len=100)

    replay = faults.FaultScheduler.from_log(scheduler.export_log())
    replayed = replay.evaluate(
        seq=1,
        connection=1,
        direction=faults.SEND,
    )
    assert replayed.action['bytes'] == decision.action['bytes']


@pytest.mark.parametrize(
    'rules',
    [
        [{}],
        [{'action': {'type': 'bogus'}}],
        [{'seq': 0, 'action': {'type': 'reset'}}],
        [{'action': {'type': 'delay', 'delay': -1}}],
        [{'action': {'type': 'short_write'}}],
        [
            {
                'direction': 'sideways',
                'action': {'type': 'reset'},
            },
        ],
        [
            {
                'action': {
                    'type': 'error_response',
                    'status': 999,
                },
            },
        ],
    ],
)
def test_invalid_config(rules):
    with pytest.raises(faults.FaultConfigError):
        faults.FaultScheduler(rules=rules)
