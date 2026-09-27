"""Seed-based network fault scheduler.

Injects network faults (``delay``, ``short_write``, ``half_packet``,
``reset``, ``timeout`` and configurable ``error_response``) matched by
request sequence number, connection, direction or time window.

Every scheduler evaluation, including fault-free ones, is recorded as a
*decision*. The decision log can be exported to JSON and later imported
for exact replay of a failing test. When the scheduler is not configured
(via the ``faults`` pytest fixture) no faults are injected at all.

The module is framework agnostic; transport-specific glue lives in
:mod:`testsuite.mockserver.server` and :mod:`testsuite.asyncio_socket`.
"""

import asyncio
import dataclasses
import json
import logging
import numbers
import pathlib
import random
import time
import typing

LOG_VERSION = 1

# Event directions.
REQUEST = 'request'
RESPONSE = 'response'
SEND = 'send'
RECV = 'recv'
DIRECTIONS = (REQUEST, RESPONSE, SEND, RECV)
ANY = 'any'

# Fault types.
DELAY = 'delay'
SHORT_WRITE = 'short_write'
HALF_PACKET = 'half_packet'
RESET = 'reset'
TIMEOUT = 'timeout'
ERROR_RESPONSE = 'error_response'
FAULT_TYPES = (
    DELAY,
    SHORT_WRITE,
    HALF_PACKET,
    RESET,
    TIMEOUT,
    ERROR_RESPONSE,
)

# Fault types applicable at the http mockserver layer.
HTTP_FAULT_TYPES = frozenset((DELAY, RESET, TIMEOUT, ERROR_RESPONSE))

logger = logging.getLogger(__name__)


class FaultError(Exception):
    """Base class for fault scheduler errors."""


class FaultConfigError(FaultError):
    """Invalid fault scheduler configuration."""


class FaultReplayMismatch(FaultError):
    """Replayed event stream diverges from the recorded decision log."""


class FaultConnectionReset(ConnectionResetError):
    """Raised after the transport was forcibly reset by the scheduler."""


@dataclasses.dataclass
class Decision:
    """A single scheduler decision, recorded for replay."""

    index: int
    seq: int
    connection: int
    direction: str
    rule_id: str | None = None
    action: dict | None = None
    elapsed: float = 0.0
    note: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Decision':
        return cls(
            index=data['index'],
            seq=data['seq'],
            connection=data['connection'],
            direction=data['direction'],
            rule_id=data.get('rule_id'),
            action=data.get('action'),
            elapsed=data.get('elapsed', 0.0),
            note=data.get('note'),
        )


@dataclasses.dataclass
class FaultRule:
    action: dict
    rule_id: str = ''
    seq: typing.Any = None
    connection: typing.Any = None
    direction: typing.Any = None
    time_window: dict | None = None

    def to_dict(self) -> dict:
        return {
            'rule_id': self.rule_id,
            'seq': self.seq,
            'connection': self.connection,
            'direction': self.direction,
            'time_window': self.time_window,
            'action': self.action,
        }

    @classmethod
    def from_config(cls, data: dict) -> 'FaultRule':
        if 'action' not in data:
            raise FaultConfigError('rule is missing required key: action')
        return cls(
            action=dict(data['action']),
            rule_id=str(data.get('rule_id', '')),
            seq=data.get('seq'),
            connection=data.get('connection'),
            direction=data.get('direction'),
            time_window=data.get('time_window'),
        )


class FaultScheduler:
    """Seed-based fault scheduler with decision log export/import."""

    def __init__(self, *, seed: int | None = None, rules: list | None = None):
        self._rules: list[FaultRule] = []
        self._rng: random.Random = random.Random()
        self._seed: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._arm_time: float = 0.0
        self._connections: dict[typing.Hashable, int] = {}
        self._request_count = 0
        self._op_counts: dict[tuple, int] = {}
        self._decisions: list[Decision] = []
        self._replay: list[Decision] | None = None
        self._on_mismatch: (
            typing.Callable[[FaultReplayMismatch], None] | None
        ) = None
        if seed is not None or rules is not None:
            self.configure(seed=seed, rules=rules)

    #
    # Configuration
    #

    def configure(
        self,
        *,
        seed: int | None = None,
        rules: list | None = None,
    ) -> None:
        """Configure scheduler seed and fault rules.

        Rules are evaluated in order, first matching rule wins. Randomized
        action parameters are drawn from a deterministic rng seeded with
        ``seed``, so two schedulers with the same seed make identical
        decisions.
        """
        if seed is not None:
            if not isinstance(seed, numbers.Integral):
                raise FaultConfigError('seed must be an integer')
            self._seed = int(seed)
        self._rng = random.Random(self._seed)
        if rules is None:
            rules = []
        self._rules = [
            self._build_rule(raw, position)
            for position, raw in enumerate(rules)
        ]
        if self._replay is not None:
            raise FaultConfigError(
                'cannot reconfigure a scheduler running in replay mode',
            )

    @property
    def configured(self) -> bool:
        return bool(self._rules)

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def rules(self) -> list[FaultRule]:
        return list(self._rules)

    @property
    def decisions(self) -> list[Decision]:
        return list(self._decisions)

    @property
    def replay_mode(self) -> bool:
        return self._replay is not None

    def set_mismatch_callback(
        self,
        callback: typing.Callable[[FaultReplayMismatch], None],
    ) -> None:
        """Register callback invoked before a replay mismatch is raised."""
        self._on_mismatch = callback

    def check_replay_exhausted(self) -> None:
        """Report recorded decisions that were never replayed."""
        if not self._replay:
            return
        exc = FaultReplayMismatch(
            f'{len(self._replay)} recorded decision(s) were not replayed, '
            'the test ended earlier',
        )
        if self._on_mismatch is not None:
            self._on_mismatch(exc)
        else:
            raise exc

    #
    # Loop binding (asyncio integration, used by AsyncioSocket auto discovery)
    #

    def arm(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Bind scheduler to an event loop and start the time window clock."""
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._arm_time = time.monotonic()
        setattr(loop, _LOOP_ATTR, self)

    def disarm(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Unbind scheduler from the event loop."""
        loop = loop or self._loop
        self._loop = None
        if loop is not None and getattr(loop, _LOOP_ATTR, None) is self:
            try:
                delattr(loop, _LOOP_ATTR)
            except AttributeError:
                pass

    @property
    def armed(self) -> bool:
        return self._loop is not None

    #
    # Counters and matching
    #

    def on_connection(self, key: typing.Hashable) -> int:
        """Return stable 1-based connection ordinal for ``key``."""
        conn_id = self._connections.get(key)
        if conn_id is None:
            conn_id = len(self._connections) + 1
            self._connections[key] = conn_id
        return conn_id

    def next_request_seq(self, key: typing.Hashable | None = None) -> int:
        """Increment and return the global request sequence counter."""
        if key is not None:
            self.on_connection(key)
        self._request_count += 1
        return self._request_count

    def next_op_index(self, key: typing.Hashable, direction: str) -> int:
        """Increment and return per-connection per-direction op counter."""
        self.on_connection(key)
        count_key = (key, direction)
        value = self._op_counts.get(count_key, 0) + 1
        self._op_counts[count_key] = value
        return value

    def evaluate(
        self,
        *,
        seq: int,
        connection: int,
        direction: str,
    ) -> Decision:
        """Evaluate scheduler for an event, returns recorded decision."""
        if self._replay is not None:
            return self._evaluate_replay(
                seq=seq,
                connection=connection,
                direction=direction,
            )
        elapsed = self._elapsed()
        index = len(self._decisions) + 1
        for rule in self._rules:
            if self._rule_matches(rule, seq, connection, direction, elapsed):
                action = self._resolve_action(rule)
                decision = Decision(
                    index=index,
                    seq=seq,
                    connection=connection,
                    direction=direction,
                    rule_id=rule.rule_id or None,
                    action=action,
                    elapsed=elapsed,
                )
                self._decisions.append(decision)
                return decision
        decision = Decision(
            index=index,
            seq=seq,
            connection=connection,
            direction=direction,
            elapsed=elapsed,
        )
        self._decisions.append(decision)
        return decision

    def resolve_action(self, decision: Decision, data_len: int) -> dict | None:
        """Finalize payload-dependent action (short write / half packet).

        Concrete byte counts are derived from the recorded random draw, so
        replay needs no rng. Idempotent.
        """
        action = decision.action
        if action is None:
            return None
        if 'bytes' in action:
            return action
        spec = action.get('bytes_spec')
        if spec is None:
            return action
        draw = action.get('_u', 0.0)
        action['bytes'] = _resolve_bytes(spec, draw, data_len)
        return action

    #
    # Async fault primitives
    #

    async def asleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)

    async def raise_timeout(self, timeout: float) -> typing.NoReturn:
        await self.asleep(timeout)
        raise asyncio.TimeoutError

    #
    # Decision log export/import
    #

    def export_log(self) -> dict:
        """Export scheduler state, including full decision log."""
        return {
            'version': LOG_VERSION,
            'seed': self._seed,
            'armed_at': self._arm_time,
            'rules': [rule.to_dict() for rule in self._rules],
            'decisions': [decision.to_dict() for decision in self._decisions],
        }

    def dump_log(self, path: str | pathlib.Path) -> None:
        """Write exported decision log to ``path``."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as file:
            json.dump(self.export_log(), file, indent=2, sort_keys=True)

    @classmethod
    def from_log(cls, data: dict) -> 'FaultScheduler':
        """Import decision log; returned scheduler runs in replay mode."""
        if data.get('version') != LOG_VERSION:
            raise FaultConfigError(
                f'unsupported decision log version: {data.get("version")!r}',
            )
        scheduler = cls()
        scheduler._seed = data.get('seed')
        scheduler._arm_time = data.get('armed_at', 0.0)
        scheduler._rules = [
            FaultRule.from_config(raw) for raw in data.get('rules', [])
        ]
        decisions = [
            Decision.from_dict(raw) for raw in data.get('decisions', [])
        ]
        scheduler._replay = list(decisions)
        return scheduler

    @classmethod
    def load_log(cls, path: str | pathlib.Path) -> 'FaultScheduler':
        """Load decision log from json file for replay."""
        with pathlib.Path(path).open(encoding='utf-8') as file:
            return cls.from_log(json.load(file))

    #
    # Service logs integration
    #

    def log_decisions(self) -> None:
        """Emit tskv log lines, one per fault decision that injected fault."""
        for decision in self._decisions:
            action = decision.action
            if action is None:
                continue
            fields = {
                '_type': 'fault_decision',
                'index': decision.index,
                'seq': decision.seq,
                'connection': decision.connection,
                'direction': decision.direction,
                'rule_id': decision.rule_id or '',
                'fault': action['type'],
                'action': json.dumps(action, sort_keys=True),
            }
            logger.info('Fault injected', extra={'tskv': fields})

    def report_section(self) -> str:
        """Render decision log for pytest report attachment."""
        exported = json.dumps(self.export_log(), indent=2, sort_keys=True)
        header = (
            'Fault scheduler decision log.\n'
            'Replay the exact fault sequence by saving this json and running '
            'pytest with --faults-replay <file>.json\n\n'
        )
        return header + exported

    #
    # Internal
    #

    def _elapsed(self) -> float:
        if not self._arm_time:
            return 0.0
        return time.monotonic() - self._arm_time

    def _build_rule(self, raw, position: int) -> FaultRule:
        if not isinstance(raw, dict):
            raise FaultConfigError(
                f'rule #{position} must be a dict, got {type(raw).__name__}',
            )
        rule = FaultRule.from_config(raw)
        if not rule.rule_id:
            rule.rule_id = f'rule{position}'
        self._validate_matchers(rule)
        rule.action = _validate_action(rule.action)
        return rule

    def _validate_matchers(self, rule: FaultRule) -> None:
        _validate_seq(rule.seq)
        _validate_connection(rule.connection)
        if rule.direction is not None:
            values = (
                rule.direction
                if isinstance(rule.direction, list)
                else [rule.direction]
            )
            for value in values:
                if value != ANY and value not in DIRECTIONS:
                    raise FaultConfigError(
                        f'rule {rule.rule_id!r}: invalid direction {value!r}, '
                        f'expected one of {DIRECTIONS + (ANY,)}',
                    )
        if rule.time_window is not None:
            window = rule.time_window
            if not isinstance(window, dict):
                raise FaultConfigError(
                    f'rule {rule.rule_id!r}: time_window must be a dict',
                )
            start = window.get('start', 0.0)
            end = window.get('end')
            _number(start, nonneg=True, name='time_window.start')
            if end is not None:
                _number(end, nonneg=True, name='time_window.end')
                if end < start:
                    raise FaultConfigError(
                        f'rule {rule.rule_id!r}: time_window end < start',
                    )

    def _rule_matches(self, rule, seq, connection, direction, elapsed) -> bool:
        if not _match_scalar(rule.seq, seq):
            return False
        if not _match_scalar(rule.connection, connection):
            return False
        if rule.direction is not None:
            wanted = rule.direction
            if isinstance(wanted, list):
                if direction not in wanted and ANY not in wanted:
                    return False
            elif wanted != direction and wanted != ANY:
                return False
        if rule.time_window is not None:
            window = rule.time_window
            start = window.get('start', 0.0)
            end = window.get('end')
            if elapsed < start:
                return False
            if end is not None and elapsed >= end:
                return False
        return True

    def _resolve_action(self, rule: FaultRule) -> dict:
        spec = rule.action
        fault_type = spec['type']
        action: dict = {'type': fault_type}
        if fault_type == DELAY:
            action['delay'] = _draw_number(spec['delay'], self._rng)
        elif fault_type == TIMEOUT:
            action['timeout'] = _draw_number(spec['timeout'], self._rng)
        elif fault_type == RESET:
            return action
        elif fault_type == ERROR_RESPONSE:
            action.update(
                status=spec['status'],
                body=spec['body'],
                headers=dict(spec['headers']),
                content_type=spec['content_type'],
            )
        elif fault_type in (SHORT_WRITE, HALF_PACKET):
            action['bytes_spec'] = spec['bytes']
            action['_u'] = self._rng.random()
            if fault_type == HALF_PACKET:
                action['delay'] = _draw_number(spec.get('delay', 0), self._rng)
        else:  # pragma: no cover - guarded by validation
            raise FaultConfigError(f'unsupported fault type {fault_type!r}')
        return action

    def _evaluate_replay(self, *, seq, connection, direction) -> Decision:
        if not self._replay:
            self._mismatch(
                f'unexpected event seq={seq} connection={connection} '
                f'direction={direction}: recorded log exhausted',
            )
        recorded = self._replay.pop(0)
        if (
            recorded.seq != seq
            or recorded.connection != connection
            or recorded.direction != direction
        ):
            self._mismatch(
                'event diverges from recorded decision: '
                f'got seq={seq} connection={connection} direction={direction}, '
                f'expected seq={recorded.seq} connection={recorded.connection} '
                f'direction={recorded.direction}',
            )
        recorded.index = len(self._decisions) + 1
        self._decisions.append(recorded)
        return recorded

    def _mismatch(self, message: str) -> typing.NoReturn:
        exc = FaultReplayMismatch(message)
        if self._on_mismatch is not None:
            self._on_mismatch(exc)
        raise exc


_LOOP_ATTR = '_testsuite_fault_scheduler'


def scheduler_for_loop(
    loop: asyncio.AbstractEventLoop,
) -> FaultScheduler | None:
    """Return scheduler armed on the loop, if any."""
    scheduler = getattr(loop, _LOOP_ATTR, None)
    if scheduler is not None and not isinstance(scheduler, FaultScheduler):
        return None
    return scheduler


#
# Spec validation helpers
#


def _validate_seq(value) -> None:
    if value is None:
        return
    if isinstance(value, numbers.Integral):
        if value < 1:
            raise FaultConfigError('seq must be >= 1')
        return
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, numbers.Integral) or item < 1:
                raise FaultConfigError(f'invalid seq entry {item!r}')
        return
    if isinstance(value, dict):
        for key in ('gte', 'lte'):
            if key in value:
                bound = value[key]
                if not isinstance(bound, numbers.Integral) or bound < 1:
                    raise FaultConfigError(f'invalid seq {key}={bound!r}')
        if not any(key in value for key in ('gte', 'lte', 'in')):
            raise FaultConfigError(
                f'seq range must contain one of gte/lte/in, got {value!r}',
            )
        if 'in' in value:
            _validate_seq(value['in'])
        return
    raise FaultConfigError(f'invalid seq matcher {value!r}')


def _validate_connection(value) -> None:
    if value is None:
        return
    if isinstance(value, numbers.Integral):
        if value < 1:
            raise FaultConfigError('connection must be >= 1')
        return
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, numbers.Integral) or item < 1:
                raise FaultConfigError(f'invalid connection entry {item!r}')
        return
    raise FaultConfigError(f'invalid connection matcher {value!r}')


def _validate_bytes_spec(value, *, name: str) -> typing.Any:
    if isinstance(value, numbers.Integral):
        if value < 1:
            raise FaultConfigError(f'{name} must be >= 1')
        return int(value)
    if isinstance(value, dict):
        if 'fraction' in value:
            fraction = value['fraction']
            _number(fraction, name=f'{name}.fraction')
            if not 0 < fraction <= 1:
                raise FaultConfigError(f'{name}.fraction must be in (0, 1]')
            return {'fraction': float(fraction)}
        if 'min' in value or 'max' in value:
            low = value.get('min', 1)
            high = value.get('max')
            if not isinstance(low, numbers.Integral) or low < 1:
                raise FaultConfigError(f'{name}.min must be integer >= 1')
            if high is not None:
                if not isinstance(high, numbers.Integral) or high < low:
                    raise FaultConfigError(
                        f'{name}.max must be integer >= {name}.min',
                    )
            return {'min': int(low), 'max': int(high) if high else None}
    raise FaultConfigError(
        f'{name} must be integer, {{"fraction": f}} or {{"min", "max"}}',
    )


def _validate_duration_spec(value, *, name: str) -> typing.Any:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        _number(float(value), nonneg=True, name=name)
        return float(value)
    if isinstance(value, dict):
        low = value.get('min', 0.0)
        high = value.get('max')
        _number(low, nonneg=True, name=f'{name}.min')
        if high is not None:
            _number(high, nonneg=True, name=f'{name}.max')
            if high < low:
                raise FaultConfigError(f'{name}.max < {name}.min')
        return {'min': float(low), 'max': float(high) if high else None}
    raise FaultConfigError(
        f'{name} must be number or {{"min", "max"}} mapping',
    )


def _validate_action(raw) -> dict:
    if not isinstance(raw, dict):
        raise FaultConfigError('action must be a dict')
    fault_type = raw.get('type')
    if fault_type not in FAULT_TYPES:
        raise FaultConfigError(
            f'action.type must be one of {FAULT_TYPES}, got {fault_type!r}',
        )
    action = {'type': fault_type}
    if fault_type == DELAY:
        action['delay'] = _validate_duration_spec(
            raw.get('delay', 0), name='delay'
        )
    elif fault_type == TIMEOUT:
        if 'timeout' not in raw:
            raise FaultConfigError('timeout action requires key: timeout')
        action['timeout'] = _validate_duration_spec(
            raw['timeout'], name='timeout'
        )
    elif fault_type == RESET:
        return action
    elif fault_type == SHORT_WRITE:
        if 'bytes' not in raw:
            raise FaultConfigError('short_write action requires key: bytes')
        action['bytes'] = _validate_bytes_spec(raw['bytes'], name='bytes')
    elif fault_type == HALF_PACKET:
        if 'bytes' not in raw:
            raise FaultConfigError('half_packet action requires key: bytes')
        action['bytes'] = _validate_bytes_spec(raw['bytes'], name='bytes')
        action['delay'] = _validate_duration_spec(
            raw.get('delay', 0), name='delay'
        )
    elif fault_type == ERROR_RESPONSE:
        status = raw.get('status', 500)
        status_int = int(status) if isinstance(status, numbers.Integral) else 0
        if (
            not isinstance(status, numbers.Integral)
            or not 100 <= status_int <= 599
        ):
            raise FaultConfigError('error_response.status must be 100..599')
        body = raw.get('body', '')
        if not isinstance(body, str):
            raise FaultConfigError('error_response.body must be a string')
        headers = raw.get('headers', {})
        if not isinstance(headers, dict):
            raise FaultConfigError('error_response.headers must be a dict')
        content_type = raw.get('content_type', 'text/plain')
        if not isinstance(content_type, str):
            raise FaultConfigError('error_response.content_type must be string')
        action.update(
            status=status_int,
            body=body,
            headers=dict(headers),
            content_type=content_type,
        )
    return action


def _number(
    value,
    *,
    nonneg: bool = False,
    positive: bool = False,
    name: str = 'value',
) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise FaultConfigError(f'{name} must be a number, got {value!r}')
    if nonneg and value < 0:
        raise FaultConfigError(f'{name} must be >= 0')
    if positive and value <= 0:
        raise FaultConfigError(f'{name} must be > 0')


def _draw_number(spec, rng: random.Random) -> float:
    if isinstance(spec, numbers.Real):
        return float(spec)
    low = spec['min']
    high = spec.get('max')
    if high is None:
        return float(low)
    return rng.uniform(low, high)


def _resolve_bytes(spec, draw: float, data_len: int) -> int:
    if isinstance(spec, numbers.Integral):
        count = int(spec)
    elif 'fraction' in spec:
        count = int(round(spec['fraction'] * data_len))
    else:
        low = spec['min']
        high = spec.get('max')
        if high is None:
            count = low
        else:
            count = low + int(draw * (high - low + 1))
    if data_len > 0:
        count = min(count, data_len)
    return max(1, count)


def _match_scalar(matcher, value) -> bool:
    if matcher is None:
        return True
    if isinstance(matcher, numbers.Integral):
        return matcher == value
    if isinstance(matcher, list):
        return value in matcher
    if isinstance(matcher, dict):
        if 'in' in matcher and value not in matcher['in']:
            return False
        if 'gte' in matcher and value < matcher['gte']:
            return False
        if 'lte' in matcher and value > matcher['lte']:
            return False
        return True
    return False
