"""Stateful mockserver scenarios.

A scenario is a per-test state machine: requests matching a state step
(by method, path, headers, query or json body) are answered from a response
template. Steps are ordered and may carry a hit count, a delay, a connection
close/abort action and a transition to another state. Client states are
isolated: each client gets its own state machine run. Requests not claimed by
any scenario fall through to the regular mockserver handler dispatch.

Scenario documents are validated on load (see :func:`validate_scenario`),
are kept for the mockserver session only and are removed automatically when
the test finishes.
"""

import base64
import dataclasses
import json
import re
import typing
import uuid

from testsuite.utils import matching

from . import exceptions

# Cap scenario delays so that a typo (e.g. delay in milliseconds) can not
# hang the mockserver indefinitely.
DEFAULT_MAX_DELAY = 30.0

DEFAULT_CLIENT_ID_HEADER = 'X-Testsuite-Scenario-Client'

DEFAULT_INITIAL_STATE = 'initial'

_CLIENT_ID_FROM = ('header', 'peer', 'header-or-peer', 'shared')

_ALLOWED_TOP_KEYS = {
    'id',
    'initial_state',
    'states',
    'client_id_from',
    'client_id_header',
}
_ALLOWED_STEP_KEYS = {
    'match',
    'times',
    'delay',
    'delay_ms',
    'response',
    'close_connection',
    'next_state',
}
_ALLOWED_MATCH_KEYS = {
    'method',
    'path',
    'path_prefix',
    'path_regex',
    'headers',
    'query',
    'json',
}
_ALLOWED_RESPONSE_KEYS = {
    'status',
    'json',
    'text',
    'body',
    'headers',
    'content_type',
}

_MISSING = object()


class ScenarioError(exceptions.MockServerError):
    """Base class for scenario runtime errors."""


class ScenarioSpecError(ScenarioError):
    """Invalid scenario document."""


@dataclasses.dataclass(frozen=True)
class ResponseSpec:
    status: int = 200
    json: object = _MISSING
    text: str | None = None
    body: bytes | None = None
    headers: typing.Mapping[str, str] | None = None
    content_type: str | None = None


@dataclasses.dataclass(frozen=True)
class RequestMatcher:
    methods: frozenset[str]
    path: str | None
    path_prefix: str | None
    path_regex: typing.Pattern[str] | None
    headers: typing.Mapping[str, tuple[str, object]]
    query: typing.Mapping[str, tuple[str, object]]
    json_matcher: object

    async def matches(
        self,
        request,
    ) -> tuple[dict[str, str], bytes] | None:
        """Returns match variables (path regex groups) and consumed body."""
        if self.methods and request.method not in self.methods:
            return None

        path = request.original_path
        variables: dict[str, str] = {}
        if self.path is not None:
            if path != self.path:
                return None
        elif self.path_prefix is not None:
            if not path.startswith(self.path_prefix):
                return None
        elif self.path_regex is not None:
            match = self.path_regex.fullmatch(path)
            if match is None:
                return None
            variables = {
                key: value
                for key, value in match.groupdict().items()
                if value is not None
            }

        for name, expected in self.headers.items():
            if name not in request.headers:
                return None
            if not _value_matches(expected, request.headers[name]):
                return None

        for name, expected in self.query.items():
            if name not in request.query:
                return None
            if not _value_matches(expected, request.query[name]):
                return None

        body = b''
        if self.json_matcher is not None:
            body = await request.content.read()
            try:
                body_json = json.loads(body)
            except (ValueError, TypeError):
                return None
            if not (self.json_matcher == body_json):
                return None

        return variables, body


@dataclasses.dataclass(frozen=True)
class StepSpec:
    match: RequestMatcher
    times: int
    delay: float
    response: ResponseSpec
    # None, 'close' (respond, then close connection) or
    # 'abort' (close transport without sending a response).
    close_connection: str | None
    next_state: str | None


@dataclasses.dataclass(frozen=True)
class ScenarioSpec:
    id: str
    initial_state: str
    states: typing.Mapping[str, tuple[StepSpec, ...]]
    client_id_from: str
    client_id_header: str


@dataclasses.dataclass(frozen=True)
class ClientStateView:
    """Snapshot of per-client scenario state, used in test assertions."""

    state: str
    step_index: int
    step_count: int
    total_calls: int


class _ClientRuntime:
    def __init__(self, initial_state: str):
        self.lock: typing.Any = None
        # Imported lazily to avoid module import cost; asyncio.Lock may only
        # be created outside running loop in modern pythons, create on demand.
        self.state = initial_state
        self.step_index = 0
        self.hit_count = 0
        self.total_calls = 0

    def get_lock(self):
        import asyncio

        if self.lock is None:
            self.lock = asyncio.Lock()
        return self.lock


@dataclasses.dataclass(frozen=True)
class Claim:
    spec: ScenarioSpec
    client_runtime: _ClientRuntime
    client_id: str
    step: StepSpec
    variables: dict[str, str]
    body: bytes
    snapshot: tuple[str, int, int, int]

    async def rollback(self) -> None:
        """Restore client state as it was before the claim."""
        runtime = self.client_runtime
        async with runtime.get_lock():
            (
                runtime.state,
                runtime.step_index,
                runtime.hit_count,
                runtime.total_calls,
            ) = self.snapshot


class ScenarioRuntime:
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self._clients: dict[str, _ClientRuntime] = {}

    async def try_claim(self, request) -> Claim | None:
        client_id = self._resolve_client_id(request)
        if client_id is None:
            return None

        created = False
        runtime = self._clients.get(client_id)
        if runtime is None:
            runtime = _ClientRuntime(self.spec.initial_state)
            created = True

        async with runtime.get_lock():
            steps = self.spec.states[runtime.state]
            index = min(runtime.step_index, len(steps) - 1)
            step = steps[index]
            try:
                result = await step.match.matches(request)
            except Exception:
                # Broken matcher (e.g. client vanished mid-read). Must not
                # leave the server in a bad shape: treat as non-match.
                # CancelledError is a BaseException and is not caught here.
                return None
            if result is None:
                return None
            variables, body = result

            if created:
                self._clients[client_id] = runtime
            snapshot = (
                runtime.state,
                runtime.step_index,
                runtime.hit_count,
                runtime.total_calls,
            )
            runtime.total_calls += 1
            runtime.hit_count += 1
            if runtime.hit_count >= step.times:
                runtime.hit_count = 0
                if step.next_state is not None:
                    runtime.state = step.next_state
                    runtime.step_index = 0
                elif index + 1 < len(steps):
                    runtime.step_index = index + 1
                # Terminal step: stay on it, keep serving (sticky).

            return Claim(
                spec=self.spec,
                client_runtime=runtime,
                client_id=client_id,
                step=step,
                variables=dict(variables),
                body=body,
                snapshot=snapshot,
            )

    def _resolve_client_id(self, request) -> str | None:
        source = self.spec.client_id_from
        if source == 'shared':
            return ''
        if source == 'peer':
            return request.remote or ''
        value = request.headers.get(self.spec.client_id_header)
        if value is not None:
            return value
        if source == 'header-or-peer':
            return request.remote or ''
        return None

    def client_ids(self) -> list[str]:
        return list(self._clients)

    def client_state(self, client_id: str) -> ClientStateView:
        runtime = self._clients[client_id]
        return ClientStateView(
            state=runtime.state,
            step_index=runtime.step_index,
            step_count=runtime.hit_count,
            total_calls=runtime.total_calls,
        )

    @property
    def times_called(self) -> int:
        return sum(runtime.total_calls for runtime in self._clients.values())


class ScenarioHandle:
    """Handle returned when a scenario is loaded.

    Use it to assert on the per-client state machine progress.
    """

    def __init__(self, runtime: ScenarioRuntime):
        self._runtime = runtime

    @property
    def id(self) -> str:
        return self._runtime.spec.id

    @property
    def times_called(self) -> int:
        return self._runtime.times_called

    @property
    def client_ids(self) -> list[str]:
        return self._runtime.client_ids()

    def client_state(self, client_id: str) -> ClientStateView:
        return self._runtime.client_state(client_id)


class ScenarioEngine:
    """Holds scenarios loaded for a single mockserver session."""

    def __init__(
        self,
        *,
        asyncexc_append,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        self._asyncexc_append = asyncexc_append
        self._max_delay = max_delay
        self._runtimes: list[ScenarioRuntime] = []
        self._ids: set[str] = set()

    @property
    def max_delay(self) -> float:
        return self._max_delay

    def add(self, spec: ScenarioSpec) -> ScenarioHandle:
        if spec.id in self._ids:
            raise ScenarioSpecError(
                f'Scenario with id {spec.id!r} is already loaded',
            )
        runtime = ScenarioRuntime(spec)
        self._runtimes.append(runtime)
        self._ids.add(spec.id)
        return ScenarioHandle(runtime)

    async def try_claim(self, request) -> Claim | None:
        for runtime in self._runtimes:
            claim = await runtime.try_claim(request)
            if claim is not None:
                return claim
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_scenario(
    doc: object,
    *,
    max_delay: float = DEFAULT_MAX_DELAY,
    default_client_header: str = DEFAULT_CLIENT_ID_HEADER,
) -> ScenarioSpec:
    """Validate scenario document and build a compiled :class:`ScenarioSpec`.

    :raises ScenarioSpecError: on any schema violation.
    """
    ctx = 'scenario'
    if not isinstance(doc, dict):
        raise ScenarioSpecError(
            f'{ctx}: mapping expected, {type(doc).__name__} given',
        )

    unknown = set(doc) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ScenarioSpecError(f'{ctx}: unknown keys {sorted(unknown)}')

    scenario_id = doc.get('id')
    if scenario_id is None:
        scenario_id = f'scenario-{uuid.uuid4().hex[:8]}'
    elif not _is_nonempty_str(scenario_id):
        raise ScenarioSpecError(f'{ctx}: id must be a non-empty string')
    ctx = f'scenario {scenario_id!r}'

    client_id_from = doc.get('client_id_from', 'header-or-peer')
    if client_id_from not in _CLIENT_ID_FROM:
        raise ScenarioSpecError(
            f'{ctx}: client_id_from must be one of {list(_CLIENT_ID_FROM)}',
        )

    client_id_header = doc.get('client_id_header', default_client_header)
    if not _is_nonempty_str(client_id_header):
        raise ScenarioSpecError(
            f'{ctx}: client_id_header must be a non-empty string',
        )

    states_doc = doc.get('states')
    if not isinstance(states_doc, dict) or not states_doc:
        raise ScenarioSpecError(
            f'{ctx}: states must be a non-empty mapping',
        )

    states: dict[str, tuple[StepSpec, ...]] = {}
    for state_name, steps_doc in states_doc.items():
        if not _is_nonempty_str(state_name):
            raise ScenarioSpecError(
                f'{ctx}: state names must be non-empty strings'
            )
        if not isinstance(steps_doc, list) or not steps_doc:
            raise ScenarioSpecError(
                f'{ctx} state {state_name!r}: must be a non-empty list of steps',
            )
        steps = []
        for index, step_doc in enumerate(steps_doc):
            step_ctx = f'{ctx} state {state_name!r} step {index}'
            steps.append(
                _validate_step(step_doc, step_ctx, max_delay),
            )
        states[state_name] = tuple(steps)

    initial_state = doc.get('initial_state', DEFAULT_INITIAL_STATE)
    if not _is_nonempty_str(initial_state):
        raise ScenarioSpecError(
            f'{ctx}: initial_state must be a non-empty string',
        )
    if initial_state not in states:
        raise ScenarioSpecError(
            f'{ctx}: initial_state {initial_state!r} is not declared in states',
        )

    for state_name, state_steps in states.items():
        for index, step in enumerate(state_steps):
            next_state = step.next_state
            if next_state is not None and next_state not in states:
                raise ScenarioSpecError(
                    f'{ctx} state {state_name!r} step {index}: '
                    f'next_state {next_state!r} is not declared in states',
                )

    return ScenarioSpec(
        id=scenario_id,
        initial_state=initial_state,
        states=states,
        client_id_from=client_id_from,
        client_id_header=client_id_header,
    )


def _validate_step(doc, ctx: str, max_delay: float) -> StepSpec:
    if not isinstance(doc, dict):
        raise ScenarioSpecError(
            f'{ctx}: mapping expected, {type(doc).__name__} given',
        )
    unknown = set(doc) - _ALLOWED_STEP_KEYS
    if unknown:
        raise ScenarioSpecError(f'{ctx}: unknown keys {sorted(unknown)}')

    match = _validate_match(doc.get('match'), f'{ctx} match')
    times = doc.get('times', 1)
    if _is_bool(times) or not isinstance(times, int) or times < 1:
        raise ScenarioSpecError(f'{ctx}: times must be a positive integer')

    delay = 0.0
    if 'delay' in doc and 'delay_ms' in doc:
        raise ScenarioSpecError(
            f'{ctx}: delay and delay_ms can not be used together',
        )
    if 'delay' in doc:
        delay = _validate_delay(doc['delay'], f'{ctx} delay')
    if 'delay_ms' in doc:
        delay = _validate_delay(doc['delay_ms'], f'{ctx} delay_ms') / 1000.0
    if delay > max_delay:
        raise ScenarioSpecError(
            f'{ctx}: delay {delay:.3f}s exceeds max allowed delay '
            f'{max_delay:.3f}s',
        )

    response = _validate_response(
        doc.get('response', {}),
        f'{ctx} response',
    )

    close_connection = doc.get('close_connection', False)
    if _is_bool(close_connection):
        close_connection = 'close' if close_connection else None
    elif close_connection not in ('close', 'abort'):
        raise ScenarioSpecError(
            f"{ctx}: close_connection must be true, false, 'close' or 'abort'",
        )

    next_state = doc.get('next_state')
    if next_state is not None and not _is_nonempty_str(next_state):
        raise ScenarioSpecError(f'{ctx}: next_state must be a non-empty string')

    return StepSpec(
        match=match,
        times=times,
        delay=delay,
        response=response,
        close_connection=close_connection,
        next_state=next_state,
    )


def _validate_match(doc, ctx: str) -> RequestMatcher:
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise ScenarioSpecError(
            f'{ctx}: mapping expected, {type(doc).__name__} given',
        )
    unknown = set(doc) - _ALLOWED_MATCH_KEYS
    if unknown:
        raise ScenarioSpecError(f'{ctx}: unknown keys {sorted(unknown)}')

    path_fields = [
        name for name in ('path', 'path_prefix', 'path_regex') if name in doc
    ]
    if len(path_fields) > 1:
        raise ScenarioSpecError(
            f'{ctx}: {path_fields} are mutually exclusive',
        )

    path = doc.get('path')
    path_prefix = doc.get('path_prefix')
    path_regex = doc.get('path_regex')
    for value, name in (
        (path, 'path'),
        (path_prefix, 'path_prefix'),
        (path_regex, 'path_regex'),
    ):
        if value is not None and not _is_nonempty_str(value):
            raise ScenarioSpecError(f'{ctx}: {name} must be a non-empty string')
    compiled_regex = None
    if path_regex is not None:
        try:
            compiled_regex = re.compile(path_regex)
        except re.error as exc:
            raise ScenarioSpecError(
                f'{ctx}: invalid path_regex: {exc}',
            ) from exc

    methods: frozenset[str] = frozenset()
    if 'method' in doc:
        method = doc['method']
        if _is_nonempty_str(method):
            methods = frozenset({method.upper()})
        elif (
            isinstance(method, list)
            and method
            and all(_is_nonempty_str(item) for item in method)
        ):
            methods = frozenset(item.upper() for item in method)
        else:
            raise ScenarioSpecError(
                f'{ctx}: method must be a non-empty string or a list '
                'of non-empty strings',
            )

    headers = _validate_value_mapping(
        doc.get('headers'),
        f'{ctx} headers',
    )
    query = _validate_value_mapping(
        doc.get('query'),
        f'{ctx} query',
    )
    json_matcher = None
    if 'json' in doc:
        json_matcher = _compile_json_matcher(doc['json'], f'{ctx} json')

    return RequestMatcher(
        methods=methods,
        path=path,
        path_prefix=path_prefix,
        path_regex=compiled_regex,
        headers=headers,
        query=query,
        json_matcher=json_matcher,
    )


def _validate_response(doc, ctx: str) -> ResponseSpec:
    if not isinstance(doc, dict):
        raise ScenarioSpecError(
            f'{ctx}: mapping expected, {type(doc).__name__} given',
        )
    unknown = set(doc) - _ALLOWED_RESPONSE_KEYS
    if unknown:
        raise ScenarioSpecError(f'{ctx}: unknown keys {sorted(unknown)}')

    status = doc.get('status', 200)
    if (
        _is_bool(status)
        or not isinstance(status, int)
        or not (100 <= status <= 599)
    ):
        raise ScenarioSpecError(
            f'{ctx}: status must be an integer in range [100, 599]',
        )

    body_fields = [name for name in ('json', 'text', 'body') if name in doc]
    if len(body_fields) > 1:
        raise ScenarioSpecError(
            f'{ctx}: {body_fields} are mutually exclusive',
        )

    json_value = doc.get('json', _MISSING)
    text_value = doc.get('text')
    body_value = doc.get('body')
    if text_value is not None and not isinstance(text_value, str):
        raise ScenarioSpecError(f'{ctx}: text must be a string')
    if body_value is not None:
        if not isinstance(body_value, str):
            raise ScenarioSpecError(
                f'{ctx}: body must be a base64-encoded string',
            )
        try:
            decoded = base64.b64decode(body_value, validate=True)
        except ValueError as exc:
            raise ScenarioSpecError(
                f'{ctx}: body is not valid base64: {exc}',
            ) from exc
        body_value = decoded

    headers = doc.get('headers')
    if headers is not None:
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ScenarioSpecError(
                f'{ctx}: headers must be a mapping of strings to strings',
            )

    content_type = doc.get('content_type')
    if content_type is not None and not isinstance(content_type, str):
        raise ScenarioSpecError(f'{ctx}: content_type must be a string')

    return ResponseSpec(
        status=status,
        json=json_value,
        text=text_value,
        body=body_value,
        headers=headers,
        content_type=content_type,
    )


def _validate_value_mapping(
    doc,
    ctx: str,
) -> typing.Mapping[str, tuple[str, object]]:
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ScenarioSpecError(
            f'{ctx}: mapping expected, {type(doc).__name__} given',
        )
    result = {}
    for name, value in doc.items():
        if not isinstance(name, str):
            raise ScenarioSpecError(f'{ctx}: keys must be strings')
        result[name] = _compile_scalar_matcher(value, f'{ctx}[{name!r}]')
    return result


def _compile_scalar_matcher(doc, ctx: str) -> tuple[str, object]:
    if isinstance(doc, str):
        return ('eq', doc)
    if (
        isinstance(doc, dict)
        and set(doc) == {'$regex'}
        and isinstance(doc['$regex'], str)
    ):
        try:
            return ('re', re.compile(doc['$regex']))
        except re.error as exc:
            raise ScenarioSpecError(f'{ctx}: invalid regex: {exc}') from exc
    raise ScenarioSpecError(
        f'{ctx}: value must be a string or a mapping {{"$regex": ...}}',
    )


def _compile_json_matcher(doc, ctx: str):
    if isinstance(doc, dict):
        keys = set(doc)
        if keys == {'$partial'}:
            return matching.recursive_partial_dict(
                _compile_json_matcher(doc['$partial'], f'{ctx}.$partial'),
            )
        if keys == {'$regex'} and isinstance(doc['$regex'], str):
            try:
                return matching.RegexString(doc['$regex'])
            except re.error as exc:
                raise ScenarioSpecError(f'{ctx}: invalid regex: {exc}') from exc
        return {
            key: _compile_json_matcher(value, f'{ctx}[{key!r}]')
            for key, value in doc.items()
        }
    if isinstance(doc, list):
        return [
            _compile_json_matcher(value, f'{ctx}[{index}]')
            for index, value in enumerate(doc)
        ]
    if doc is None or isinstance(doc, (str, int, float, bool)):
        return doc
    raise ScenarioSpecError(
        f'{ctx}: unsupported value of type {type(doc).__name__}',
    )


def _value_matches(expected: tuple[str, object], value: str) -> bool:
    kind, matcher = expected
    if kind == 'eq':
        return value == matcher
    assert isinstance(matcher, typing.Pattern)
    return matcher.search(value) is not None


def _validate_delay(value, ctx: str) -> float:
    if _is_bool(value) or not isinstance(value, (int, float)) or value < 0:
        raise ScenarioSpecError(
            f'{ctx}: must be a non-negative number',
        )
    return float(value)


def _is_nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value)


def _is_bool(value) -> bool:
    return isinstance(value, bool)


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r'\{\{\s*([A-Za-z_][\w.\-]*(?:\[\d+\])*)\s*\}\}',
)
_PATH_SEGMENT_RE = re.compile(r'^([\w\-]+)((?:\[\d+\])*)$')


def build_render_context(claim: Claim, body: bytes, request) -> dict:
    context: dict = dict(claim.variables)
    context['method'] = request.method
    context['path'] = request.original_path
    context['query'] = request.query
    context['header'] = request.headers
    if body:
        try:
            context['json'] = json.loads(body)
        except (ValueError, TypeError):
            pass
    return context


def render_string(text: str, context: typing.Mapping) -> str:
    def replace(match: re.Match) -> str:
        return str(_resolve(match.group(1), context))

    return _PLACEHOLDER_RE.sub(replace, text)


def _resolve(expr: str, context) -> object:
    current: object = context
    for segment in expr.split('.'):
        match = _PATH_SEGMENT_RE.fullmatch(segment)
        if match is None:
            raise ScenarioError(
                f'invalid template variable segment {segment!r}',
            )
        name, indices = match.groups()
        if isinstance(current, typing.Mapping):
            if name not in current:
                raise ScenarioError(
                    f'template variable {expr!r} can not be resolved: '
                    f'{name!r} is missing',
                )
            current = current[name]
        else:
            raise ScenarioError(
                f'template variable {expr!r} can not be resolved at {name!r}',
            )
        for index in re.findall(r'\[(\d+)\]', indices):
            try:
                current = current[int(index)]  # type: ignore[index]
            except (IndexError, KeyError, TypeError) as exc:
                raise ScenarioError(
                    f'template variable {expr!r} can not be resolved: {exc}',
                ) from exc
    return current
