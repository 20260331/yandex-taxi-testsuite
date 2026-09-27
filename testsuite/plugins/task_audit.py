"""Optional async resource leak audit.

When enabled (off by default) the plugin instruments event loops used by the
test session and records, at test and fixture boundaries:

* asyncio tasks (``loop.create_task`` / ``asyncio.ensure_future``);
* threaded callbacks scheduled via ``loop.run_in_executor``;
* socket watchers registered via ``loop.add_reader`` / ``loop.add_writer``.

At node and session boundaries the audit waits for a configurable graceful
window and reports:

* resources that are still running when the window expires;
* tasks whose exception was never retrieved;
* tasks that did not finish after cancellation was requested;
* resources bound to an event loop other than the audit's home loop.

Depending on the policy leaks are either escalated to test failures or
reported as warnings. The audit is compatible with pytest-asyncio old and
new modes, plain synchronous tests and user-managed event loops.
"""

import asyncio
import contextlib
import contextvars
import enum
import itertools
import logging
import pathlib
import time
import traceback as traceback_mod
import weakref
from typing import Protocol

import pytest

logger = logging.getLogger(__name__)


DEFAULT_GRACE_TIMEOUT = 5.0
DEFAULT_CANCEL_TIMEOUT = 1.0
STACK_LIMIT = 8

MARKER_IGNORE = 'task_audit_ignore'


class BaseError(Exception):
    pass


class TaskAuditError(BaseError):
    """Raised on resource leak when failure policy is in effect."""


class AuditScope(str, enum.Enum):
    NODE = 'node'
    SESSION = 'session'
    BOTH = 'both'


class Policy(str, enum.Enum):
    FAIL = 'fail'
    WARN = 'warn'


class ResourceKind(str, enum.Enum):
    TASK = 'asyncio task'
    THREADED = 'threaded callback'
    READER = 'socket reader'
    WRITER = 'socket writer'

    @property
    def is_watcher(self) -> bool:
        return self in (ResourceKind.READER, ResourceKind.WRITER)


class AuditContext:
    """Provenance information attached to newly tracked resources."""

    __slots__ = ('scope', 'label', 'nodeid')

    def __init__(self, scope: str, label: str, nodeid: str | None = None):
        self.scope = scope
        self.label = label
        self.nodeid = nodeid


_current_context: contextvars.ContextVar[AuditContext | None] = (
    contextvars.ContextVar('testsuite_task_audit_context', default=None)
)

_record_counter = itertools.count()


class TrackedRecord:
    """Single tracked async resource."""

    def __init__(
        self,
        kind: ResourceKind,
        loop: asyncio.AbstractEventLoop,
        *,
        name: str | None,
        scope: str,
        label: str,
        nodeid: str | None,
        source: list[str],
        obj: object | None = None,
        fd: int | None = None,
        callback: object | None = None,
    ):
        self.id = next(_record_counter)
        self.kind = kind
        self.loop_ref = _weakref(loop)
        self._name = name
        self.scope = scope
        self.label = label
        self.nodeid = nodeid
        self.source = source
        self.fd = fd
        self.created_at = time.monotonic()
        self.ignored = False
        self.superseded = False
        self._callback_repr = _safe_repr(callback)
        # Strong reference: a dead task with an unretrieved exception would
        # otherwise be garbage collected before the boundary check runs.
        # Records are released at node/session boundary.
        self.obj = obj

    @property
    def name(self) -> str:
        # asyncio.create_task() calls loop.create_task() before set_name(),
        # so resolve the task name lazily.
        if self._name is not None:
            return self._name
        obj = self.obj
        if isinstance(obj, asyncio.Task):
            with contextlib.suppress(Exception):
                return obj.get_name()
        return f'<{self.kind.value} #{self.id}>'

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def loop(self):
        return self.loop_ref()

    @property
    def age(self) -> float:
        return time.monotonic() - self.created_at

    def is_live(self) -> bool:
        if self.kind.is_watcher:
            return not self.superseded
        obj = self.obj
        return isinstance(obj, asyncio.Future) and not obj.done()

    def is_cross_loop(self, home_loop) -> bool:
        loop = self.loop
        return (
            loop is not None and home_loop is not None and loop is not home_loop
        )

    def unretrieved_exception(self):
        """Return the task exception, if it was never retrieved."""
        obj = self.obj
        if not isinstance(obj, asyncio.Task) or not obj.done():
            return None
        if obj.cancelled():
            return None
        # ``_log_traceback`` is reset as soon as the exception is consumed
        # (await / result / exception), on both C and Python task impls.
        if not getattr(obj, '_log_traceback', False):
            return None
        try:
            return obj.exception()
        except (asyncio.InvalidStateError, asyncio.CancelledError):
            return None

    def describe(self) -> str:
        age = f'{self.age:.1f}s'
        if self.kind.is_watcher:
            return (
                f'{self.kind.value} for fd {self.fd}'
                f' ({self._callback_repr}), created {self.label}, age {age}'
            )
        return (
            f'{self.kind.value} {self.name!r}, created {self.label}, age {age}'
        )


class _StrongRef:
    """Fallback for objects that do not support weak references."""

    def __init__(self, obj):
        self._obj = obj

    def __call__(self):
        return self._obj


def _weakref(obj):
    try:
        return weakref.ref(obj)
    except TypeError:
        return _StrongRef(obj)


def _safe_repr(obj) -> str:
    try:
        return repr(obj)
    except Exception:  # pragma: no cover - defensive
        return f'<{type(obj).__name__}>'


def _is_user_frame(filename: str, roots: tuple[str, ...]) -> bool:
    if filename.startswith('<'):
        return False
    return any(filename.startswith(root) for root in roots)


def capture_source(roots: tuple[str, ...]) -> list[str]:
    stack = traceback_mod.extract_stack()[:-1]
    user_frames = [
        entry
        for entry in stack
        if _is_user_frame(entry.filename, roots)
        and not entry.filename.endswith('plugins/task_audit.py')
    ]
    if not user_frames:
        user_frames = [
            entry
            for entry in stack
            if not entry.filename.endswith('plugins/task_audit.py')
        ][-1:]
    chosen = user_frames[-STACK_LIMIT:]
    lines = []
    for entry in chosen:
        lines.append(
            f'  File "{entry.filename}", line {entry.lineno}, in {entry.name}'
        )
        if entry.line:
            lines.append(f'    {entry.line}')
    return lines


class _WrappedMethod:
    """Replacement for a loop method, preserving its signature."""

    def __init__(self, original, handler):
        self._original = original
        self._handler = handler

    def __call__(self, *args, **kwargs):
        return self._handler(self._original, *args, **kwargs)


class AuditorLike(Protocol):
    """Minimal auditor interface used by loop instrumentation."""

    suppressed: bool

    def current_context(self) -> AuditContext: ...

    def capture_source(self) -> list[str]: ...


class LoopAudit:
    """Instrumentation state for a single event loop."""

    def __init__(self, auditor: AuditorLike, loop):
        self._auditor = auditor
        self._loop = loop
        self._records: set[TrackedRecord] = set()
        self._watchers: dict[tuple[ResourceKind, int], TrackedRecord] = {}
        self._originals: dict[str, object] = {}
        self._installed = False

    @property
    def loop(self):
        return self._loop

    @property
    def records(self) -> set[TrackedRecord]:
        return self._records

    @records.setter
    def records(self, value: set[TrackedRecord]) -> None:
        self._records = value

    def install(self) -> bool:
        if self._installed:
            return True
        try:
            self._patch('create_task', self._wrap_create_task)
            self._patch('run_in_executor', self._wrap_run_in_executor)
            if hasattr(self._loop, 'add_reader'):
                self._patch('add_reader', self._wrap_add_reader)
                self._patch('remove_reader', self._wrap_remove_reader)
            if hasattr(self._loop, 'add_writer'):
                self._patch('add_writer', self._wrap_add_writer)
                self._patch('remove_writer', self._wrap_remove_writer)
        except (AttributeError, TypeError):
            # Loops that do not allow per-instance patching (custom C-backed
            # loop implementations) cannot be instrumented.
            self.restore()
            logger.warning(
                'testsuite task audit: failed to instrument event loop %r',
                self._loop,
            )
            return False
        self._installed = True
        return True

    def restore(self) -> None:
        for attr_name, original in self._originals.items():
            with contextlib.suppress(AttributeError, TypeError):
                if original is None:
                    delattr(self._loop, attr_name)
                else:
                    setattr(self._loop, attr_name, original)
        self._originals.clear()
        self._installed = False

    def _patch(self, attr_name, handler) -> None:
        original = getattr(self._loop, attr_name)
        self._originals[attr_name] = original
        # A plain callable instance avoids function descriptor binding,
        # so the wrapped loop method keeps its original signature.
        setattr(self._loop, attr_name, _WrappedMethod(original, handler))

    def _track(self, **kwargs) -> TrackedRecord:
        record = TrackedRecord(**kwargs)
        self._records.add(record)
        return record

    def find_record(self, obj) -> TrackedRecord | None:
        for record in self._records:
            if record.obj is obj:
                return record
        return None

    def _wrap_create_task(self, original, coro, **kwargs):
        task = original(coro, **kwargs)
        if not self._auditor.suppressed:
            ctx = self._auditor.current_context()
            self._track(
                kind=ResourceKind.TASK,
                loop=self._loop,
                name=kwargs.get('name'),
                scope=ctx.scope,
                label=ctx.label,
                nodeid=ctx.nodeid,
                source=self._auditor.capture_source(),
                obj=task,
            )
        return task

    def _wrap_run_in_executor(self, original, executor, func, *args):
        future = original(executor, func, *args)
        if not self._auditor.suppressed:
            ctx = self._auditor.current_context()
            name = getattr(func, '__qualname__', None) or _safe_repr(func)
            self._track(
                kind=ResourceKind.THREADED,
                loop=self._loop,
                name=name,
                scope=ctx.scope,
                label=ctx.label,
                nodeid=ctx.nodeid,
                source=self._auditor.capture_source(),
                obj=future,
            )
        return future

    def _wrap_add_reader(self, original, fd, callback, *args):
        return self._add_watcher(
            ResourceKind.READER, original, fd, callback, args
        )

    def _wrap_add_writer(self, original, fd, callback, *args):
        return self._add_watcher(
            ResourceKind.WRITER, original, fd, callback, args
        )

    def _add_watcher(self, kind, original, fd, callback, args):
        key = (kind, fd)
        previous = self._watchers.get(key)
        if previous is not None:
            previous.superseded = True
        result = original(fd, callback, *args)
        if not self._auditor.suppressed:
            ctx = self._auditor.current_context()
            record = self._track(
                kind=kind,
                loop=self._loop,
                name=f'{kind.value} fd={fd}',
                scope=ctx.scope,
                label=ctx.label,
                nodeid=ctx.nodeid,
                source=self._auditor.capture_source(),
                fd=fd,
                callback=callback,
            )
            self._watchers[key] = record
        return result

    def _wrap_remove_reader(self, original, fd):
        return self._remove_watcher(ResourceKind.READER, original, fd)

    def _wrap_remove_writer(self, original, fd):
        return self._remove_watcher(ResourceKind.WRITER, original, fd)

    def _remove_watcher(self, kind, original, fd):
        record = self._watchers.pop((kind, fd), None)
        if record is not None:
            record.superseded = True
        return original(fd)


class LeakReport:
    """Aggregated audit findings for a single boundary."""

    def __init__(self, boundary: str):
        self.boundary = boundary
        self.still_running: list[TrackedRecord] = []
        self.unretrieved: list[tuple[TrackedRecord, BaseException]] = []
        self.cancel_incomplete: list[TrackedRecord] = []
        self.cross_loop: list[TrackedRecord] = []
        self.watchers: list[TrackedRecord] = []

    @property
    def has_leaks(self) -> bool:
        return bool(
            self.still_running
            or self.unretrieved
            or self.cancel_incomplete
            or self.cross_loop
            or self.watchers
        )

    @property
    def total(self) -> int:
        return (
            len(self.still_running)
            + len(self.unretrieved)
            + len(self.cancel_incomplete)
            + len(self.cross_loop)
            + len(self.watchers)
        )

    def _dump(self, lines, title, records):
        if not records:
            return
        lines.append(f'* {title}:')
        for record in records:
            lines.append(f'  - {record.describe()}')
            lines.extend(f'    {line}' for line in record.source)

    def render(self) -> str:
        lines = [
            f'testsuite task audit detected {self.total} leaked resource(s)'
            f' at {self.boundary} boundary:',
        ]
        self._dump(
            lines,
            'still running after graceful window'
            ' (cancellation requested by audit)',
            self.still_running,
        )
        if self.unretrieved:
            lines.append('* tasks with exception that was never retrieved:')
            for record, exc in self.unretrieved:
                lines.append(
                    f'  - {record.describe()}: {type(exc).__name__}: {exc}'
                )
                lines.extend(f'    {line}' for line in record.source)
        self._dump(
            lines,
            'still running after cancellation was requested',
            self.cancel_incomplete,
        )
        self._dump(lines, 'bound to a foreign event loop', self.cross_loop)
        self._dump(
            lines,
            'socket watchers still registered, removed by audit',
            self.watchers,
        )
        return '\n'.join(lines)


class TaskAuditor:
    """Plugin agnostic auditor: loop registry, contexts and checks."""

    def __init__(self, config):
        self._config = config
        self.grace: float = config.option.task_audit_grace
        self.cancel_timeout: float = config.option.task_audit_cancel_timeout
        self.policy = Policy(config.option.task_audit_policy)
        scope = AuditScope(config.option.task_audit_scope)
        self.check_node = scope in (AuditScope.NODE, AuditScope.BOTH)
        self.check_session = scope in (
            AuditScope.SESSION,
            AuditScope.BOTH,
        )
        self._loop_audits: dict[int, LoopAudit] = {}
        self._ignored_nodes: set[str] = set()
        self._policy_original = None
        self._current_nodeid: str | None = None
        self.suppressed = False
        self._source_roots = self._build_source_roots(config)

    @staticmethod
    def _build_source_roots(config) -> tuple[str, ...]:
        roots = [str(pathlib.Path(config.rootpath).resolve())]
        package_root = pathlib.Path(__file__).resolve().parents[1]
        roots.append(str(package_root))
        return tuple(roots)

    def capture_source(self) -> list[str]:
        return capture_source(self._source_roots)

    # -- installation -----------------------------------------------------

    def install(self) -> None:
        try:
            self._policy_original = asyncio.get_event_loop_policy()
            asyncio.set_event_loop_policy(
                _AuditPolicy(self, self._policy_original),
            )
        except Exception:  # pragma: no cover - defensive
            self._policy_original = None
            logger.exception('testsuite task audit: failed to wrap policy')

    def shutdown(self) -> None:
        if self._policy_original is not None:
            asyncio.set_event_loop_policy(self._policy_original)
            self._policy_original = None
        for loop_audit in list(self._loop_audits.values()):
            loop_audit.restore()

    def bind_loop(self, loop) -> LoopAudit:
        loop_audit = self._loop_audits.get(id(loop))
        if loop_audit is None:
            loop_audit = LoopAudit(self, loop)
            if loop_audit.install():
                self._loop_audits[id(loop)] = loop_audit
        return loop_audit

    # -- provenance -------------------------------------------------------

    def current_context(self) -> AuditContext:
        context = _current_context.get()
        if context is not None:
            return context
        if self._current_nodeid is not None:
            return AuditContext(
                'function',
                'test setup/teardown',
                self._current_nodeid,
            )
        return AuditContext('session', 'session startup')

    @staticmethod
    def push_context(context: AuditContext):
        return _current_context.set(context)

    @staticmethod
    def pop_context(token) -> None:
        _current_context.reset(token)

    def set_current_node(self, nodeid: str | None) -> None:
        self._current_nodeid = nodeid

    # -- public marking API ----------------------------------------------

    def attach(self, obj, *, name: str | None = None) -> TrackedRecord:
        """Explicitly mark an externally managed task/future."""
        loop = _object_loop(obj)
        if loop is None:
            raise TaskAuditError(
                'Cannot attach resource without an event loop binding',
            )
        loop_audit = self.bind_loop(loop)
        existing = loop_audit.find_record(obj)
        if existing is not None:
            if name is not None:
                existing.name = name
            return existing
        kind = (
            ResourceKind.TASK
            if isinstance(obj, asyncio.Task)
            else ResourceKind.THREADED
        )
        ctx = self.current_context()
        record = TrackedRecord(
            kind,
            loop,
            name=name,
            scope=ctx.scope,
            label=ctx.label,
            nodeid=ctx.nodeid,
            source=self.capture_source(),
            obj=obj,
        )
        loop_audit.records.add(record)
        return record

    # -- boundary checks --------------------------------------------------

    async def audit_node(self, request) -> LeakReport | None:
        node = request.node
        try:
            if node.get_closest_marker(MARKER_IGNORE):
                self._ignored_nodes.add(node.nodeid)
                for record in self._iter_records():
                    if record.nodeid == node.nodeid:
                        record.ignored = True
                return None
            records = [
                record
                for record in self._iter_records()
                if record.nodeid == node.nodeid and record.scope == 'function'
            ]
            report = await self._collect(f'node {node.nodeid}', records)
            self._apply_policy(request, report)
            return report
        finally:
            self._discard_node(node.nodeid)
            self.set_current_node(None)

    async def audit_session(self, request) -> LeakReport:
        records = [
            record for record in self._iter_records() if not record.ignored
        ]
        report = await self._collect('session', records)
        self._apply_policy(request, report)
        return report

    async def _collect(self, boundary, records) -> LeakReport:
        report = LeakReport(boundary)
        home_loop = asyncio.get_running_loop()
        self.bind_loop(home_loop)

        live = []
        current_task = asyncio.current_task(home_loop)
        for record in records:
            if record.ignored:
                continue
            if record.obj is current_task:
                # The audit itself runs inside a task created on the loop
                # (e.g. pytest-asyncio fixture finalizer); never audit it.
                continue
            exc = record.unretrieved_exception()
            if exc is not None:
                report.unretrieved.append((record, exc))
                continue
            if not record.is_live():
                continue
            if record.is_cross_loop(home_loop):
                report.cross_loop.append(record)
            elif record.kind.is_watcher:
                report.watchers.append(record)
            else:
                live.append(record)

        grace_start = time.monotonic()
        await self._wait_grace(home_loop, live)
        await self._classify_after_grace(live, report)

        # Spend the rest of the graceful window waiting for resources that
        # cannot be awaited from the home loop (foreign loops, watchers).
        elapsed = time.monotonic() - grace_start
        remaining = self.grace - elapsed
        if remaining > 0 and (report.cross_loop or report.watchers):
            await asyncio.sleep(remaining)
        report.cross_loop = [
            record for record in report.cross_loop if record.is_live()
        ]
        live_watchers = [
            record for record in report.watchers if record.is_live()
        ]
        report.watchers = live_watchers
        for record in report.watchers:
            self._remove_watcher(record)
        return report

    async def _wait_grace(self, home_loop, live) -> None:
        awaitables = [
            record.obj
            for record in live
            if isinstance(record.obj, asyncio.Future)
            and not record.is_cross_loop(home_loop)
        ]
        if not awaitables:
            return
        self.suppressed = True
        try:
            await asyncio.wait(awaitables, timeout=self.grace)
        finally:
            self.suppressed = False

    async def _classify_after_grace(self, live, report: LeakReport) -> None:
        pending_records = []
        for record in live:
            obj = record.obj
            exc = record.unretrieved_exception()
            if exc is not None:
                report.unretrieved.append((record, exc))
            elif obj is None or not obj.done():
                pending_records.append(record)
        if not pending_records:
            return
        for record in pending_records:
            obj = record.obj
            if obj is not None:
                with contextlib.suppress(asyncio.InvalidStateError):
                    obj.cancel()
        pending_futures = [
            record.obj
            for record in pending_records
            if isinstance(record.obj, asyncio.Future)
        ]
        if pending_futures:
            self.suppressed = True
            try:
                await asyncio.wait(pending_futures, timeout=self.cancel_timeout)
            finally:
                self.suppressed = False
        for record in pending_records:
            obj = record.obj
            if obj is not None and not obj.done():
                report.cancel_incomplete.append(record)
            else:
                report.still_running.append(record)

    def _remove_watcher(self, record: TrackedRecord) -> None:
        loop_audit = self._loop_audits.get(id(record.loop))
        if loop_audit is None or not record.kind.is_watcher:
            return
        method_name = (
            'remove_reader'
            if record.kind is ResourceKind.READER
            else 'remove_writer'
        )
        original = loop_audit._originals.get(method_name)
        fd = record.fd
        if fd is None:
            return
        record.superseded = True
        loop_audit._watchers.pop((record.kind, fd), None)
        with contextlib.suppress(OSError, ValueError):
            if callable(original):
                original(fd)

    # -- bookkeeping ------------------------------------------------------

    def _iter_records(self):
        for loop_audit in self._loop_audits.values():
            yield from list(loop_audit.records)

    def _discard_node(self, nodeid) -> None:
        for loop_audit in self._loop_audits.values():
            loop_audit.records = {
                record
                for record in loop_audit.records
                if record.nodeid != nodeid or record.scope != 'function'
            }

    def _apply_policy(self, request, report) -> None:
        if not report.has_leaks:
            return
        message = report.render()
        reporter = request.config.pluginmanager.get_plugin('terminalreporter')
        if reporter is not None:
            color = 'red' if self.policy is Policy.FAIL else 'yellow'
            reporter.write_sep('=', message, **{color: True})
        if self.policy is Policy.FAIL:
            raise TaskAuditError(message)
        warn = getattr(request.node, 'warn', None)
        if warn is not None:
            warn(pytest.PytestWarning(message))
        else:  # pragma: no cover - defensive
            import warnings

            warnings.warn(pytest.PytestWarning(message), stacklevel=2)


def _object_loop(obj):
    loop = getattr(obj, '_loop', None)
    if loop is not None:
        return loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class _AuditPolicy(asyncio.AbstractEventLoopPolicy):
    """Event loop policy proxy that instruments newly created loops."""

    def __init__(self, auditor: TaskAuditor, original):
        self._auditor = auditor
        self._original = original

    def get_event_loop(self):
        return self._original.get_event_loop()

    def set_event_loop(self, loop):
        self._original.set_event_loop(loop)

    def new_event_loop(self):
        loop = self._original.new_event_loop()
        self._auditor.bind_loop(loop)
        return loop

    def get_child_watcher(self):
        return self._original.get_child_watcher()

    def set_child_watcher(self, watcher):
        self._original.set_child_watcher(watcher)


class TaskAuditHandle:
    """Fixture exposed to test code for explicit resource marking."""

    def __init__(self, auditor: TaskAuditor, request):
        self._auditor = auditor
        self._request = request

    def attach(self, task, *, name: str | None = None):
        """Mark an externally managed task/future for leak auditing."""
        return self._auditor.attach(task, name=name)


class _SessionAuditRequest:
    """Minimal request-like object for session boundary checks."""

    def __init__(self, config):
        self.config = config
        self.node = config.pluginmanager.get_plugin('session')


class TaskAuditPlugin:
    """pytest hooks and fixtures, registered only when audit is enabled."""

    def __init__(self, auditor: TaskAuditor):
        self.auditor = auditor

    def pytest_unconfigure(self, config):
        self.auditor.shutdown()

    def pytest_sessionfinish(self, session, exitstatus):
        if self.auditor.check_session:
            self._report_foreign_leftovers(session)

    def _report_foreign_leftovers(self, session) -> None:
        leftovers = [
            record
            for record in self.auditor._iter_records()
            if not record.ignored and record.is_live()
        ]
        if not leftovers:
            return
        report = LeakReport('session (event loop already closed)')
        report.cross_loop.extend(leftovers)
        reporter = session.config.pluginmanager.get_plugin('terminalreporter')
        if reporter is not None:
            reporter.write_sep('=', report.render(), yellow=True)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        fixture_name = getattr(fixturedef, 'argname', None) or getattr(
            fixturedef, 'arg', '<fixture>'
        )
        context = AuditContext(
            fixturedef.scope,
            f'fixture {fixture_name!r}',
            request.node.nodeid,
        )
        token = self.auditor.push_context(context)
        try:
            outcome = yield
        finally:
            self.auditor.pop_context(token)
        outcome.get_result()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        context = AuditContext('function', 'test call', item.nodeid)
        token = self.auditor.push_context(context)
        try:
            yield
        finally:
            self.auditor.pop_context(token)

    @pytest.fixture(scope='session', autouse=True)
    @staticmethod
    async def _testsuite_task_audit_session(pytestconfig):
        auditor: TaskAuditor = pytestconfig._testsuite_task_auditor
        if not auditor.check_session:
            yield
            return
        auditor.bind_loop(asyncio.get_running_loop())
        token = auditor.push_context(AuditContext('session', 'session fixture'))
        try:
            yield
        finally:
            auditor.pop_context(token)
            auditor.suppressed = False
            await auditor.audit_session(_SessionAuditRequest(pytestconfig))

    @pytest.fixture(autouse=True)
    @staticmethod
    async def _testsuite_task_audit_node(request):
        auditor: TaskAuditor = request.config._testsuite_task_auditor
        if not auditor.check_node:
            yield
            return
        auditor.bind_loop(asyncio.get_running_loop())
        auditor.set_current_node(request.node.nodeid)
        token = auditor.push_context(
            AuditContext(
                'function',
                f'node {request.node.nodeid!r}',
                request.node.nodeid,
            )
        )
        try:
            yield
        finally:
            try:
                auditor.suppressed = False
                await auditor.audit_node(request)
            finally:
                auditor.pop_context(token)

    @pytest.fixture
    @staticmethod
    def task_audit(request) -> TaskAuditHandle:
        return TaskAuditHandle(request.config._testsuite_task_auditor, request)


def _audit_enabled(config) -> bool:
    return bool(config.option.task_audit or config.getini('task_audit'))


def pytest_addoption(parser):
    group = parser.getgroup('task-audit', 'async task leak audit')
    group.addoption(
        '--task-audit',
        action='store_true',
        default=False,
        help='Enable async task / thread / socket watcher leak audit.',
    )
    group.addoption(
        '--task-audit-scope',
        choices=[scope.value for scope in AuditScope],
        default=None,
        help='Boundaries to audit: node, session or both (default).',
    )
    group.addoption(
        '--task-audit-grace',
        type=float,
        default=None,
        help=(
            'Seconds to wait for resources to finish before reporting them'
            f' (default {DEFAULT_GRACE_TIMEOUT}).'
        ),
    )
    group.addoption(
        '--task-audit-cancel-timeout',
        type=float,
        default=None,
        help=(
            'Seconds to wait after cancelling leaked tasks'
            f' (default {DEFAULT_CANCEL_TIMEOUT}).'
        ),
    )
    group.addoption(
        '--task-audit-policy',
        choices=[policy.value for policy in Policy],
        default=None,
        help='Leak policy: fail (default) or warn.',
    )
    parser.addini(
        'task_audit',
        'Enable async task leak audit.',
        type='bool',
        default=False,
    )
    parser.addini(
        'task_audit_scope',
        'Boundaries to audit: node, session or both.',
        default=AuditScope.BOTH.value,
    )
    parser.addini(
        'task_audit_grace',
        'Graceful window in seconds.',
        default=str(DEFAULT_GRACE_TIMEOUT),
    )
    parser.addini(
        'task_audit_cancel_timeout',
        'Cancellation timeout in seconds.',
        default=str(DEFAULT_CANCEL_TIMEOUT),
    )
    parser.addini(
        'task_audit_policy',
        'Leak policy: fail or warn.',
        default=Policy.FAIL.value,
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        f'{MARKER_IGNORE}: exclude node from task leak auditing',
    )
    if not _audit_enabled(config):
        return
    scope = config.option.task_audit_scope or config.getini('task_audit_scope')
    policy = config.option.task_audit_policy or config.getini(
        'task_audit_policy'
    )
    config.option.task_audit_scope = AuditScope(scope).value
    config.option.task_audit_policy = Policy(policy).value
    if config.option.task_audit_grace is None:
        config.option.task_audit_grace = float(
            config.getini('task_audit_grace')
        )
    if config.option.task_audit_cancel_timeout is None:
        config.option.task_audit_cancel_timeout = float(
            config.getini('task_audit_cancel_timeout')
        )
    auditor = TaskAuditor(config)
    auditor.install()
    config.pluginmanager.register(
        TaskAuditPlugin(auditor), 'testsuite_task_audit'
    )
    config._testsuite_task_auditor = auditor
