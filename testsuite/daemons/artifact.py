"""Failure artifact bundle core.

Collects per-test service failure artifacts:

* service stdout/stderr slices;
* structured service log file slices;
* sanitized service startup summaries (command, ports, environment);
* trace id and append-only timeline;
* atomic manifest that links test node, service instances and exit reasons.

The module is pytest-agnostic, see
``testsuite.plugins.artifact_bundle`` for the pytest integration.
"""

import collections
import collections.abc
import contextlib
import dataclasses
import datetime
import fcntl
import inspect
import io
import json
import logging
import os
import pathlib
import re
import shutil
import threading
import time
import urllib.parse
import uuid
import zipfile
from typing import Any

from testsuite.daemons import spawn as _spawn

logger = logging.getLogger(__name__)

MODE_OFF = 'off'
MODE_FAILURE = 'failure'
MODE_ALWAYS = 'always'
MODES = (MODE_OFF, MODE_FAILURE, MODE_ALWAYS)

FORMAT_DIR = 'dir'
FORMAT_ZIP = 'zip'
FORMATS = (FORMAT_DIR, FORMAT_ZIP)

MANIFEST_NAME = 'manifest.json'
TIMELINE_NAME = 'timeline.jsonl'
SCHEMA_VERSION = 1

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_SESSION_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_RETAIN = 5

REDACTED_VALUE = '***redacted***'
FAILURE_EXCERPT_LIMIT = 4096
_PENDING_LIMIT_BYTES = 256 * 1024

DEFAULT_SENSITIVE_PATTERNS = (
    'PASSWORD',
    'PASSWD',
    'PWD',
    'TOKEN',
    'SECRET',
    'API_KEY',
    'APIKEY',
    'CREDENTIAL',
    'PRIVATE_KEY',
    'COOKIE',
    'AUTHORIZATION',
)

_SIZE_UNITS = {
    '': 1,
    'B': 1,
    'K': 1024,
    'KIB': 1024,
    'KB': 1000,
    'M': 1024**2,
    'MIB': 1024**2,
    'MB': 1000**2,
    'G': 1024**3,
    'GIB': 1024**3,
    'GB': 1000**3,
}


def parse_size(value: str | int) -> int:
    """Parse size value, e.g. ``10M``, ``1.5GiB`` or plain bytes.

    ``0`` means unlimited.
    """
    if isinstance(value, int):
        return value
    text = str(value).strip().upper()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([A-Z]*)', text)
    if not match:
        raise ValueError(f'Invalid size value: {value!r}')
    number, unit = match.groups()
    if unit not in _SIZE_UNITS:
        raise ValueError(f'Invalid size unit in value: {value!r}')
    return int(float(number) * _SIZE_UNITS[unit])


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def slugify(text: str) -> str:
    text = text.replace('/', '.').replace('::', '.')
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('._')
    return text[:80] or 'test'


def atomic_write_json(path: pathlib.Path, data: dict) -> None:
    """Atomically write json file: temp file in same dir + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with tmp_path.open('w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2, sort_keys=True, ensure_ascii=False)
        fp.write('\n')
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp_path, path)


@contextlib.contextmanager
def _file_lock(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class Redactor:
    """Redacts sensitive environment values and command line arguments."""

    def __init__(self, extra_patterns: collections.abc.Iterable[str] = ()):
        patterns = tuple(
            dict.fromkeys(
                str(pattern).upper()
                for pattern in (*DEFAULT_SENSITIVE_PATTERNS, *extra_patterns)
                if str(pattern).strip()
            )
        )
        self._patterns = patterns

    @property
    def patterns(self) -> tuple[str, ...]:
        return self._patterns

    def is_sensitive_key(self, key: str) -> bool:
        upper_key = str(key).upper()
        return any(pattern in upper_key for pattern in self._patterns)

    def redact_mapping(
        self, mapping: collections.abc.Mapping[str, Any] | None
    ) -> dict[str, str]:
        result = {}
        for key, value in (mapping or {}).items():
            if self.is_sensitive_key(key):
                result[str(key)] = REDACTED_VALUE
            else:
                result[str(key)] = str(value)
        return result

    def redact_args(self, args: collections.abc.Sequence) -> list[str]:
        result = []
        for arg in args:
            text = str(arg)
            key, sep, _value = text.partition('=')
            if sep and self.is_sensitive_key(key):
                result.append(f'{key}={REDACTED_VALUE}')
            else:
                result.append(text)
        return result

    def command_line(self, args: collections.abc.Sequence) -> str:
        return ' '.join(self.redact_args(args))

    def ports_from_env(
        self, env: collections.abc.Mapping[str, Any] | None
    ) -> list[dict[str, str]]:
        result = []
        for key, value in (env or {}).items():
            if 'PORT' in str(key).upper():
                result.append({'name': str(key), 'value': str(value)})
        return result

    def secret_values(
        self, mapping: collections.abc.Mapping[str, Any] | None
    ) -> list[str]:
        result = []
        for key, value in (mapping or {}).items():
            text = str(value)
            if self.is_sensitive_key(key) and text:
                result.append(text)
        return result

    def redact_text(self, text: str, secrets: collections.abc.Iterable[str]):
        result = text
        for secret in secrets:
            if secret and secret in result:
                result = result.replace(secret, REDACTED_VALUE)
        return result


@dataclasses.dataclass(frozen=True)
class ArtifactConfig:
    mode: str
    root_dir: pathlib.Path
    bundle_format: str
    max_bytes: int
    session_max_bytes: int
    retain: int
    redactor: Redactor

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF

    @property
    def retain_on_success(self) -> bool:
        return self.mode == MODE_ALWAYS


class _CappedStream:
    """Line writer bounded by per-stream byte cap.

    Callers are expected to truncate ``data`` to remaining stream and
    session quotas and to pass the dropped amount explicitly.
    """

    def __init__(self, path: pathlib.Path, max_bytes: int):
        self._path = path
        self._max_bytes = max_bytes
        self.written = 0
        self.dropped = 0
        self._fp: io.BufferedWriter | None = None
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        if not self._max_bytes:
            return -1
        return max(0, self._max_bytes - self.written)

    def write(self, data: bytes, dropped: int = 0) -> None:
        with self._lock:
            self.dropped += dropped
            if not data:
                return
            if self._fp is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._fp = self._path.open('ab')
            self._fp.write(data)
            self.written += len(data)

    def close(self) -> None:
        with self._lock:
            if self._fp is None:
                return
            if self.dropped:
                self._fp.write(
                    b'[testsuite artifacts: dropped %d bytes due to '
                    b'stream capacity %d]\n' % (self.dropped, self._max_bytes)
                )
            self._fp.close()
            self._fp = None

    def exists(self) -> bool:
        return self._path.exists()


class ServiceCapture:
    """Tracks a single service process spawn attempt."""

    def __init__(self, store: 'ArtifactStore', startup: dict[str, Any]):
        self.id = uuid.uuid4().hex
        self._store = store
        self.startup = startup
        self.started_at = utcnow()
        self.ready_at: str | None = None
        self.exited_at: str | None = None
        self.exit_code: int | None = None
        self.exit_reason: str | None = None
        self.spawn_error: str | None = None
        self.daemon_instance_id: str | None = None
        self._process = None
        self._bundle: TestBundle | None = None
        self._pending: collections.deque[tuple[str, bytes]] = (
            collections.deque()
        )
        self._pending_bytes = 0

    def wrap_handlers(self, stdout_handler, stderr_handler):
        return (
            self._wrap_handler(stdout_handler, 'stdout'),
            self._wrap_handler(stderr_handler, 'stderr'),
        )

    def _wrap_handler(self, user_handler, stream: str):
        if user_handler is None:

            def capture_only_handler(line: bytes) -> None:
                self._safe_write(stream, line)

            return capture_only_handler
        if inspect.iscoroutinefunction(user_handler):

            async def async_handler(line: bytes):
                self._safe_write(stream, line)
                return await user_handler(line)

            return async_handler

        def sync_handler(line: bytes):
            self._safe_write(stream, line)
            return user_handler(line)

        return sync_handler

    def _safe_write(self, stream: str, line: bytes) -> None:
        try:
            self.write(stream, line)
        except Exception:
            logger.exception('Failed to capture service %s', stream)

    def write(self, stream: str, line: bytes) -> None:
        bundle = self._bundle
        if bundle is not None:
            bundle.write_service_stream(self.id, stream, line)
            return
        # Bound later (e.g. output during health check): keep bounded tail.
        self._pending.append((stream, line))
        self._pending_bytes += len(line)
        while self._pending_bytes > _PENDING_LIMIT_BYTES and self._pending:
            _stream, dropped_line = self._pending.popleft()
            self._pending_bytes -= len(dropped_line)

    def attach(self, process) -> None:
        self._process = process
        self.ready_at = utcnow()
        self._store._on_capture_attached(self, process)

    def bind(self, bundle: 'TestBundle') -> None:
        if self._bundle is bundle:
            return
        self._bundle = bundle
        bundle.bind_service(self)
        while self._pending:
            stream, line = self._pending.popleft()
            self._pending_bytes -= len(line)
            bundle.write_service_stream(self.id, stream, line)

    def set_scope(self, scope_name: str | None) -> None:
        if scope_name is not None:
            self.startup['scope'] = scope_name

    def fail(self, exc: BaseException) -> None:
        self.spawn_error = f'{type(exc).__name__}: {exc}'

    def finish(self) -> None:
        process = self._process
        if process is not None:
            self.exit_code = process.returncode
            if self.exit_code is not None:
                self.exited_at = utcnow()
                if self.exit_code != 0:
                    with contextlib.suppress(Exception):
                        self.exit_reason = _spawn.exit_code_error(process).args[
                            0
                        ]
        self._store._on_capture_finished(self)

    def describe(self, redactor: Redactor) -> dict[str, Any]:
        startup = self.startup
        env = {}
        env.update(startup.get('env') or {})
        env.update(startup.get('options_env') or {})
        command = [str(arg) for arg in startup.get('command') or ()]
        ports = list(redactor.ports_from_env(env))
        ping_url = startup.get('ping_url')
        if ping_url:
            parsed = urllib.parse.urlsplit(ping_url)
            if parsed.hostname:
                ports.append(
                    {
                        'name': 'ping_url',
                        'value': f'{parsed.hostname}:{parsed.port or 80}',
                    }
                )
        return {
            'instance_id': self.id,
            'daemon_instance_id': self.daemon_instance_id,
            'scope': startup.get('scope'),
            'mode': startup.get('mode'),
            'pid': getattr(self._process, 'pid', None),
            'command': redactor.redact_args(command),
            'command_line': redactor.command_line(command),
            'base_command': [
                str(arg) for arg in startup.get('base_command') or ()
            ],
            'cwd': startup.get('cwd') or os.getcwd(),
            'ping_url': ping_url,
            'ports': ports,
            'env': redactor.redact_mapping(env),
            'shutdown_signal': startup.get('shutdown_signal'),
            'started_at': self.started_at,
            'ready_at': self.ready_at,
            'exited_at': self.exited_at,
            'exit_code': self.exit_code,
            'exit_reason': self.exit_reason,
            'spawn_error': self.spawn_error,
        }


class TestBundle:
    """Artifact bundle of a single test node."""

    def __init__(
        self,
        *,
        store: 'ArtifactStore',
        path: pathlib.Path,
        nodeid: str,
        session: dict[str, Any],
    ):
        self._store = store
        self._config = store.config
        self.path = path
        self.nodeid = nodeid
        self.session = session
        self.started_at = utcnow()
        self._start_monotonic = time.monotonic()
        self._streams: dict[tuple[str, str], _CappedStream] = {}
        self._services: dict[str, ServiceCapture] = {}
        self._log_positions: dict[tuple[str, str], int] = {}
        self._timeline_fp: io.BufferedWriter | None = None
        self._lock = threading.Lock()
        self.finalized = False
        self.retained_path: pathlib.Path | None = None

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def add_event(self, event: str, **data) -> None:
        try:
            record = {
                'ts': utcnow(),
                'monotonic': round(time.monotonic() - self._start_monotonic, 6),
                'event': event,
                **data,
            }
            with self._lock:
                if self._timeline_fp is None:
                    self.path.mkdir(parents=True, exist_ok=True)
                    self._timeline_fp = (self.path / TIMELINE_NAME).open('ab')
                line = json.dumps(
                    record, sort_keys=True, ensure_ascii=False, default=str
                )
                self._timeline_fp.write((line + '\n').encode('utf-8'))
                self._timeline_fp.flush()
        except Exception:
            logger.exception('Failed to write artifact timeline event')

    def snapshot_log_positions(
        self, logfiles: collections.abc.Iterable[tuple[pathlib.Path, str]]
    ) -> None:
        for path, title in logfiles:
            key = (str(path), title)
            if key in self._log_positions:
                continue
            try:
                position = path.stat().st_size if path.exists() else 0
            except OSError:
                position = 0
            self._log_positions[key] = position

    def bind_service(self, capture: ServiceCapture) -> None:
        if capture.id in self._services:
            return
        self._services[capture.id] = capture

    def write_service_stream(
        self, capture_id: str, stream: str, data: bytes
    ) -> None:
        capped = self._streams.get((capture_id, stream))
        if capped is None:
            capped = _CappedStream(
                self.path / 'services' / capture_id / f'{stream}.log',
                self._config.max_bytes,
            )
            self._streams[(capture_id, stream)] = capped
        original_len = len(data)
        remaining = capped.remaining
        if remaining == 0:
            capped.write(b'', dropped=original_len)
            return
        if remaining > 0:
            data = data[:remaining]
        if self._config.session_max_bytes:
            allowed = self._store.reserve(len(data))
            data = data[:allowed]
        capped.write(data, dropped=original_len - len(data))

    def service_stream_info(
        self, capture_id: str, stream: str
    ) -> dict[str, Any] | None:
        capped = self._streams.get((capture_id, stream))
        if capped is None or not capped.exists():
            return None
        return {
            'path': str(
                (pathlib.Path('services') / capture_id / f'{stream}.log')
            ),
            'bytes': capped.written,
            'dropped_bytes': capped.dropped,
        }

    def close(self) -> None:
        for capped in self._streams.values():
            with contextlib.suppress(Exception):
                capped.close()
        if self._timeline_fp is not None:
            with contextlib.suppress(Exception):
                self._timeline_fp.close()
            self._timeline_fp = None

    def finalize(
        self,
        *,
        outcome: str,
        trace_id: str | None,
        failure_excerpt: str | None,
        structured_logfiles,
    ) -> pathlib.Path | None:
        if self.finalized:
            return self.retained_path
        self.finalized = True
        finished_at = utcnow()
        retain = outcome != 'passed' or self._config.retain_on_success
        if retain:
            # Flush truncation markers before manifest and archive are
            # produced.
            self._close_streams()
            self._materialize(
                outcome=outcome,
                trace_id=trace_id,
                failure_excerpt=failure_excerpt,
                finished_at=finished_at,
                structured_logfiles=structured_logfiles,
            )
        self.close()
        if not retain:
            shutil.rmtree(self.path, ignore_errors=True)
            self.retained_path = None
            return None
        if self._config.bundle_format == FORMAT_ZIP:
            self.retained_path = self._make_zip()
        else:
            self.retained_path = self.path
        return self.retained_path

    def _close_streams(self) -> None:
        for capped in self._streams.values():
            with contextlib.suppress(Exception):
                capped.close()

    def _materialize(
        self,
        *,
        outcome: str,
        trace_id: str | None,
        failure_excerpt: str | None,
        finished_at: str,
        structured_logfiles,
    ) -> None:
        secrets = self._collect_secrets()
        services = []
        for capture in self._services.values():
            entry = capture.describe(self._config.redactor)
            service_dir = self.path / 'services' / capture.id
            try:
                service_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(service_dir / 'startup.json', entry)
            except Exception:
                logger.exception('Failed to write service startup summary')
            entry['stdout'] = self.service_stream_info(capture.id, 'stdout')
            entry['stderr'] = self.service_stream_info(capture.id, 'stderr')
            services.append(entry)
        service_logs = self._slice_structured_logs(structured_logfiles)
        if failure_excerpt:
            failure_excerpt = self._config.redactor.redact_text(
                failure_excerpt[:FAILURE_EXCERPT_LIMIT], secrets
            )
        manifest = {
            'schema_version': SCHEMA_VERSION,
            'session': self.session,
            'test': {
                'nodeid': self.nodeid,
                'path': self.nodeid.split('::', 1)[0],
                'name': self.nodeid.split('::')[-1],
                'outcome': outcome,
                'trace_id': trace_id,
                'started_at': self.started_at,
                'finished_at': finished_at,
                'duration_ms': round(
                    (time.monotonic() - self._start_monotonic) * 1000, 3
                ),
                'failure_excerpt': failure_excerpt,
            },
            'services': services,
            'service_logs': service_logs,
            'artifacts': {
                'manifest': MANIFEST_NAME,
                'timeline': TIMELINE_NAME,
                'services_dir': 'services',
                'service_logs_dir': 'service_logs',
            },
        }
        atomic_write_json(self.path / MANIFEST_NAME, manifest)

    def _collect_secrets(self) -> list[str]:
        secrets = []
        for capture in self._services.values():
            secrets.extend(
                self._config.redactor.secret_values(capture.startup.get('env'))
            )
            secrets.extend(
                self._config.redactor.secret_values(
                    capture.startup.get('options_env')
                )
            )
        return secrets

    def _slice_structured_logs(self, logfiles) -> list[dict[str, Any]]:
        result = []
        log_dir = self.path / 'service_logs'
        for path, title in logfiles or ():
            start = self._log_positions.get((str(path), title), 0)
            try:
                if not path.exists():
                    continue
                with path.open('rb') as fp:
                    fp.seek(start)
                    data = fp.read(self._config.max_bytes + 1)
            except OSError:
                logger.exception('Failed to read service log %s', path)
                continue
            if not data:
                continue
            truncated = len(data) > self._config.max_bytes
            data = data[: self._config.max_bytes]
            name = slugify(title) + '.log'
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                target = log_dir / name
                with target.open('wb') as fp:
                    fp.write(data)
                    if truncated:
                        fp.write(
                            b'\n[testsuite artifacts: service log truncated]\n'
                        )
            except OSError:
                logger.exception('Failed to write service log slice %s', name)
                continue
            result.append(
                {
                    'title': title,
                    'path': str(pathlib.Path('service_logs') / name),
                    'bytes': len(data),
                    'truncated': truncated,
                }
            )
        return result

    def _make_zip(self) -> pathlib.Path:
        # Bundle directory names contain dots, append the suffix instead of
        # replacing the last path suffix.
        zip_target = self.path.parent / f'{self.path.name}.zip'
        zip_tmp = self.path.with_name(
            f'.{self.path.name}.{uuid.uuid4().hex}.zip.tmp'
        )
        with zipfile.ZipFile(zip_tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
            for child in self.path.rglob('*'):
                if child.is_file():
                    archive.write(
                        child,
                        str(
                            pathlib.Path(self.path.name)
                            / child.relative_to(self.path)
                        ),
                    )
        os.replace(zip_tmp, zip_target)
        shutil.rmtree(self.path, ignore_errors=True)
        return zip_target


class ArtifactStore:
    """Session-scoped store of test artifact bundles."""

    def __init__(
        self,
        *,
        config: ArtifactConfig,
        session_id: str,
        worker: str,
        session_meta: dict[str, Any],
    ):
        self.config = config
        self.session_id = session_id
        self.worker = worker
        self.session_meta = dict(session_meta)
        self.session_meta.update(
            {
                'id': session_id,
                'worker': worker,
                'started_at': utcnow(),
            }
        )
        self.session_dir = config.root_dir / f'session-{session_id}'
        self.worker_dir = self.session_dir / worker
        self.bundles_dir = self.worker_dir / 'bundles'
        self.current: TestBundle | None = None
        self.retained: list[pathlib.Path] = []
        self._captures_by_process: dict[int, ServiceCapture] = {}
        self._last_capture: ServiceCapture | None = None
        self._session_bytes = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return True

    def start(self) -> None:
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            self.prune_old_sessions(
                self.config.root_dir, self.config.retain, self.session_id
            )

    def begin_test(self, nodeid: str) -> TestBundle:
        name = f'{slugify(nodeid)}-{uuid.uuid4().hex[:8]}'
        bundle = TestBundle(
            store=self,
            path=self.bundles_dir / name,
            nodeid=nodeid,
            session=self.session_meta,
        )
        self.current = bundle
        return bundle

    def end_current(self, **kwargs) -> pathlib.Path | None:
        bundle = self.current
        if bundle is None:
            return None
        try:
            retained = bundle.finalize(**kwargs)
            if retained is not None:
                self.retained.append(retained)
            return retained
        except Exception:
            logger.exception('Failed to finalize artifact bundle')
            return None
        finally:
            self.current = None

    def start_capture(self, startup: dict[str, Any]) -> ServiceCapture:
        capture = ServiceCapture(self, startup)
        self._last_capture = capture
        if self.current is not None:
            # Bind immediately so that failed spawns (which never reach
            # attach) still appear in the bundle.
            capture.bind(self.current)
            self.current.add_event(
                'service_spawn',
                capture_id=capture.id,
                command=self.config.redactor.command_line(
                    startup.get('command') or ()
                ),
            )
        return capture

    def bind_process(
        self,
        process,
        scope_name: str | None = None,
        daemon_instance_id: str | None = None,
    ) -> None:
        try:
            capture: ServiceCapture | None
            if process is None:
                capture = self._last_capture
            else:
                capture = self._captures_by_process.get(id(process))
            if capture is None:
                return
            capture.set_scope(scope_name)
            if daemon_instance_id is not None:
                capture.daemon_instance_id = daemon_instance_id
            if self.current is not None:
                capture.bind(self.current)
                self.current.add_event(
                    'service_bound',
                    capture_id=capture.id,
                    scope=scope_name,
                )
        except Exception:
            logger.exception('Failed to bind service capture')

    def reserve(self, size: int) -> int:
        if not self.config.session_max_bytes:
            return size
        with self._lock:
            allowed = max(
                0,
                self.config.session_max_bytes - self._session_bytes,
            )
            allowed = min(allowed, size)
            self._session_bytes += allowed
            return allowed

    def _on_capture_attached(self, capture: ServiceCapture, process) -> None:
        if process is not None:
            self._captures_by_process[id(process)] = capture
        self._last_capture = capture
        if self.current is not None:
            capture.bind(self.current)
            self.current.add_event(
                'service_ready',
                capture_id=capture.id,
                pid=getattr(process, 'pid', None),
            )

    def _on_capture_finished(self, capture: ServiceCapture) -> None:
        if self.current is not None:
            self.current.add_event(
                'service_exit',
                capture_id=capture.id,
                exit_code=capture.exit_code,
                exit_reason=capture.exit_reason,
                spawn_error=capture.spawn_error,
            )

    def cleanup_empty_dirs(self) -> None:
        # Remove empty leaf dirs bottom-up; the root stays (holds the
        # cross-process prune lock).
        with contextlib.suppress(OSError):
            os.removedirs(self.bundles_dir)

    @staticmethod
    def prune_old_sessions(
        root: pathlib.Path, retain: int, current_session_id: str
    ) -> None:
        if retain <= 0:
            return
        with _file_lock(root / '.lock'):
            sessions = [
                path
                for path in root.glob('session-*')
                if path.is_dir()
                and path.name != f'session-{current_session_id}'
            ]
            sessions.sort(key=lambda path: path.stat().st_mtime)
            for old_session in sessions[: max(0, len(sessions) - retain + 1)]:
                shutil.rmtree(old_session, ignore_errors=True)


class NullCapture:
    """No-op capture used when artifacts are disabled."""

    id = None
    spawn_error = None

    def wrap_handlers(self, stdout_handler, stderr_handler):
        return stdout_handler, stderr_handler

    def attach(self, process) -> None:
        pass

    def finish(self) -> None:
        pass

    def fail(self, exc: BaseException) -> None:
        pass

    def set_scope(self, scope_name) -> None:
        pass


class NullStore:
    """Zero-cost store used when artifacts are disabled."""

    enabled = False
    current = None
    retained: list = []

    def start(self) -> None:
        pass

    def begin_test(self, nodeid: str) -> None:
        return None

    def end_current(self, **kwargs) -> None:
        return None

    def start_capture(self, startup) -> NullCapture:
        return NullCapture()

    def bind_process(
        self, process, scope_name=None, daemon_instance_id=None
    ) -> None:
        pass

    def cleanup_empty_dirs(self) -> None:
        pass
