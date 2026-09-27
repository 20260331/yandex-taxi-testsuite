"""Failure artifact bundle pytest plugin.

Collects per-test service failure artifacts (service stdout/stderr slices,
structured service logs, redacted startup summaries, trace id and timeline)
linked by an atomic manifest. Bundles are deleted on success by default and
retained on failure as a directory or as a zip archive.
"""

import datetime
import logging
import os
import pathlib
import platform
import socket

import pytest

import testsuite
from testsuite.daemons import artifact

logger = logging.getLogger(__name__)

_SESSION_ID_ENV = 'TESTSUITE_ARTIFACT_SESSION_ID'
_XDIST_WORKER_ENV = 'PYTEST_XDIST_WORKER'
_DEFAULT_DIR = 'testsuite-artifacts'


def pytest_addoption(parser) -> None:
    group = parser.getgroup('services')
    group.addoption(
        '--service-artifacts',
        choices=artifact.MODES,
        default=None,
        help=(
            'Failure artifact bundle mode: '
            'off (disable), failure (retain only on failure, default), '
            'always (retain for every test).'
        ),
    )
    group.addoption(
        '--service-artifacts-dir',
        metavar='PATH',
        default=None,
        help='Directory to store artifact bundles in.',
    )
    group.addoption(
        '--service-artifacts-format',
        choices=artifact.FORMATS,
        default=None,
        help='Retained bundle format: dir (default) or zip.',
    )
    group.addoption(
        '--service-artifacts-max-bytes',
        metavar='SIZE',
        default=None,
        help=(
            'Maximum size in bytes of a single captured stream '
            '(supports K/M/G suffixes, 0 means unlimited). '
            'Default is 10M.'
        ),
    )
    group.addoption(
        '--service-artifacts-session-max-bytes',
        metavar='SIZE',
        default=None,
        help=(
            'Maximum total bytes written per worker session '
            '(0 means unlimited). Default is 256M.'
        ),
    )
    group.addoption(
        '--service-artifacts-retain',
        metavar='N',
        type=int,
        default=None,
        help='Number of recent artifact sessions to retain. Default is 5.',
    )
    group.addoption(
        '--service-artifacts-redact',
        metavar='PATTERNS',
        default=None,
        help=(
            'Comma separated list of additional case-insensitive env key '
            'patterns whose values must be redacted.'
        ),
    )

    parser.addini(
        'service_artifacts',
        'Failure artifact bundle mode (off, failure or always).',
        default=artifact.MODE_FAILURE,
    )
    parser.addini(
        'service_artifacts_dir',
        'Directory to store artifact bundles in.',
        default='',
    )
    parser.addini(
        'service_artifacts_format',
        'Retained bundle format (dir or zip).',
        default=artifact.FORMAT_DIR,
    )
    parser.addini(
        'service_artifacts_max_bytes',
        'Per-stream size cap (supports K/M/G suffixes, 0 unlimited).',
        default='10M',
    )
    parser.addini(
        'service_artifacts_session_max_bytes',
        'Per-session total size cap (0 unlimited).',
        default='256M',
    )
    parser.addini(
        'service_artifacts_retain',
        'Number of recent artifact sessions to retain.',
        default='5',
    )
    parser.addini(
        'service_artifacts_redact',
        'Additional env key patterns to redact (line separated).',
        type='linelist',
        default=[],
    )


def _setting(config, option_name: str, ini_name: str, default):
    value = config.getoption(option_name)
    if value is None:
        value = config.getini(ini_name)
    if value is None or value == '':
        return default
    return value


def build_config(config) -> artifact.ArtifactConfig:
    mode = _setting(config, 'service_artifacts', 'service_artifacts', None)
    bundle_format = _setting(
        config,
        'service_artifacts_format',
        'service_artifacts_format',
        artifact.FORMAT_DIR,
    )
    root_dir = pathlib.Path(
        _setting(
            config,
            'service_artifacts_dir',
            'service_artifacts_dir',
            config.rootpath / _DEFAULT_DIR,
        )
    )
    max_bytes = artifact.parse_size(
        _setting(
            config,
            'service_artifacts_max_bytes',
            'service_artifacts_max_bytes',
            artifact.DEFAULT_MAX_BYTES,
        )
    )
    session_max_bytes = artifact.parse_size(
        _setting(
            config,
            'service_artifacts_session_max_bytes',
            'service_artifacts_session_max_bytes',
            artifact.DEFAULT_SESSION_MAX_BYTES,
        )
    )
    retain = int(
        _setting(
            config,
            'service_artifacts_retain',
            'service_artifacts_retain',
            artifact.DEFAULT_RETAIN,
        )
    )
    extra_patterns = list(config.getini('service_artifacts_redact') or ())
    option_patterns = config.getoption('service_artifacts_redact')
    if option_patterns:
        extra_patterns.extend(
            pattern for pattern in option_patterns.split(',') if pattern.strip()
        )
    return artifact.ArtifactConfig(
        mode=mode,
        root_dir=root_dir,
        bundle_format=bundle_format,
        max_bytes=max_bytes,
        session_max_bytes=session_max_bytes,
        retain=retain,
        redactor=artifact.Redactor(extra_patterns),
    )


class ArtifactBundlePlugin:
    def __init__(self, config) -> None:
        self._config = config
        self._artifact_config = build_config(config)
        self._reports: dict[int, dict[str, pytest.TestReport]] = {}
        self._trace_ids: dict[int, str] = {}
        if self._artifact_config.enabled:
            self.store: artifact.ArtifactStore | artifact.NullStore = (
                artifact.ArtifactStore(
                    config=self._artifact_config,
                    session_id=self._session_id(),
                    worker=os.environ.get(_XDIST_WORKER_ENV, 'main'),
                    session_meta={
                        'hostname': socket.gethostname(),
                        'python': platform.python_version(),
                        'testsuite_version': testsuite.__version__,
                    },
                )
            )
        else:
            self.store = artifact.NullStore()

    @staticmethod
    def _session_id() -> str:
        session_id = os.environ.get(_SESSION_ID_ENV)
        if not session_id:
            session_id = (
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                + f'-{os.getpid()}'
            )
            os.environ[_SESSION_ID_ENV] = session_id
        return session_id

    def pytest_sessionstart(self, session) -> None:
        if not self.store.enabled:
            return
        try:
            self.store.start()
        except Exception:
            logger.exception(
                'Failed to initialize failure artifact store, '
                'artifact collection is disabled for this session'
            )
            self.store = artifact.NullStore()

    def pytest_runtest_setup(self, item) -> None:
        if not self.store.enabled:
            return
        try:
            bundle = self.store.begin_test(item.nodeid)
            bundle.add_event('test_setup', nodeid=item.nodeid)
            self._snapshot_log_positions(bundle, item)
        except Exception:
            logger.exception('Failed to start artifact bundle')

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        if not self.store.enabled:
            return report
        try:
            key = id(item)
            self._reports.setdefault(key, {})[report.when] = report
            bundle = self.store.current
            if bundle is not None:
                bundle.add_event(
                    'test_phase',
                    when=report.when,
                    outcome=report.outcome,
                )
            if report.when in ('setup', 'call'):
                trace_id = self._get_trace_id(item)
                if trace_id is not None:
                    self._trace_ids[key] = trace_id
            if report.when == 'setup':
                if bundle is not None:
                    self._snapshot_log_positions(bundle, item)
            elif report.when == 'teardown':
                reports = self._reports.pop(key, {})
                trace_id = self._trace_ids.pop(key, None)
                self._finalize(item, reports, trace_id)
        except Exception:
            logger.exception('Failure artifact report hook failed')
        return report

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if not self.store.enabled:
            return
        try:
            if self.store.current is not None:
                self.store.end_current(
                    outcome='interrupted',
                    trace_id=None,
                    failure_excerpt=None,
                    structured_logfiles=(),
                )
            self.store.cleanup_empty_dirs()
        except Exception:
            logger.exception('Failed to finish artifact session')

    def pytest_terminal_summary(
        self, terminalreporter, exitstatus, config
    ) -> None:
        retained = getattr(self.store, 'retained', None)
        if not retained:
            return
        root = self._artifact_config.root_dir
        terminalreporter.write_sep(
            '-',
            f'failure artifact bundles retained: {len(retained)} in {root}',
        )
        for path in retained[:20]:
            terminalreporter.write_line(f'  {path}')
        if len(retained) > 20:
            terminalreporter.write_line(f'  ... and {len(retained) - 20} more')

    def _finalize(self, item, reports, trace_id) -> None:
        failed_reports = [
            report for report in reports.values() if report.failed
        ]
        outcome = 'failed' if failed_reports else 'passed'
        failure_excerpt = None
        if failed_reports:
            failure_excerpt = failed_reports[0].longreprtext
        logfiles = self._structured_logfiles(flush=bool(failed_reports))
        self.store.end_current(
            outcome=outcome,
            trace_id=trace_id,
            failure_excerpt=failure_excerpt,
            structured_logfiles=logfiles,
        )

    def _structured_logfiles(self, *, flush: bool):
        logs_plugin = self._config.pluginmanager.get_plugin(
            'service_logs_plugin'
        )
        if logs_plugin is None:
            return ()
        if flush:
            logs_plugin.flush()
        return list(logs_plugin.iter_logfiles())

    def _snapshot_log_positions(self, bundle, item) -> None:
        logs_plugin = self._config.pluginmanager.get_plugin(
            'service_logs_plugin'
        )
        if logs_plugin is not None:
            bundle.snapshot_log_positions(logs_plugin.iter_logfiles())

    @staticmethod
    def _get_trace_id(item) -> str | None:
        try:
            return item.funcargs.get('testsuite_trace_id')
        except (AttributeError, KeyError, RuntimeError):
            return None


def pytest_configure(config) -> None:
    config.pluginmanager.register(
        ArtifactBundlePlugin(config),
        'artifact_bundle_plugin',
    )


@pytest.fixture(scope='session')
def _artifact_bundle_store(pytestconfig):
    plugin = pytestconfig.pluginmanager.get_plugin('artifact_bundle_plugin')
    return plugin.store


@pytest.fixture
def artifact_bundle(_artifact_bundle_store):
    """Artifact bundle of the currently running test node.

    Returns ``None`` when artifact collection is disabled. The bundle exposes
    ``path`` and ``add_event(event, **data)`` for custom timeline records.
    """
    return _artifact_bundle_store.current
