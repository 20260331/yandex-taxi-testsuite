import asyncio
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from testsuite.plugins import task_audit
from testsuite.plugins.task_audit import (
    AuditContext,
    LeakReport,
    LoopAudit,
    ResourceKind,
    TaskAuditError,
    TaskAuditor,
    TrackedRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

CONFTEST = """
pytest_plugins = ['testsuite.pytest_plugin']
"""

INI = """
[pytest]
asyncio_mode = auto
"""

COMMON_SNIPPETS = """
import asyncio
import os
import threading
import pytest


async def forever():
    await asyncio.Event().wait()


async def die():
    raise ValueError('boom-exc')


async def stubborn():
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        # Swallow cancellation and keep running.
        await asyncio.Event().wait()
"""


def run_audit_pytest(tmp_path, test_source, *extra_args, conftest_extra=''):
    (tmp_path / 'conftest.py').write_text(CONFTEST + conftest_extra)
    (tmp_path / 'pytest.ini').write_text(INI)
    (tmp_path / 'test_audit_case.py').write_text(
        COMMON_SNIPPETS + textwrap.dedent(test_source)
    )
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            str(tmp_path / 'test_audit_case.py'),
            '-p',
            'no:cacheprovider',
            *extra_args,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc


AUDIT_ARGS = (
    '--task-audit',
    '--task-audit-grace=0.3',
    '--task-audit-cancel-timeout=0.3',
)


# ---------------------------------------------------------------------------
# Subprocess integration tests
# ---------------------------------------------------------------------------


def test_disabled_by_default(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_leak_without_audit():
            asyncio.create_task(forever(), name='leak')
        """,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '1 passed' in proc.stdout


def test_still_running_task_fails_node(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_leak():
            asyncio.create_task(forever(), name='leaky-task')
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'still running after graceful window' in proc.stdout
    assert 'leaky-task' in proc.stdout
    assert 'TaskAuditError' in proc.stdout


def test_finishes_within_grace_passes(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_ok():
            async def short():
                await asyncio.sleep(0.05)

            task = asyncio.create_task(short())
            await task
        """,
        '--task-audit',
        '--task-audit-grace=2',
        '--task-audit-cancel-timeout=0.3',
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '1 passed' in proc.stdout
    assert 'leaked resource' not in proc.stdout


def test_unretrieved_exception_fails_node(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_unretrieved():
            asyncio.create_task(die())
            await asyncio.sleep(0.05)
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'exception that was never retrieved' in proc.stdout
    assert 'ValueError' in proc.stdout
    assert 'boom-exc' in proc.stdout


def test_cancel_incomplete_fails_node(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_cancel_incomplete():
            asyncio.create_task(stubborn(), name='stubborn-task')
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'still running after cancellation was requested' in proc.stdout
    assert 'stubborn-task' in proc.stdout


def test_socket_watcher_leak_fails_and_is_removed(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_watcher():
            reader, writer = os.pipe()
            loop = asyncio.get_running_loop()
            loop.add_reader(reader, lambda: None)
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'socket watchers still registered' in proc.stdout
    assert 'socket reader' in proc.stdout


def test_threaded_callback_leak_fails_node(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_threaded():
            import concurrent.futures
            import concurrent.futures.thread as futures_thread
            import weakref

            class DaemonPoolExecutor(
                concurrent.futures.ThreadPoolExecutor
            ):
                def _adjust_thread_count(self):
                    if self._idle_semaphore.acquire(timeout=0):
                        return
                    num_threads = len(self._threads)
                    if num_threads >= self._max_workers:
                        return
                    name = '%s_%d' % (
                        self._thread_name_prefix or self,
                        num_threads,
                    )
                    worker = threading.Thread(
                        name=name,
                        target=futures_thread._worker,
                        args=(
                            weakref.ref(self),
                            self._work_queue,
                            self._initializer,
                            self._initargs,
                        ),
                    )
                    worker.daemon = True
                    worker.start()
                    self._threads.add(worker)
                    futures_thread._threads_queues[worker] = (
                        self._work_queue
                    )

            loop = asyncio.get_running_loop()
            executor = DaemonPoolExecutor(max_workers=1)
            loop.run_in_executor(
                executor, lambda: threading.Event().wait()
            )
            await asyncio.sleep(0.1)
            # Blocked daemon workers must not delay interpreter exit.
            for worker in list(executor._threads):
                futures_thread._threads_queues.pop(worker, None)
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'threaded callback' in proc.stdout


def test_foreign_loop_task_is_cross_loop_leak(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        def test_foreign_loop():
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run_foreign_loop():
                asyncio.set_event_loop(loop)
                loop.create_task(forever())
                ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=run_foreign_loop, daemon=True
            )
            thread.start()
            assert ready.wait(5)
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'bound to a foreign event loop' in proc.stdout


def test_warn_policy_passes_with_warning(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_leak():
            asyncio.create_task(foreever_name := forever())
        """,
        *AUDIT_ARGS,
        '--task-audit-policy=warn',
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '1 passed' in proc.stdout
    assert 'PytestWarning' in proc.stdout
    assert 'leaked resource' in proc.stdout


def test_marker_ignore_skips_node_audit(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        @pytest.mark.task_audit_ignore
        async def test_ignored():
            asyncio.create_task(forever(), name='ignored-task')
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '1 passed' in proc.stdout
    assert 'leaked resource' not in proc.stdout


def test_session_fixture_leak_fails_session_boundary(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        @pytest.fixture(scope='session')
        async def leaky_session_fixture():
            asyncio.create_task(
                forever(), name='session-leak'
            )
            yield


        async def test_uses_session_fixture(leaky_session_fixture):
            assert True
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'session boundary' in proc.stdout
    assert 'session-leak' in proc.stdout
    assert "fixture 'leaky_session_fixture'" in proc.stdout


def test_scope_node_ignores_session_leak(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        @pytest.fixture(scope='session')
        async def leaky_session_fixture():
            asyncio.create_task(forever())
            yield


        async def test_uses_session_fixture(leaky_session_fixture):
            assert True
        """,
        '--task-audit',
        '--task-audit-scope=node',
        '--task-audit-grace=0.3',
        '--task-audit-cancel-timeout=0.3',
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '1 passed' in proc.stdout


def test_enabled_via_ini(tmp_path):
    (tmp_path / 'conftest.py').write_text(CONFTEST)
    (tmp_path / 'pytest.ini').write_text(
        textwrap.dedent(
            """
            [pytest]
            asyncio_mode = auto
            task_audit = true
            task_audit_grace = 0.3
            task_audit_cancel_timeout = 0.3
            """
        ).strip()
    )
    (tmp_path / 'test_audit_case.py').write_text(
        COMMON_SNIPPETS
        + textwrap.dedent(
            """
            async def test_leak():
                asyncio.create_task(
                    forever(), name='ini-enabled-leak'
                )
            """
        )
    )
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            str(tmp_path / 'test_audit_case.py'),
            '-p',
            'no:cacheprovider',
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert 'ini-enabled-leak' in proc.stdout


def test_user_managed_event_loop_fixture(tmp_path):
    user_loop_conftest = """
    import asyncio as _asyncio
    import pytest


    @pytest.fixture
    def event_loop():
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        yield loop
        loop.close()
    """
    proc = run_audit_pytest(
        tmp_path,
        """
        async def test_leak_on_user_loop():
            asyncio.create_task(
                forever(), name='user-loop-leak'
            )


        async def test_clean_on_user_loop():
            await asyncio.sleep(0)
        """,
        *AUDIT_ARGS,
        conftest_extra=textwrap.dedent(user_loop_conftest),
    )
    assert proc.returncode != 0
    assert 'user-loop-leak' in proc.stdout
    assert '2 passed, 1 error' in proc.stdout


def test_sync_test_with_leaking_async_fixture(tmp_path):
    proc = run_audit_pytest(
        tmp_path,
        """
        @pytest.fixture
        async def leaky_fixture():
            asyncio.create_task(
                forever(), name='sync-test-leak'
            )
            yield


        def test_sync(leaky_fixture):
            assert True
        """,
        *AUDIT_ARGS,
    )
    assert proc.returncode != 0
    assert 'sync-test-leak' in proc.stdout


# ---------------------------------------------------------------------------
# Direct unit tests for instrumentation primitives
# ---------------------------------------------------------------------------


class _StubAuditor:
    suppressed = False

    def current_context(self):
        return AuditContext('function', 'unit test', 'test-node')

    def capture_source(self):
        return ['  File "unit_test.py", line 1, in test']


def test_loop_instrumentation_records_resources():
    # Construct the loop directly, bypassing the event loop policy which
    # may already instrument loops when the outer session is audited.
    loop = asyncio.SelectorEventLoop()
    loop_audit = LoopAudit(_StubAuditor(), loop)
    assert loop_audit.install()

    async def scenario():
        done_task = loop.create_task(asyncio.sleep(0))
        await done_task

        future = loop.run_in_executor(None, lambda: 1)
        await future

        reader, writer = os.pipe()
        loop.add_reader(reader, lambda: None)

        records = list(loop_audit.records)
        kinds = {record.kind for record in records}
        assert ResourceKind.TASK in kinds
        assert ResourceKind.THREADED in kinds
        assert ResourceKind.READER in kinds

        reader_record = next(
            record for record in records if record.kind is ResourceKind.READER
        )
        assert reader_record.is_live()
        assert reader_record.nodeid == 'test-node'
        loop.remove_reader(reader)
        assert not reader_record.is_live()
        os.close(reader)
        os.close(writer)

        task_record = loop_audit.find_record(done_task)
        assert task_record is not None
        assert not task_record.is_live()

    loop.run_until_complete(scenario())

    # restore() returns original loop methods.
    loop_audit.restore()
    assert not isinstance(loop.create_task, task_audit._WrappedMethod)
    loop.close()


def test_attach_marks_external_task_and_is_idempotent():
    config = SimpleNamespace(
        option=SimpleNamespace(
            task_audit_grace=0.3,
            task_audit_cancel_timeout=0.3,
            task_audit_policy='fail',
            task_audit_scope='both',
        ),
        rootpath='/tmp',
    )
    config.getini = lambda name: ''

    auditor = TaskAuditor(config)
    loop = asyncio.SelectorEventLoop()

    async def scenario():
        task = loop.create_task(asyncio.sleep(10))
        await asyncio.sleep(0)
        record = auditor.attach(task, name='external-mark')
        assert record is auditor.attach(task)
        assert record.name == 'external-mark'
        assert record.obj is task
        task.cancel()
        return record

    auditor.bind_loop(loop)
    loop.run_until_complete(scenario())
    auditor.shutdown()
    loop.close()


def test_attach_without_loop_binding_raises():
    config = SimpleNamespace(
        option=SimpleNamespace(
            task_audit_grace=0.3,
            task_audit_cancel_timeout=0.3,
            task_audit_policy='fail',
            task_audit_scope='both',
        ),
        rootpath='/tmp',
    )
    config.getini = lambda name: ''
    auditor = TaskAuditor(config)
    with pytest.raises(TaskAuditError):
        auditor.attach(object())


def test_leak_report_renders_all_categories():
    loop = asyncio.new_event_loop()
    report = LeakReport('node test-x')

    def record(kind, **kwargs):
        return TrackedRecord(
            kind,
            loop,
            name=kwargs.get('name', f'record-{kind}'),
            scope='function',
            label='test call',
            nodeid='test-x',
            source=['  File "t.py", line 1, in test'],
            fd=kwargs.get('fd'),
            obj=kwargs.get('obj'),
        )

    report.still_running.append(
        record(ResourceKind.TASK, name='running-task', obj=None)
    )
    report.cancel_incomplete.append(
        record(ResourceKind.TASK, name='stub-task', obj=None)
    )
    report.cross_loop.append(
        record(ResourceKind.TASK, name='foreign-task', obj=None)
    )
    report.watchers.append(
        record(ResourceKind.READER, fd=7, callback=lambda: None)
    )
    report.unretrieved.append(
        (
            record(ResourceKind.TASK, name='failed-task', obj=None),
            ValueError('x'),
        )
    )

    assert report.has_leaks
    assert report.total == 5
    text = report.render()
    assert 'still running after graceful window' in text
    assert 'never retrieved' in text
    assert 'still running after cancellation' in text
    assert 'foreign event loop' in text
    assert 'socket watchers still registered' in text
    loop.close()
