# pylint: disable=protected-access
import json
import os
import pathlib
import signal
import zipfile

from testsuite.daemons import artifact


class FakeProcess:
    def __init__(self, pid=4242, returncode=None):
        self.pid = pid
        self.returncode = returncode


def make_config(tmp_path, **overrides) -> artifact.ArtifactConfig:
    return artifact.ArtifactConfig(
        mode=overrides.get('mode', artifact.MODE_FAILURE),
        root_dir=tmp_path / 'artifacts',
        bundle_format=overrides.get('bundle_format', artifact.FORMAT_DIR),
        max_bytes=overrides.get('max_bytes', artifact.DEFAULT_MAX_BYTES),
        session_max_bytes=overrides.get(
            'session_max_bytes', artifact.DEFAULT_SESSION_MAX_BYTES
        ),
        retain=overrides.get('retain', artifact.DEFAULT_RETAIN),
        redactor=artifact.Redactor(overrides.get('redact', ())),
    )


def make_store(tmp_path, session_id='session-test', **config_overrides):
    store = artifact.ArtifactStore(
        config=make_config(tmp_path, **config_overrides),
        session_id=session_id,
        worker='gw0',
        session_meta={'hostname': 'testhost'},
    )
    store.start()
    return store


def read_manifest(bundle_path: pathlib.Path) -> dict:
    with (bundle_path / artifact.MANIFEST_NAME).open() as fp:
        return json.load(fp)


def read_timeline(bundle_path: pathlib.Path) -> list[dict]:
    result = []
    with (bundle_path / artifact.TIMELINE_NAME).open() as fp:
        for line in fp:
            result.append(json.loads(line))
    return result


def spawn_service(
    store: artifact.ArtifactStore,
    process: FakeProcess | None = None,
    *,
    env=None,
    options_env=None,
    ping_url='http://localhost:8080/ping',
    scope_name='my.service',
    daemon_instance_id='di-1',
):
    startup = {
        'command': ('svc-binary', '--api-token=raw-token'),
        'base_command': ('launcher',),
        'env': env or {'PASSWORD': 'secret-password', 'NODE': '1'},
        'options_env': options_env or {'SERVICE_PORT': '8080'},
        'cwd': None,
        'ping_url': ping_url,
        'shutdown_signal': 'SIGINT',
        'mode': 'spawn',
    }
    capture = store.start_capture(startup)
    stdout_handler, stderr_handler = capture.wrap_handlers(None, None)
    process = process if process is not None else FakeProcess()
    capture.attach(process)
    store.bind_process(
        process,
        scope_name=scope_name,
        daemon_instance_id=daemon_instance_id,
    )
    return capture, process, stdout_handler, stderr_handler


def test_failure_bundle_contains_all_artifacts(tmp_path):
    store = make_store(tmp_path)
    store.begin_test('tests/test_svc.py::test_request')
    capture, process, stdout_handler, stderr_handler = spawn_service(store)

    # Lines produced during health check (before bind) are buffered.
    capture.write('stdout', b'boot line\n')
    stdout_handler(b'request handled\n')
    stderr_handler(b'warn line\n')
    process.returncode = -int(signal.SIGABRT)
    capture.finish()

    bundle_path = store.end_current(
        outcome='failed',
        trace_id='trace-xyz',
        failure_excerpt='assert failed: secret-password',
        structured_logfiles=(),
    )

    assert bundle_path is not None
    assert bundle_path.exists()
    manifest = read_manifest(bundle_path)
    assert manifest['schema_version'] == artifact.SCHEMA_VERSION
    assert manifest['session']['id'] == 'session-test'
    assert manifest['session']['hostname'] == 'testhost'
    test_entry = manifest['test']
    assert test_entry['nodeid'] == 'tests/test_svc.py::test_request'
    assert test_entry['outcome'] == 'failed'
    assert test_entry['trace_id'] == 'trace-xyz'
    assert artifact.REDACTED_VALUE in test_entry['failure_excerpt']
    assert 'secret-password' not in test_entry['failure_excerpt']

    service = manifest['services'][0]
    assert service['scope'] == 'my.service'
    assert service['daemon_instance_id'] == 'di-1'
    assert service['pid'] == 4242
    assert service['command'] == [
        'svc-binary',
        f'--api-token={artifact.REDACTED_VALUE}',
    ]
    assert service['command_line'] == (
        f'svc-binary --api-token={artifact.REDACTED_VALUE}'
    )
    assert service['base_command'] == ['launcher']
    assert service['env']['PASSWORD'] == artifact.REDACTED_VALUE
    assert service['env']['NODE'] == '1'
    assert service['env']['SERVICE_PORT'] == '8080'
    assert {'name': 'SERVICE_PORT', 'value': '8080'} in service['ports']
    assert any(item['name'] == 'ping_url' for item in service['ports'])
    assert service['exit_code'] == -signal.SIGABRT
    assert 'SIGABRT' in service['exit_reason']
    assert service['mode'] == 'spawn'

    service_dir = bundle_path / 'services' / capture.id
    assert (service_dir / 'startup.json').exists()
    stdout = (bundle_path / service['stdout']['path']).read_bytes()
    assert stdout == b'boot line\nrequest handled\n'
    stderr = (bundle_path / service['stderr']['path']).read_bytes()
    assert stderr == b'warn line\n'

    events = [entry['event'] for entry in read_timeline(bundle_path)]
    assert 'service_spawn' in events
    assert 'service_ready' in events
    assert 'service_bound' in events
    assert 'service_exit' in events

    # Manifest is written atomically, no temp files left behind.
    assert not list(bundle_path.glob(f'.{artifact.MANIFEST_NAME}.*.tmp'))


def test_success_deletes_bundle_in_failure_mode(tmp_path):
    store = make_store(tmp_path)
    store.begin_test('tests/test_svc.py::test_ok')
    capture, process, stdout_handler, _ = spawn_service(store)
    stdout_handler(b'noise\n')
    process.returncode = 0
    capture.finish()

    bundle_path = store.current.path
    retained = store.end_current(
        outcome='passed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    assert retained is None
    assert not bundle_path.exists()


def test_always_mode_retains_successful_bundle(tmp_path):
    store = make_store(tmp_path, mode=artifact.MODE_ALWAYS)
    store.begin_test('tests/test_svc.py::test_ok')
    spawn_service(store)
    bundle_path = store.current.path

    retained = store.end_current(
        outcome='passed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    assert retained == bundle_path
    assert (bundle_path / artifact.MANIFEST_NAME).exists()


def test_zip_format_bundles_bundle(tmp_path):
    store = make_store(tmp_path, bundle_format=artifact.FORMAT_ZIP)
    store.begin_test('tests/test_svc.py::test_fail')
    capture, _process, stdout_handler, _ = spawn_service(store)
    stdout_handler(b'payload\n')

    retained = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    assert retained is not None
    assert retained.suffix == '.zip'
    assert not retained.with_suffix('').exists()
    with zipfile.ZipFile(retained) as archive:
        names = archive.namelist()
        assert any(name.endswith(artifact.MANIFEST_NAME) for name in names)
        dirname = retained.name[: -len('.zip')]
        stdout_name = f'{dirname}/services/{capture.id}/stdout.log'
        assert archive.read(stdout_name) == b'payload\n'


def test_per_stream_capacity_truncation(tmp_path):
    store = make_store(tmp_path, max_bytes=10)
    store.begin_test('tests/test_svc.py::test_fail')
    capture, _proc, stdout_handler, _ = spawn_service(store)
    stdout_handler(b'0123456789ABCDEFGHIJ\n')

    bundle_path = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    manifest = read_manifest(bundle_path)
    stdout_info = manifest['services'][0]['stdout']
    assert stdout_info['bytes'] == 10
    assert stdout_info['dropped_bytes'] == 11
    content = (bundle_path / stdout_info['path']).read_bytes()
    assert content.startswith(b'0123456789')
    assert b'dropped 11 bytes' in content


def test_session_quota_drops_extra_output(tmp_path):
    store = make_store(tmp_path, max_bytes=0, session_max_bytes=5)
    store.begin_test('tests/test_svc.py::test_fail')
    capture, _proc, stdout_handler, _ = spawn_service(store)
    stdout_handler(b'0123456789')

    bundle_path = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    manifest = read_manifest(bundle_path)
    stdout_info = manifest['services'][0]['stdout']
    assert stdout_info['bytes'] == 5
    assert stdout_info['dropped_bytes'] == 5
    assert (bundle_path / stdout_info['path']).read_bytes().startswith(b'01234')


def test_pending_buffer_keeps_bounded_tail(tmp_path):
    # Service starts outside of an active test: output is buffered until
    # the capture is bound to a freshly started bundle.
    store = make_store(tmp_path)
    assert store.current is None
    capture = store.start_capture({'command': ('svc',), 'mode': 'spawn'})
    total = artifact._PENDING_LIMIT_BYTES // 10 + 100
    for index in range(total):
        capture.write('stdout', f'line-{index:06d}\n'.encode())

    store.begin_test('tests/test_svc.py::test_fail')
    capture.attach(FakeProcess(pid=7, returncode=0))

    bundle_path = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    manifest = read_manifest(bundle_path)
    stdout_info = manifest['services'][0]['stdout']
    assert stdout_info['bytes'] <= artifact._PENDING_LIMIT_BYTES
    # Oldest lines were dropped, tail is retained.
    content = (bundle_path / stdout_info['path']).read_bytes()
    assert f'line-{total - 1:06d}\n'.encode() in content
    assert b'line-000000' not in content


def test_exit_reason_status_code(tmp_path):
    store = make_store(tmp_path)
    store.begin_test('tests/test_svc.py::test_fail')
    capture, _process, _out, _err = spawn_service(
        store, FakeProcess(returncode=1)
    )
    capture.finish()
    bundle_path = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    service = read_manifest(bundle_path)['services'][0]
    assert service['exit_code'] == 1
    assert service['exit_reason'] == 'Service exited with status code 1'


def test_still_running_service_has_no_exit_code(tmp_path):
    store = make_store(tmp_path)
    store.begin_test('tests/test_svc.py::test_fail')
    spawn_service(store, FakeProcess(returncode=None))
    bundle_path = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    service = read_manifest(bundle_path)['services'][0]
    assert service['exit_code'] is None
    assert service['exit_reason'] is None


def test_structured_log_slice(tmp_path):
    logfile = tmp_path / 'service.log'
    logfile.write_bytes(b'old log line\n')

    store = make_store(tmp_path)
    bundle = store.begin_test('tests/test_svc.py::test_fail')
    bundle.snapshot_log_positions([(logfile, 'my-service log')])
    with logfile.open('ab') as fp:
        fp.write(b'new failure line\n')
    spawn_service(store)

    bundle_path = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=[(logfile, 'my-service log')],
    )
    manifest = read_manifest(bundle_path)
    assert len(manifest['service_logs']) == 1
    entry = manifest['service_logs'][0]
    assert entry['title'] == 'my-service log'
    assert entry['truncated'] is False
    content = (bundle_path / entry['path']).read_bytes()
    assert content == b'new failure line\n'
    assert b'old log line' not in content


def test_structured_log_slice_truncated(tmp_path):
    logfile = tmp_path / 'service.log'
    store = make_store(tmp_path, max_bytes=4)
    bundle = store.begin_test('tests/test_svc.py::test_fail')
    bundle.snapshot_log_positions([(logfile, 'svc')])
    logfile.write_bytes(b'0123456789')
    spawn_service(store)

    bundle_path = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=[(logfile, 'svc')],
    )
    entry = read_manifest(bundle_path)['service_logs'][0]
    assert entry['bytes'] == 4
    assert entry['truncated'] is True


def test_prune_old_sessions(tmp_path):
    root = tmp_path / 'artifacts'
    root.mkdir()
    for name in ('a', 'b', 'c', 'd', 'e'):
        session_dir = root / f'session-{name}'
        session_dir.mkdir()
        (session_dir / 'marker').write_text(name)
        # Make creation order deterministic.
        os.utime(session_dir, (1000 + ord(name), 1000 + ord(name)))

    # ``retain`` counts the current session as well, so with retain=3
    # only the 2 newest other sessions survive.
    artifact.ArtifactStore.prune_old_sessions(
        root, retain=3, current_session_id='new'
    )
    surviving = sorted(path.name for path in root.glob('session-*'))
    assert surviving == ['session-d', 'session-e']


def test_reused_service_bound_to_each_bundle(tmp_path):
    """Daemon reused across tests appears in every bound bundle."""
    store = make_store(tmp_path)
    process = FakeProcess(returncode=None)

    store.begin_test('tests/test_svc.py::test_one')
    capture1, _, out1, _ = spawn_service(
        store, process, daemon_instance_id='di'
    )
    out1(b'one\n')
    first = store.end_current(
        outcome='passed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    assert first is None  # deleted on success

    store.begin_test('tests/test_svc.py::test_two')
    # Simulate daemon store reuse: same process requested again.
    store.bind_process(
        process, scope_name='my.service', daemon_instance_id='di'
    )
    capture1.write('stdout', b'two\n')
    second = store.end_current(
        outcome='failed',
        trace_id=None,
        failure_excerpt=None,
        structured_logfiles=(),
    )
    services = read_manifest(second)['services']
    assert [item['instance_id'] for item in services] == [capture1.id]
    stream = services[0]['stdout']
    assert (second / stream['path']).read_bytes() == b'two\n'


def test_null_store_is_noop(tmp_path):
    store = artifact.NullStore()
    assert store.enabled is False
    capture = store.start_capture({})
    handler, stderr_handler = capture.wrap_handlers(
        lambda line: None, lambda line: None
    )
    assert callable(handler) and callable(stderr_handler)
    capture.attach(FakeProcess())
    capture.finish()
    store.bind_process(FakeProcess(), scope_name='x')
    assert store.end_current(outcome='failed') is None
    assert not (tmp_path / 'artifacts').exists()
