import json
import os
import pathlib
import sys
import uuid

import pytest

DUMMY_DAEMON = (
    pathlib.Path(__file__).parents[2]
    / 'daemons'
    / 'daemons'
    / 'dummy_daemon.py'
)

CONFTEST = """
pytest_plugins = ['testsuite.pytest_plugin']
"""

TEST_FILE = f"""
import asyncio
import sys

import pytest

DUMMY = {str(DUMMY_DAEMON)!r}


@pytest.fixture
async def dummy_scope(
    register_daemon_scope, service_spawner_factory
):
    async def health_check(*, session, process):
        await asyncio.sleep(0.1)
        return True

    factory = service_spawner_factory(
        [sys.executable, DUMMY],
        env={{'PASSWORD': 'super-secret-value'}},
        health_check=health_check,
    )
    async with register_daemon_scope(
        'dummy.scope', spawn=factory
    ) as scope:
        yield scope


async def test_one_ok(ensure_daemon_started, dummy_scope):
    await ensure_daemon_started(dummy_scope)


async def test_two_fail(
    ensure_daemon_started, dummy_scope, testsuite_trace_id
):
    await ensure_daemon_started(dummy_scope)
    assert testsuite_trace_id
    raise AssertionError('boom super-secret-value')
"""


@pytest.fixture
def isolated_session_id(monkeypatch):
    # pytester redirects HOME to its temp directory, which hides the
    # interpreter user-site (where pytest and testsuite dependencies live).
    # Replicate the outer interpreter sys.path through PYTHONPATH.
    python_path = os.pathsep.join(
        path for path in sys.path if path and os.path.isdir(path)
    )
    monkeypatch.setenv('PYTHONPATH', python_path)
    session_id = f'e2e-{uuid.uuid4().hex}'
    monkeypatch.setenv('TESTSUITE_ARTIFACT_SESSION_ID', session_id)
    return session_id


def _write_project(pytester):
    pytester.makepyfile(conftest=CONFTEST)
    pytester.makepyfile(test_svc=TEST_FILE)


def _bundles(artifacts_dir):
    return [
        path
        for path in artifacts_dir.glob('session-*/main/bundles/*')
        if path.is_dir()
    ]


def test_failure_retained_success_deleted(pytester, isolated_session_id):
    _write_project(pytester)
    artifacts_dir = pytester.path / 'artifacts'

    result = pytester.runpytest_subprocess(
        '-p',
        'no:cacheprovider',
        '--service-artifacts-dir',
        str(artifacts_dir),
    )

    result.assert_outcomes(passed=1, failed=1)
    bundles = _bundles(artifacts_dir)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert 'test_two_fail' in bundle.name

    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['test']['outcome'] == 'failed'
    assert manifest['test']['trace_id']
    service = manifest['services'][0]
    assert service['scope'] == 'dummy.scope'
    assert service['pid'] > 0
    assert service['env']['PASSWORD'] == '***redacted***'
    stdout = bundle / service['stdout']['path']
    assert stdout.read_bytes() == b'ready\n'

    timeline = (bundle / 'timeline.jsonl').read_text()
    assert 'service_spawn' in timeline
    assert 'service_ready' in timeline
    assert 'service_exit' in timeline

    # Raw secret must not leak anywhere in the retained bundle.
    for path in bundle.rglob('*'):
        if path.is_file():
            assert b'super-secret-value' not in path.read_bytes(), path

    # Passing test bundle was deleted.
    assert not list(bundle.parent.glob('*test_one_ok*'))


def test_mode_off_creates_nothing(pytester, isolated_session_id):
    _write_project(pytester)
    artifacts_dir = pytester.path / 'artifacts-off'

    result = pytester.runpytest_subprocess(
        '-p',
        'no:cacheprovider',
        '--service-artifacts-dir',
        str(artifacts_dir),
        '--service-artifacts',
        'off',
    )

    result.assert_outcomes(passed=1, failed=1)
    assert not artifacts_dir.exists()


def test_zip_format(pytester, isolated_session_id):
    _write_project(pytester)
    artifacts_dir = pytester.path / 'artifacts-zip'

    result = pytester.runpytest_subprocess(
        '-p',
        'no:cacheprovider',
        '--service-artifacts-dir',
        str(artifacts_dir),
        '--service-artifacts-format',
        'zip',
    )

    result.assert_outcomes(passed=1, failed=1)
    archives = list(artifacts_dir.glob('session-*/main/bundles/*.zip'))
    assert len(archives) == 1
    assert not _bundles(artifacts_dir)


def test_always_mode_via_ini(pytester, isolated_session_id):
    pytester.makeini(
        """
        [pytest]
        service_artifacts = always
        """
    )
    _write_project(pytester)
    artifacts_dir = pytester.path / 'artifacts-always'

    result = pytester.runpytest_subprocess(
        '-p',
        'no:cacheprovider',
        '--service-artifacts-dir',
        str(artifacts_dir),
    )

    result.assert_outcomes(passed=1, failed=1)
    bundles = _bundles(artifacts_dir)
    assert len(bundles) == 2
    outcomes = {
        json.loads((bundle / 'manifest.json').read_text())['test']['outcome']
        for bundle in bundles
    }
    assert outcomes == {'passed', 'failed'}
