import json
import os
import pathlib
import site

import pytest

PREFLIGHT_PLUGIN = 'testsuite.plugins.capabilities'
REPO_ROOT = pathlib.Path(__file__).parents[4]
# Captured before pytester redirects HOME to its temporary directory: user
# site-packages (and editable installs such as pytest) must stay visible in
# subprocess runs via PYTHONUSERBASE.
USER_BASE = site.getuserbase()

PASS_TEST = """
def test_ok():
    assert True
"""

HEALTHY_CONFTEST = """
def pytest_capability_register(register_capability):
    register_capability(
        name='alpha',
        provider='prov.alpha',
        version='1.0',
        fixture_scope='session',
        teardown='required',
    )
    register_capability(
        name='beta',
        provider='prov.beta',
        version='2.0',
        requires=[('alpha', '>=1.0,<3')],
        fixture_scope='session',
        teardown='best-effort',
    )
    register_capability(
        name='gamma',
        provider='prov.gamma',
        version='1.0',
        requires=[{'name': 'extras', 'optional': True}],
        fixture_scope='function',
        teardown='none',
    )
"""

DUPLICATE_CONFTEST = """
def pytest_capability_register(register_capability):
    register_capability(name='dup', provider='prov.first')
    register_capability(name='dup', provider='prov.second')
"""

ERROR_CASES = [
    (
        'duplicate_provider',
        """
def pytest_capability_register(register_capability):
    register_capability(name='dup', provider='prov.first')
    register_capability(name='dup', provider='prov.second')
""",
    ),
    (
        'version_incompatible',
        """
def pytest_capability_register(register_capability):
    register_capability(name='base', provider='prov.base', version='1.0')
    register_capability(
        name='app',
        provider='prov.app',
        requires=[('base', '>=2.0')],
    )
""",
    ),
    (
        'missing_dependency',
        """
def pytest_capability_register(register_capability):
    register_capability(
        name='alpha',
        provider='prov.alpha',
        requires=[('absent', '>=1.0')],
    )
""",
    ),
    (
        'circular_dependency',
        """
def pytest_capability_register(register_capability):
    register_capability(name='alpha', provider='prov.alpha',
                        requires=[('beta',)])
    register_capability(name='beta', provider='prov.beta',
                        requires=[('alpha',)])
""",
    ),
    (
        'scope_too_broad',
        """
def pytest_capability_register(register_capability):
    register_capability(name='base', provider='prov.base',
                        fixture_scope='function')
    register_capability(name='app', provider='prov.app',
                        requires=[('base',)], fixture_scope='session')
""",
    ),
    (
        'invalid_declaration',
        """
def pytest_capability_register(register_capability):
    register_capability(name='broken', provider='prov.broken',
                        fixture_scope='global')
""",
    ),
    (
        'conflicting_capability',
        """
def pytest_capability_register(register_capability):
    register_capability(name='alpha', provider='prov.alpha',
                        conflicts=['beta'])
    register_capability(name='beta', provider='prov.beta')
""",
    ),
]


@pytest.fixture(autouse=True)
def _repo_on_pythonpath(monkeypatch):
    # Subprocess pytester runs from a temporary directory and testsuite may
    # not be installed into site-packages.
    old_pythonpath = os.environ.get('PYTHONPATH', '')
    monkeypatch.setenv(
        'PYTHONPATH',
        str(REPO_ROOT)
        + (os.pathsep + old_pythonpath if old_pythonpath else ''),
    )
    monkeypatch.setenv('PYTHONUSERBASE', USER_BASE)


@pytest.fixture
def healthy_project(pytester):
    pytester.makepyfile(conftest=HEALTHY_CONFTEST)
    pytester.makepyfile(test_ok=PASS_TEST)
    return pytester


def read_report(pytester, name='report.json'):
    return json.loads((pytester.path / name).read_text())


def test_preflight_disabled_by_default(pytester):
    pytester.makepyfile(conftest=DUPLICATE_CONFTEST)
    pytester.makepyfile(test_ok=PASS_TEST)
    result = pytester.runpytest_subprocess('-p', PREFLIGHT_PLUGIN)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*1 passed*'])
    assert 'testsuite preflight:' not in result.stdout.str()
    assert not (pytester.path / 'report.json').exists()


def test_preflight_enabled_healthy(healthy_project):
    result = healthy_project.runpytest_subprocess(
        '-p',
        PREFLIGHT_PLUGIN,
        '--testsuite-preflight',
        '--testsuite-preflight-report',
        'report.json',
    )
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            '*testsuite preflight: ok*',
            '*1 passed*',
        ],
    )
    data = read_report(healthy_project)
    assert data['schema_version'] == '1.0'
    assert data['status'] == 'ok'
    assert [item['name'] for item in data['capabilities']] == [
        'alpha',
        'beta',
        'gamma',
    ]
    assert data['summary']['warnings'] == 1
    assert data['diagnostics'][0]['code'] == ('missing_optional_dependency')

    order = data['graph']['order']
    assert order.index('alpha') < order.index('beta')
    # gamma has teardown policy 'none' and must not appear.
    assert data['teardown_order'] == ['beta', 'alpha']


def test_report_path_implies_preflight(healthy_project):
    result = healthy_project.runpytest_subprocess(
        '-p',
        PREFLIGHT_PLUGIN,
        '--testsuite-preflight-report',
        'report.json',
    )
    assert result.ret == 0
    data = read_report(healthy_project)
    assert data['status'] == 'ok'


@pytest.mark.parametrize(
    ('code', 'conftest_body'),
    ERROR_CASES,
    ids=[case[0] for case in ERROR_CASES],
)
def test_preflight_error_cases(pytester, code, conftest_body):
    pytester.makepyfile(conftest=conftest_body)
    pytester.makepyfile(test_ok=PASS_TEST)
    result = pytester.runpytest_subprocess(
        '-p',
        PREFLIGHT_PLUGIN,
        '--testsuite-preflight',
        '--testsuite-preflight-report',
        'report.json',
    )
    assert result.ret == 4
    data = read_report(pytester)
    assert data['status'] == 'error'
    assert code in [item['code'] for item in data['diagnostics']]


def test_preflight_report_is_deterministic(healthy_project):
    first = healthy_project.runpytest_subprocess(
        '-p',
        PREFLIGHT_PLUGIN,
        '--testsuite-preflight-report',
        'report1.json',
    )
    assert first.ret == 0
    second = healthy_project.runpytest_subprocess(
        '-p',
        PREFLIGHT_PLUGIN,
        '--testsuite-preflight-report',
        'report2.json',
    )
    assert second.ret == 0
    first_data = read_report(healthy_project, 'report1.json')
    second_data = read_report(healthy_project, 'report2.json')
    assert first_data == second_data


def test_builtin_capabilities_are_healthy(pytester):
    pytester.makepyfile(test_ok=PASS_TEST)
    result = pytester.runpytest_subprocess(
        '-p',
        'testsuite.pytest_plugin',
        '--no-env',
        '--testsuite-preflight',
        '--testsuite-preflight-report',
        'report.json',
    )
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*1 passed*'])
    data = read_report(pytester)
    assert data['status'] == 'ok'
    names = {item['name'] for item in data['capabilities']}
    assert {
        'asyncio',
        'daemons',
        'environment',
        'mockserver',
        'tracing',
    } <= names
    # Database plugins are not loaded by default.
    assert 'clickhouse' not in names
