import json

import pytest

from testsuite._internal.capabilities import (
    SCHEMA_VERSION,
    CapabilityRegistry,
    run_preflight,
)
from testsuite._internal.capabilities import (
    DiagnosticCode as Code,
)


def make_report(*items):
    registry = CapabilityRegistry()
    for item in items:
        registry.register(item)
    return run_preflight(registry)


def spec(
    name,
    *,
    provider=None,
    version='1.0',
    requires=(),
    conflicts=(),
    scope='function',
    teardown='best-effort',
):
    return {
        'name': name,
        'provider': provider or f'p.{name}',
        'version': version,
        'requires': requires,
        'conflicts': conflicts,
        'fixture_scope': scope,
        'teardown': teardown,
    }


def codes(report):
    return [diagnostic.code for diagnostic in report.diagnostics]


def find(report, code, capability=None):
    matches = [
        diagnostic
        for diagnostic in report.diagnostics
        if diagnostic.code == code
        and (capability is None or diagnostic.capability == capability)
    ]
    assert matches, f'diagnostic {code} not found'
    return matches[0]


def test_healthy_graph_report():
    report = make_report(
        spec('asyncio', scope='session', teardown='required'),
        spec(
            'mockserver',
            version='2.0',
            requires=[('asyncio', '>=1.0,<3')],
            scope='session',
        ),
    )
    assert report.status == 'ok'
    assert report.errors == ()
    assert report.warnings == ()
    data = report.to_dict()
    assert data['schema_version'] == SCHEMA_VERSION
    assert data['summary'] == {
        'capabilities': 2,
        'providers': 2,
        'errors': 0,
        'warnings': 0,
    }
    assert data['graph']['order'] == ['asyncio', 'mockserver']
    # Required teardown reverses setup order; 'none' policies are excluded.
    assert report.teardown_order == ('mockserver', 'asyncio')


def test_report_is_json_serializable_and_deterministic():
    report = make_report(
        spec('base', scope='session', teardown='required'),
        spec(
            'app',
            requires=[
                ('base', '>=1.0'),
                {'name': 'extra', 'optional': True},
            ],
            conflicts=['absent'],
        ),
    )
    first = json.dumps(report.to_dict(), sort_keys=True)
    second = json.dumps(report.to_dict(), sort_keys=True)
    assert first == second


def test_duplicate_providers():
    report = make_report(
        spec('alpha', provider='p.first'),
        spec('alpha', provider='p.second'),
    )
    diagnostic = find(report, Code.DUPLICATE_PROVIDER, 'alpha')
    assert diagnostic.severity == 'error'
    assert diagnostic.details['providers'] == ['p.first', 'p.second']
    assert report.status == 'error'


def test_duplicate_registration_same_provider():
    report = make_report(
        spec('alpha', provider='p.same', version='1.0'),
        spec('alpha', provider='p.same', version='2.0'),
    )
    diagnostic = find(report, Code.DUPLICATE_REGISTRATION, 'alpha')
    assert diagnostic.severity == 'error'
    assert diagnostic.details['count'] == 2


def test_missing_required_dependency_is_error():
    report = make_report(spec('alpha', requires=[('missing',)]))
    diagnostic = find(report, Code.MISSING_DEPENDENCY, 'alpha')
    assert diagnostic.severity == 'error'
    assert diagnostic.details['optional'] is False
    assert report.status == 'error'


def test_missing_optional_dependency_is_warning():
    report = make_report(
        spec(
            'alpha',
            requires=[{'name': 'extras', 'optional': True}],
        ),
    )
    diagnostic = find(report, Code.MISSING_OPTIONAL_DEPENDENCY, 'alpha')
    assert diagnostic.severity == 'warning'
    assert report.status == 'ok'


def test_version_incompatible():
    report = make_report(
        spec('base', version='1.5'),
        spec('app', requires=[('base', '>=2.0')]),
    )
    diagnostic = find(report, Code.VERSION_INCOMPATIBLE, 'app')
    assert diagnostic.severity == 'error'
    assert diagnostic.details['provided_version'] == '1.5'
    assert diagnostic.details['specifier'] == '>=2.0'


@pytest.mark.parametrize(
    ('required_version', 'has_diagnostic'),
    [('>=2.0', True), ('>=1.0', False), ('>1.5,<2.0', True)],
)
def test_version_specifier_matching(required_version, has_diagnostic):
    report = make_report(
        spec('base', version='1.5'),
        spec('app', requires=[('base', required_version)]),
    )
    assert (Code.VERSION_INCOMPATIBLE in codes(report)) is has_diagnostic


def test_conflicting_capabilities_reported_once_per_pair():
    report = make_report(
        spec('alpha', conflicts=['beta']),
        spec('beta', conflicts=['alpha']),
    )
    assert codes(report).count(Code.CONFLICTING_CAPABILITY) == 1
    diagnostic = find(report, Code.CONFLICTING_CAPABILITY)
    assert diagnostic.details['with'] in {'alpha', 'beta'}


def test_circular_dependency_diagnostic():
    report = make_report(
        spec('alpha', requires=[('beta',)]),
        spec('beta', requires=[('alpha',)]),
    )
    diagnostic = find(report, Code.CIRCULAR_DEPENDENCY)
    assert diagnostic.severity == 'error'
    assert set(diagnostic.details['cycle']) == {'alpha', 'beta'}
    assert 'alpha' in diagnostic.message
    assert 'beta' in diagnostic.message


def test_scope_too_broad_when_dependency_is_narrower():
    report = make_report(
        spec('base', scope='function'),
        spec('app', requires=[('base',)], scope='session'),
    )
    diagnostic = find(report, Code.SCOPE_TOO_BROAD, 'app')
    assert diagnostic.severity == 'error'
    assert diagnostic.details['scope'] == 'session'
    assert diagnostic.details['dependency_scope'] == 'function'


def test_narrow_scope_depending_on_broad_scope_is_allowed():
    report = make_report(
        spec('base', scope='session'),
        spec('app', requires=[('base',)], scope='function'),
    )
    assert Code.SCOPE_TOO_BROAD not in codes(report)


def test_invalid_declaration_diagnostic():
    registry = CapabilityRegistry()
    registry.register(name='broken', provider='p.broken', teardown='maybe')
    registry.register(name='healthy', provider='p.healthy')
    report = run_preflight(registry)
    diagnostic = find(report, Code.INVALID_DECLARATION, 'broken')
    assert diagnostic.severity == 'error'
    assert diagnostic.provider == 'p.broken'
    assert diagnostic.details['raw']['name'] == 'broken'
    assert report.status == 'error'
    assert [item['name'] for item in report.capabilities] == ['healthy']


def test_diagnostics_sorted_deterministically():
    report_a = make_report(
        spec('zzz', requires=[('missing-z',)]),
        spec('aaa', requires=[('missing-a',)]),
    )
    report_b = make_report(
        spec('aaa', requires=[('missing-a',)]),
        spec('zzz', requires=[('missing-z',)]),
    )
    assert [item.to_dict() for item in report_a.diagnostics] == [
        item.to_dict() for item in report_b.diagnostics
    ]
    assert codes(report_a) == [
        Code.MISSING_DEPENDENCY,
        Code.MISSING_DEPENDENCY,
    ]


def test_teardown_order_excludes_none_policy():
    report = make_report(
        spec('base', scope='session', teardown='required'),
        spec('svc', requires=[('base',)], scope='session'),
        spec('trace', teardown='none'),
    )
    assert report.teardown_order == ('svc', 'base')
    assert 'trace' not in report.teardown_order
