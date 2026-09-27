import pytest

from testsuite._internal.capabilities import (
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilitySpec,
    coerce_spec,
)


def make_registry(*items):
    registry = CapabilityRegistry()
    for item in items:
        registry.register(item)
    return registry


def test_register_valid_keywords():
    registry = CapabilityRegistry()
    registry.register(
        name='mockserver',
        provider='testsuite.mockserver',
        version='1.2',
        requires=[('asyncio', '>=1.0')],
        conflicts=('tcp_mockserver',),
        fixture_scope='session',
        teardown='required',
    )
    assert len(registry.specs) == 1
    spec = registry.specs[0]
    assert spec.name == 'mockserver'
    assert spec.requires == (CapabilityRequirement('asyncio', '>=1.0'),)
    assert spec.conflicts == ('tcp_mockserver',)
    assert registry.errors == []


def test_register_accepts_spec_instance_and_mapping():
    spec = CapabilitySpec(name='a', provider='p.a', version='1.0')
    registry = CapabilityRegistry()
    registry.register(spec)
    registry.register({'name': 'b', 'provider': 'p.b'})
    assert [item.name for item in registry.specs] == ['a', 'b']


def test_register_rejects_spec_together_with_kwargs():
    registry = CapabilityRegistry()
    with pytest.raises(TypeError):
        registry.register(
            CapabilitySpec(name='a', provider='p.a'),
            name='b',
        )


def test_register_rejects_unsupported_spec_type():
    registry = CapabilityRegistry()
    with pytest.raises(TypeError):
        registry.register(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'fields',
    [
        {'provider': 'p.a'},  # missing name
        {'name': 'a'},  # missing provider
        {'name': '', 'provider': 'p.a'},
        {'name': 'a', 'provider': ''},
        {'name': 1, 'provider': 'p.a'},
        {'name': 'a', 'provider': 'p.a', 'version': 1},
        {'name': 'a', 'provider': 'p.a', 'version': 'not-a-version'},
        {'name': 'a', 'provider': 'p.a', 'fixture_scope': 'global'},
        {'name': 'a', 'provider': 'p.a', 'teardown': 'maybe'},
        {'name': 'a', 'provider': 'p.a', 'unknown': 'x'},
        {'name': 'a', 'provider': 'p.a', 'requires': 'asyncio'},
        {'name': 'a', 'provider': 'p.a', 'requires': [('x', '??')]},
        {'name': 'a', 'provider': 'p.a', 'requires': [('x',), ('x',)]},
        {'name': 'a', 'provider': 'p.a', 'conflicts': 'b'},
        {'name': 'a', 'provider': 'p.a', 'conflicts': ['']},
        {'name': 'a', 'provider': 'p.a', 'conflicts': ['b', 'b']},
        {'name': 'a', 'provider': 'p.a', 'conflicts': ['a']},
        {'name': 'a', 'provider': 'p.a', 'requires': [()]},
        {
            'name': 'a',
            'provider': 'p.a',
            'requires': [('x', '>=1', False, True)],
        },
    ],
)
def test_coerce_spec_rejects_invalid_declarations(fields):
    result = coerce_spec(fields)
    assert not isinstance(result, CapabilitySpec)
    assert result.message


def test_requirement_normalization_forms():
    registry = make_registry(
        {
            'name': 'a',
            'provider': 'p.a',
            'requires': [
                'b',
                ('c', '>=1.0'),
                ('d', '>=2.0', True),
                {'name': 'e', 'specifier': '<3', 'optional': False},
                CapabilityRequirement('f', '~=1.2', True),
            ],
        },
    )
    requires = registry.specs[0].requires
    assert requires == (
        CapabilityRequirement('b'),
        CapabilityRequirement('c', '>=1.0'),
        CapabilityRequirement('d', '>=2.0', True),
        CapabilityRequirement('e', '<3'),
        CapabilityRequirement('f', '~=1.2', True),
    )


def test_invalid_registration_collected_as_error():
    registry = CapabilityRegistry()
    registry.register(name='a', provider='p.a', fixture_scope='global')
    registry.register(name='b', provider='p.b')
    assert len(registry.specs) == 1
    assert registry.specs[0].name == 'b'
    assert len(registry.errors) == 1
    assert registry.errors[0].name == 'a'
    assert 'fixture_scope' in registry.errors[0].message
    assert registry.errors[0].raw['provider'] == 'p.a'


def test_default_values():
    spec = coerce_spec({'name': 'a', 'provider': 'p.a'})
    assert spec.version == '0'
    assert spec.requires == ()
    assert spec.conflicts == ()
    assert spec.fixture_scope == 'function'
    assert spec.teardown == 'best-effort'
