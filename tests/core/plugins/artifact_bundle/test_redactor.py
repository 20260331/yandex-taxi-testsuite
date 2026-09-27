import pytest

from testsuite.daemons import artifact


def test_redact_mapping_defaults():
    redactor = artifact.Redactor()
    result = redactor.redact_mapping(
        {
            'PASSWORD': 'p@ss',
            'APP_TOKEN': 'tok',
            'X_SECRET_VALUE': 's',
            'normal_key': 'value',
        }
    )
    assert result == {
        'PASSWORD': artifact.REDACTED_VALUE,
        'APP_TOKEN': artifact.REDACTED_VALUE,
        'X_SECRET_VALUE': artifact.REDACTED_VALUE,
        'normal_key': 'value',
    }


def test_redact_mapping_extra_patterns():
    redactor = artifact.Redactor(['tenant'])
    result = redactor.redact_mapping({'TENANT_ID': 't1', 'KEEP': 'v'})
    assert result == {
        'TENANT_ID': artifact.REDACTED_VALUE,
        'KEEP': 'v',
    }


def test_redact_args_key_value():
    redactor = artifact.Redactor()
    assert redactor.redact_args(
        ['serve', '--api-token=abc', '--password=qwerty', '--port=80']
    ) == [
        'serve',
        f'--api-token={artifact.REDACTED_VALUE}',
        f'--password={artifact.REDACTED_VALUE}',
        '--port=80',
    ]


def test_ports_from_env():
    redactor = artifact.Redactor()
    ports = redactor.ports_from_env(
        {'SERVICE_PORT': '8080', 'MONGO_PORT': '27017', 'HOST': 'localhost'}
    )
    names = sorted(item['name'] for item in ports)
    assert names == ['MONGO_PORT', 'SERVICE_PORT']


def test_secret_values_and_redact_text():
    redactor = artifact.Redactor()
    secrets = redactor.secret_values({'PASSWORD': 'hunter2'})
    assert secrets == ['hunter2']
    text = 'failure: bad credentials hunter2 were used'
    redacted = redactor.redact_text(text, secrets)
    assert 'hunter2' not in redacted
    assert artifact.REDACTED_VALUE in redacted


@pytest.mark.parametrize(
    'value,expected',
    [
        (0, 0),
        (1024, 1024),
        ('1024', 1024),
        ('10K', 10 * 1024),
        ('10M', 10 * 1024**2),
        ('1.5GiB', int(1.5 * 1024**3)),
        ('500MB', 500 * 1000**2),
    ],
)
def test_parse_size(value, expected):
    assert artifact.parse_size(value) == expected


@pytest.mark.parametrize('value', ['abc', '10X'])
def test_parse_size_invalid(value):
    with pytest.raises(ValueError):
        artifact.parse_size(value)


def test_slugify():
    assert (
        artifact.slugify('tests/x.py::test_a[param-1]')
        == 'tests.x.py.test_a_param-1'
    )
