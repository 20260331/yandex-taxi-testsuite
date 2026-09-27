"""Capability registry.

A *capability* describes a piece of functionality exposed by a pytest plugin
(e.g. mockserver, databases, daemons). Plugins register capabilities through
the ``pytest_capability_register`` hook, while this module itself stays free of
pytest dependencies so that it can be unit tested and reused.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

#: Allowed pytest fixture scopes, ordered from broadest to narrowest.
VALID_FIXTURE_SCOPES: tuple[str, ...] = (
    'session',
    'package',
    'module',
    'class',
    'function',
)

#: Teardown policies supported by capability providers.
VALID_TEARDOWN_POLICIES: tuple[str, ...] = (
    'required',
    'best-effort',
    'none',
)

#: Broadest scope has the smallest rank, i.e. session(0) ... function(4).
SCOPE_RANK: dict[str, int] = {
    scope: rank for rank, scope in enumerate(VALID_FIXTURE_SCOPES)
}


@dataclasses.dataclass(frozen=True)
class CapabilityRequirement:
    """Dependency of a capability on another capability.

    :param name: required capability name.
    :param specifier: PEP 440 version specifier, empty string allows any.
    :param optional: missing provider is a warning instead of an error.
    """

    name: str
    specifier: str = ''
    optional: bool = False


@dataclasses.dataclass(frozen=True)
class CapabilitySpec:
    """Static capability declaration published by a plugin."""

    #: Capability name, unique per provider graph.
    name: str
    #: Identifier of the plugin providing the capability.
    provider: str
    #: PEP 440 version of the capability contract.
    version: str = '0'
    #: Capabilities this one depends on.
    requires: tuple[CapabilityRequirement, ...] = dataclasses.field(
        default_factory=tuple,
    )
    #: Capabilities that must not be present together with this one.
    conflicts: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    #: Primary pytest fixture scope of the capability state.
    fixture_scope: str = 'function'
    #: Teardown policy, see :data:`VALID_TEARDOWN_POLICIES`.
    teardown: str = 'best-effort'


@dataclasses.dataclass(frozen=True)
class RegistrationError:
    """Rejected capability declaration, reported as a diagnostic."""

    provider: str | None
    name: str | None
    message: str
    raw: dict[str, Any]


def _coerce_requirement(value: Any) -> CapabilityRequirement:
    if isinstance(value, CapabilityRequirement):
        requirement = value
    elif isinstance(value, str):
        requirement = CapabilityRequirement(value)
    elif isinstance(value, Mapping):
        try:
            requirement = CapabilityRequirement(**dict(value))
        except TypeError as exc:
            raise ValueError(f'invalid requirement: {exc}') from exc
    elif isinstance(value, Sequence):
        if len(value) == 1:
            requirement = CapabilityRequirement(str(value[0]))
        elif len(value) == 2:
            requirement = CapabilityRequirement(
                str(value[0]),
                str(value[1]),
            )
        elif len(value) == 3:
            requirement = CapabilityRequirement(
                str(value[0]),
                str(value[1]),
                bool(value[2]),
            )
        else:
            raise ValueError(
                f'invalid requirement {value!r}, expected 1 to 3 fields',
            )
    else:
        raise ValueError(f'unsupported requirement value: {value!r}')
    if not requirement.name:
        raise ValueError('requirement name must be non-empty string')
    if requirement.specifier:
        try:
            SpecifierSet(requirement.specifier)
        except InvalidSpecifier as exc:
            raise ValueError(
                f'invalid version specifier {requirement.specifier!r}: {exc}',
            ) from exc
    return requirement


def coerce_spec(
    fields: Mapping[str, Any],
) -> CapabilitySpec | RegistrationError:
    """Validate raw declaration fields.

    Returns either a :class:`CapabilitySpec` or a
    :class:`RegistrationError` describing the first problem found.
    """

    raw = dict(fields)
    provider = raw.get('provider')
    name = raw.get('name')

    def fail(message: str) -> RegistrationError:
        return RegistrationError(
            provider=provider if isinstance(provider, str) else None,
            name=name if isinstance(name, str) else None,
            message=message,
            raw=raw,
        )

    known_fields = {field.name for field in dataclasses.fields(CapabilitySpec)}
    unknown_fields = sorted(set(raw) - known_fields)
    if unknown_fields:
        return fail(f'unknown capability fields: {", ".join(unknown_fields)}')
    if not isinstance(name, str) or not name:
        return fail('capability name must be non-empty string')
    if not isinstance(provider, str) or not provider:
        return fail('capability provider must be non-empty string')

    version = raw.get('version', '0')
    if not isinstance(version, str):
        return fail('capability version must be a string')
    try:
        Version(version)
    except InvalidVersion as exc:
        return fail(f'invalid version {version!r}: {exc}')

    requires_raw = raw.get('requires', ())
    if not isinstance(requires_raw, Sequence) or isinstance(
        requires_raw,
        str,
    ):
        return fail('requires must be a sequence of requirements')
    requires: list[CapabilityRequirement] = []
    try:
        for item in requires_raw:
            requires.append(_coerce_requirement(item))
    except ValueError as exc:
        return fail(str(exc))
    required_names = [item.name for item in requires]
    if len(set(required_names)) != len(required_names):
        return fail('duplicate requirement names in requires')

    conflicts_raw = raw.get('conflicts', ())
    if not isinstance(conflicts_raw, Sequence) or isinstance(
        conflicts_raw,
        str,
    ):
        return fail('conflicts must be a sequence of capability names')
    conflicts: list[str] = []
    for item in conflicts_raw:
        if not isinstance(item, str) or not item:
            return fail('conflict names must be non-empty strings')
        conflicts.append(item)
    if len(set(conflicts)) != len(conflicts):
        return fail('duplicate conflict names')
    if name in conflicts:
        return fail('capability must not conflict with itself')

    fixture_scope = raw.get('fixture_scope', 'function')
    if fixture_scope not in VALID_FIXTURE_SCOPES:
        return fail(
            'invalid fixture_scope '
            f'{fixture_scope!r}, expected one of {VALID_FIXTURE_SCOPES}',
        )

    teardown = raw.get('teardown', 'best-effort')
    if teardown not in VALID_TEARDOWN_POLICIES:
        return fail(
            'invalid teardown policy '
            f'{teardown!r}, expected one of {VALID_TEARDOWN_POLICIES}',
        )

    return CapabilitySpec(
        name=name,
        provider=provider,
        version=version,
        requires=tuple(requires),
        conflicts=tuple(conflicts),
        fixture_scope=fixture_scope,
        teardown=teardown,
    )


class CapabilityRegistry:
    """Collects capability declarations from plugins."""

    def __init__(self) -> None:
        self._specs: list[CapabilitySpec] = []
        self._errors: list[RegistrationError] = []

    @property
    def specs(self) -> Sequence[CapabilitySpec]:
        return self._specs

    @property
    def errors(self) -> Sequence[RegistrationError]:
        return self._errors

    def register(
        self,
        spec: CapabilitySpec | Mapping[str, Any] | None = None,
        /,
        **fields: Any,
    ) -> None:
        """Register a capability.

        Accepts a :class:`CapabilitySpec`, a mapping or keyword arguments::

            register_capability(name='mockserver', provider='testsuite',
                                requires=[('asyncio', '>=1.0')])
        """

        if spec is not None:
            if fields:
                raise TypeError(
                    'register() takes either a spec or keyword arguments',
                )
            if isinstance(spec, CapabilitySpec):
                self._specs.append(spec)
                return
            if isinstance(spec, Mapping):
                fields = dict(spec)
            else:
                raise TypeError(f'unsupported capability spec: {spec!r}')
        result = coerce_spec(fields)
        if isinstance(result, RegistrationError):
            self._errors.append(result)
        else:
            self._specs.append(result)
