"""Preflight diagnostics for the capability graph."""

from __future__ import annotations

import dataclasses
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from .graph import CapabilityGraph, build_graph
from .registry import (
    SCOPE_RANK,
    CapabilityRegistry,
    CapabilitySpec,
    RegistrationError,
)

#: Version of the machine-readable report format.
SCHEMA_VERSION = '1.0'

SEVERITY_ERROR = 'error'
SEVERITY_WARNING = 'warning'


class DiagnosticCode:
    INVALID_DECLARATION = 'invalid_declaration'
    DUPLICATE_PROVIDER = 'duplicate_provider'
    DUPLICATE_REGISTRATION = 'duplicate_registration'
    MISSING_DEPENDENCY = 'missing_dependency'
    MISSING_OPTIONAL_DEPENDENCY = 'missing_optional_dependency'
    VERSION_INCOMPATIBLE = 'version_incompatible'
    CONFLICTING_CAPABILITY = 'conflicting_capability'
    CIRCULAR_DEPENDENCY = 'circular_dependency'
    SCOPE_TOO_BROAD = 'scope_too_broad'


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    capability: str | None = None
    provider: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'severity': self.severity,
            'message': self.message,
            'capability': self.capability,
            'provider': self.provider,
            'details': self.details,
        }


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    status: str
    capabilities: tuple[dict[str, Any], ...]
    providers: tuple[str, ...]
    graph: CapabilityGraph
    teardown_order: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == SEVERITY_ERROR
        )

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == SEVERITY_WARNING
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': SCHEMA_VERSION,
            'status': self.status,
            'summary': {
                'capabilities': len(self.capabilities),
                'providers': len(self.providers),
                'errors': len(self.errors),
                'warnings': len(self.warnings),
            },
            'capabilities': list(self.capabilities),
            'providers': list(self.providers),
            'graph': {
                'nodes': list(self.graph.nodes),
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'optional': edge.optional,
                    }
                    for edge in self.graph.edges
                ],
                'order': list(self.graph.order),
            },
            'teardown_order': list(self.teardown_order),
            'diagnostics': [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


def run_preflight(registry: CapabilityRegistry) -> PreflightReport:
    """Build the capability graph and run all preflight checks."""

    diagnostics: list[Diagnostic] = []

    for error in registry.errors:
        diagnostics.append(_invalid_declaration_diagnostic(error))

    grouped: dict[str, list[CapabilitySpec]] = {}
    for spec in registry.specs:
        grouped.setdefault(spec.name, []).append(spec)

    representatives: dict[str, CapabilitySpec] = {}
    for name, entries in sorted(grouped.items()):
        provider_names = sorted({entry.provider for entry in entries})
        if len(provider_names) > 1:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.DUPLICATE_PROVIDER,
                    severity=SEVERITY_ERROR,
                    message=(
                        f"capability '{name}' is provided by multiple "
                        f'providers: {", ".join(provider_names)}'
                    ),
                    capability=name,
                    details={'providers': provider_names},
                ),
            )
        elif len(entries) > 1:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.DUPLICATE_REGISTRATION,
                    severity=SEVERITY_ERROR,
                    message=(
                        f"capability '{name}' was registered more than once "
                        f"by provider '{provider_names[0]}'"
                    ),
                    capability=name,
                    provider=provider_names[0],
                    details={'count': len(entries)},
                ),
            )
        chosen_provider = provider_names[0]
        representatives[name] = next(
            entry for entry in entries if entry.provider == chosen_provider
        )

    graph = build_graph(representatives.values())

    for cycle in graph.cycles:
        path = ' -> '.join([*cycle, cycle[0]])
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.CIRCULAR_DEPENDENCY,
                severity=SEVERITY_ERROR,
                message=f'circular dependency detected: {path}',
                capability=cycle[0],
                details={'cycle': list(cycle)},
            ),
        )

    seen_conflict_pairs: set[tuple[str, str]] = set()
    for name in graph.nodes:
        spec = representatives[name]
        diagnostics.extend(
            _conflict_diagnostics(
                spec,
                representatives,
                seen_conflict_pairs,
            ),
        )
        diagnostics.extend(_requirement_diagnostics(spec, representatives))

    diagnostics.sort(
        key=lambda diagnostic: (
            diagnostic.code,
            diagnostic.capability or '',
            diagnostic.provider or '',
            diagnostic.message,
        ),
    )

    capabilities = tuple(
        _serialize_spec(representatives[name]) for name in graph.nodes
    )
    providers: tuple[str, ...] = tuple(
        sorted(
            {
                entry.provider
                for entries in grouped.values()
                for entry in entries
            },
        ),
    )
    teardown_order = tuple(
        name
        for name in reversed(graph.order)
        if representatives[name].teardown != 'none'
    )
    status = (
        SEVERITY_ERROR
        if any(d.severity == SEVERITY_ERROR for d in diagnostics)
        else 'ok'
    )
    return PreflightReport(
        status=status,
        capabilities=capabilities,
        providers=providers,
        graph=graph,
        teardown_order=teardown_order,
        diagnostics=tuple(diagnostics),
    )


def _invalid_declaration_diagnostic(
    error: RegistrationError,
) -> Diagnostic:
    return Diagnostic(
        code=DiagnosticCode.INVALID_DECLARATION,
        severity=SEVERITY_ERROR,
        message=f'invalid capability declaration: {error.message}',
        capability=error.name,
        provider=error.provider,
        details={'raw': error.raw},
    )


def _conflict_diagnostics(
    spec: CapabilitySpec,
    representatives: dict[str, CapabilitySpec],
    seen_pairs: set[tuple[str, str]],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for target in sorted(set(spec.conflicts)):
        if target not in representatives:
            continue
        left, right = sorted((spec.name, target))
        pair = (left, right)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.CONFLICTING_CAPABILITY,
                severity=SEVERITY_ERROR,
                message=(
                    f"capabilities '{spec.name}' and '{target}' conflict "
                    'with each other'
                ),
                capability=spec.name,
                provider=spec.provider,
                details={'with': target},
            ),
        )
    return diagnostics


def _requirement_diagnostics(
    spec: CapabilitySpec,
    representatives: dict[str, CapabilitySpec],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    requirements = sorted(
        spec.requires,
        key=lambda item: (item.name, item.specifier, item.optional),
    )
    for requirement in requirements:
        target = representatives.get(requirement.name)
        if target is None:
            code = (
                DiagnosticCode.MISSING_OPTIONAL_DEPENDENCY
                if requirement.optional
                else DiagnosticCode.MISSING_DEPENDENCY
            )
            severity = (
                SEVERITY_WARNING if requirement.optional else SEVERITY_ERROR
            )
            kind = (
                'optional dependency' if requirement.optional else 'dependency'
            )
            diagnostics.append(
                Diagnostic(
                    code=code,
                    severity=severity,
                    message=(
                        f"missing {kind} '{requirement.name}' required by "
                        f"capability '{spec.name}'"
                    ),
                    capability=spec.name,
                    provider=spec.provider,
                    details={
                        'requirement': requirement.name,
                        'specifier': requirement.specifier,
                        'optional': requirement.optional,
                    },
                ),
            )
            continue
        if requirement.specifier:
            specifier = SpecifierSet(requirement.specifier)
            if Version(target.version) not in specifier:
                diagnostics.append(
                    Diagnostic(
                        code=DiagnosticCode.VERSION_INCOMPATIBLE,
                        severity=SEVERITY_ERROR,
                        message=(
                            f"capability '{spec.name}' requires "
                            f"'{requirement.name}{requirement.specifier}' "
                            f"but provider '{target.provider}' provides "
                            f"version '{target.version}'"
                        ),
                        capability=spec.name,
                        provider=spec.provider,
                        details={
                            'requirement': requirement.name,
                            'specifier': requirement.specifier,
                            'provided_version': target.version,
                        },
                    ),
                )
        if SCOPE_RANK[spec.fixture_scope] < SCOPE_RANK[target.fixture_scope]:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.SCOPE_TOO_BROAD,
                    severity=SEVERITY_ERROR,
                    message=(
                        f"capability '{spec.name}' uses scope "
                        f"'{spec.fixture_scope}' which is broader than "
                        f"dependency '{requirement.name}' scope "
                        f"'{target.fixture_scope}'"
                    ),
                    capability=spec.name,
                    provider=spec.provider,
                    details={
                        'requirement': requirement.name,
                        'scope': spec.fixture_scope,
                        'dependency_scope': target.fixture_scope,
                    },
                ),
            )
    return diagnostics


def _serialize_spec(spec: CapabilitySpec) -> dict[str, Any]:
    return {
        'name': spec.name,
        'provider': spec.provider,
        'version': spec.version,
        'fixture_scope': spec.fixture_scope,
        'teardown': spec.teardown,
        'requires': [
            {
                'name': requirement.name,
                'specifier': requirement.specifier,
                'optional': requirement.optional,
            }
            for requirement in sorted(
                spec.requires,
                key=lambda item: (item.name, item.specifier, item.optional),
            )
        ],
        'conflicts': sorted(spec.conflicts),
    }
