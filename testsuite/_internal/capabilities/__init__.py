"""Testsuite capability registry and preflight diagnostics."""

from .diagnostics import (
    SCHEMA_VERSION,
    Diagnostic,
    DiagnosticCode,
    PreflightReport,
    run_preflight,
)
from .graph import CapabilityEdge, CapabilityGraph, build_graph
from .registry import (
    SCOPE_RANK,
    VALID_FIXTURE_SCOPES,
    VALID_TEARDOWN_POLICIES,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilitySpec,
    RegistrationError,
    coerce_spec,
)

__all__ = [
    'SCHEMA_VERSION',
    'SCOPE_RANK',
    'VALID_FIXTURE_SCOPES',
    'VALID_TEARDOWN_POLICIES',
    'CapabilityEdge',
    'CapabilityGraph',
    'CapabilityRegistry',
    'CapabilityRequirement',
    'CapabilitySpec',
    'Diagnostic',
    'DiagnosticCode',
    'PreflightReport',
    'RegistrationError',
    'build_graph',
    'coerce_spec',
    'run_preflight',
]
