"""Capability registration and preflight diagnostics plugin.

Plugins declare capabilities (name, version, dependencies/conflicts, fixture
scope and teardown policy) through the ``pytest_capability_register`` hook.
When preflight is enabled testsuite builds a deterministic capability graph
and reports duplicate providers, version incompatibilities, circular
dependencies, scope mismatches and missing optional dependencies.

The plugin is inert unless ``--testsuite-preflight`` or
``--testsuite-preflight-report`` is passed, so plugin discovery and fixture
lifecycle stay unchanged by default.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from testsuite._internal.capabilities import (
    CapabilityRegistry,
    PreflightReport,
    run_preflight,
)

_REPORT_ATTR = '_testsuite_preflight_report'


class CapabilityHookspec:
    def pytest_capability_register(self, register_capability):
        """Register testsuite capability for preflight diagnostics."""


def pytest_addhooks(pluginmanager):
    pluginmanager.add_hookspecs(CapabilityHookspec)


def pytest_addoption(parser):
    group = parser.getgroup(
        'testsuite-preflight',
        'Testsuite capability preflight diagnostics',
    )
    group.addoption(
        '--testsuite-preflight',
        action='store_true',
        help=(
            'Run capability preflight diagnostics at startup: build the '
            'capability graph and fail on duplicate providers, version '
            'conflicts, circular dependencies and fixture scope mismatches.'
        ),
    )
    group.addoption(
        '--testsuite-preflight-report',
        metavar='PATH',
        default=None,
        help=(
            'Write machine-readable preflight report (JSON) to PATH. '
            'Implies --testsuite-preflight.'
        ),
    )


def pytest_configure(config):
    report_path = config.option.testsuite_preflight_report
    if not config.option.testsuite_preflight and not report_path:
        return

    registry = CapabilityRegistry()
    config.pluginmanager.hook.pytest_capability_register(
        register_capability=registry.register,
    )
    report = run_preflight(registry)
    setattr(config, _REPORT_ATTR, report)

    if report_path:
        try:
            path = pathlib.Path(report_path)
            path.write_text(json.dumps(report.to_dict(), indent=2) + '\n')
        except OSError as exc:
            raise pytest.UsageError(
                f'failed to write preflight report to {report_path}: {exc}',
            ) from exc

    if report.status == 'error':
        raise pytest.UsageError(_format_error_summary(report, report_path))


def pytest_report_header(config):
    report: PreflightReport | None = getattr(
        config,
        _REPORT_ATTR,
        None,
    )
    if report is None:
        return None
    summary = report.to_dict()['summary']
    return [
        'testsuite preflight: {status}, {capabilities} capabilities from '
        '{providers} providers, {errors} errors, {warnings} warnings'.format(
            status=report.status,
            capabilities=summary['capabilities'],
            providers=summary['providers'],
            errors=summary['errors'],
            warnings=summary['warnings'],
        ),
    ]


def _format_error_summary(report: PreflightReport, report_path: str | None):
    lines = ['testsuite capability preflight failed:']
    for diagnostic in report.errors:
        lines.append(f'  [{diagnostic.code}] {diagnostic.message}')
    if report.warnings:
        lines.append(
            f'  ({len(report.warnings)} warning(s), see report for details)',
        )
    if report_path:
        lines.append(f'  machine-readable report: {report_path}')
    return '\n'.join(lines)
