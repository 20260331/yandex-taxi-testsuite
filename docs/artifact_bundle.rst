Failure artifact bundles
************************

For every testcase testsuite can collect a *failure artifact bundle* with the
service failure scene: service stdout/stderr slices, structured service log
slices, a redacted startup command summary, ports and environment information,
the testcase trace id and an append-only event timeline. An atomic manifest
links the test node, service instances and their exit reasons.

By default bundles are collected for every test, retained on failure and
deleted on success. Artifact collection failures never affect test execution.

Bundle layout
=============

Bundles are stored under ``--service-artifacts-dir``::

  testsuite-artifacts/
    session-<timestamp>-<pid>/
      main/                      # or pytest-xdist worker id (gw0, gw1, ...)
        bundles/
          <nodeid-slug>-<id>/
            manifest.json       # atomic: test node, services, exit reasons
            timeline.jsonl      # spawn/ready/bind/exit and test phase events
            services/
              <instance-id>/
                startup.json    # redacted startup summary
                stdout.log      # per-test slice, capacity bounded
                stderr.log
            service_logs/
              <title>.log       # slices of registered service log files

With ``--service-artifacts-format=zip`` each retained bundle is replaced by a
``zip`` archive next to its directory.

Manifest
========

``manifest.json`` (schema version 1) contains:

* ``session`` — session id, xdist worker id, hostname, python and testsuite
  versions;
* ``test`` — node id, outcome, trace id, started/finished timestamps, duration
  and a redacted failure excerpt;
* ``services`` — for every service instance used by the test: instance id,
  scope name, pid, command and command line, base command, working directory,
  ping url, listening ports, redacted environment, shutdown signal, start/
  ready/exit timestamps, exit code and exit reason, references to stdout/stderr
  slices with byte counts;
* ``service_logs`` — structured log file slices with truncation markers;
* ``artifacts`` — index of the files above.

The manifest is written atomically (temporary file + atomic replace).

Options
=======

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Description
   * - ``--service-artifacts``
     - ``failure`` (default, retain only failed bundles), ``always`` (retain
       every bundle) or ``off`` (disable collection).
   * - ``--service-artifacts-dir PATH``
     - Root directory for bundles (default: ``testsuite-artifacts`` under
       pytest root dir).
   * - ``--service-artifacts-format``
     - ``dir`` (default) or ``zip`` archive.
   * - ``--service-artifacts-max-bytes SIZE``
     - Per stream (stdout/stderr/service log) byte cap, e.g. ``10M`` (default
       10 MiB, ``0`` disables the cap). Extra output is dropped with a marker.
   * - ``--service-artifacts-session-max-bytes SIZE``
     - Total output bytes per worker session (default 256 MiB, ``0``
       disables the cap).
   * - ``--service-artifacts-retain N``
     - Number of recent sessions to keep (default 5). Older session
       directories are pruned on session start under a file lock, which makes
       pruning safe with pytest-xdist.
   * - ``--service-artifacts-redact PATTERNS``
     - Comma separated list of additional case-insensitive environment key
       patterns to redact.

Every option has a corresponding ini setting (``service_artifacts``,
``service_artifacts_dir``, ``service_artifacts_format``,
``service_artifacts_max_bytes``, ``service_artifacts_session_max_bytes``,
``service_artifacts_retain`` and ``service_artifacts_redact``). Command line
options take precedence over ini settings.

Redaction
==========

Environment values and ``key=value`` command line arguments are redacted when
the key contains one of the default patterns (case-insensitive):
``PASSWORD``, ``PASSWD``, ``PWD``, ``TOKEN``, ``SECRET``, ``API_KEY``,
``APIKEY``, ``CREDENTIAL``, ``PRIVATE_KEY``, ``COOKIE``, ``AUTHORIZATION``.
Extra patterns can be added with ``--service-artifacts-redact``. Redacted
values are replaced with ``***redacted***``. Secret values observed in the
service environment are also scrubbed from the failure excerpt.

Only environment variables explicitly passed to the service spawner (and
matching port-bearing keys) are recorded; the ambient process environment is
not copied into the bundle.

Fixtures
========

artifact_bundle
~~~~~~~~~~~~~~~

.. py:function:: artifact_bundle

   Returns the bundle of the currently running test node (or ``None`` when
   collection is disabled). The bundle object exposes ``path`` and
   ``add_event(event, **data)`` for custom timeline records.

   .. code-block:: python

      async def test_publish(artifact_bundle):
          artifact_bundle.add_event('publish_started', queue='orders')
          ...

Notes
=====

* Service stdout/stderr are captured per test: when collection is enabled,
  service pipes are owned by testsuite even if no user handlers were passed
  (standard output no longer inherits the terminal). Live service logs
  (enabled with ``-s``) are not affected: they stream from registered log
  files.
* Services may be reused across testcases; a service instance is linked to
  every testcase that used it. Its exit code and reason are recorded for the
  testcase during which the service terminated.
* Service log files registered with ``servicelogs_register_logfile`` are
  sliced from the testcase start position; flushers registered with
  ``servicelogs_register_flusher`` are run before slicing.
