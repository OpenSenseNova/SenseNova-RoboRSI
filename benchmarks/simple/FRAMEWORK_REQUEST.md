# SIMPLE source registration

This guide was migrated from NewRPent on 2026-11-18. Upstream
[PR #86](https://github.com/SenseTime-FVG/RoboRSI/pull/86) merged as
`2d06bcf1417f779f68611ab3671cd0e13ee63056`; references to #85, #86 and #88 below
describe that source repository. This repository independently registers the
eleven-file SIMPLE bundle in `roborsi/benchmarks/runtime_source.py` and accepts
`simple` in the standard exporter. Imported image/source digests do not identify
a build against this checkout; see [VALIDATION.md](VALIDATION.md).

The registration reuses the common HTTP/protocol sources and these eleven
benchmark-owned files:

```text
robots/simple/contracts.py
robots/simple/sources.json
robots/simple/catalog.py
robots/simple/episode.py
robots/simple/native.py
robots/simple/runtime_setup.py
robots/simple/worker.py
robots/simple/roborsi_adapter.py
robots/simple/benchmark_service.py
benchmarks/simple/__init__.py
benchmarks/simple/prepare.py
```

No Agent preset, model SDK, credential, generated catalog, historical ICL,
native checkout or dataset belongs in this source bundle. Both registration
and adapter/recipe files must be present; the existing missing-file and source
identity checks remain enabled.

## Verify the registered source

```bash
python -m benchmarks.package --benchmark simple --output /data/fresh-simple-context
python -c 'from roborsi.benchmarks.runtime_source import benchmark_source_files; print(benchmark_source_files("simple"))'
```

The standard exporter must preserve checkout/export digests. The runtime verifier
rejects undeclared files and inherited Agent SDKs. Basic imports and service CLI
help must not load a simulator or model SDK. Agent preset/prompt edits must not
change the server source digest. See BUILD.md and VALIDATION.md for checks.

## Historical evidence

Before #86 merged, isolated combined checkouts verified standard export,
source identity, guarded imports and HTTP metadata. Those historical checks
were not evidence of a merged dependency. The old v0.1.0 service image includes
that earlier proposal but lacks the later adapter/facade cleanup retry fixes.
It must not be presented as the fixed delivery image. Rebuild from the current
source and keep the image digest and exported source digest paired.

The upstream record kept PR #85 Draft pending review and image acceptance.
It does not establish this branch's review or CI status. Draft-skipped checks
are not executed CI; report checks actually run for this checkout.
