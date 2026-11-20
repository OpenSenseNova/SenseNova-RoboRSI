# Benchmark build recipes

Benchmark and Agent services deploy independently. Use the shared
[evaluation guide](../docs/evaluation.md) and
[benchmark guide index](../docs/README.md) for the existing services.

The SIMPLE integration candidate adds a private reset catalog, a main-thread
native worker, an external Agent preset and independent native/service image
recipes ([English](simple/README.md) / [简体中文](simple/README.zh-CN.md)). Its
runtime source is registered and exportable from this checkout:

```bash
python -m benchmarks.package --benchmark simple --output /data/new-simple-context
```

Prepare assets and the native base using [ASSETS.md](simple/ASSETS.md) and
[BUILD.md](simple/BUILD.md). Imported image digests and native validation records
identify upstream builds; see [VALIDATION.md](simple/VALIDATION.md) for the
current migration checks and rebuild before deploying this shared framework.

This index preserves the SIMPLE entry introduced by upstream adapter commit
`237484ebf1d9863fdcb9694e4f4e329876d3730a`, updated to reflect source registration
and this repository's guide layout.
