# RoboDojo resource verification

`resource-lock.json` records the SHA-256 of the published Agent resources.
Verify the inventory from the repository root:

```bash
python -m benchmarks.robodojo.provenance
```

The native source inventory is maintained separately in
`robots/robodojo/native_files.json`. Third-party source licenses and copyright
notices are retained alongside the corresponding dependencies.
