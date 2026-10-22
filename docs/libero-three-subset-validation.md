# Three-LIBERO subset validation

The single-image validation ran Long++ state 1 (12 cases), Ultra-Long state 1
(9 cases), and PRO state 3 (80 cases) with one GPU and 16 concurrent workers.
All used GPT-6 Astra with max reasoning, the locked ICL/tokenizer, no VLA or
SAM3, and the official 1x physical-step boundary.

| Benchmark | Closest historical result | Image first pass | After retrying only invalid attempts | Difference |
| --- | ---: | ---: | ---: | ---: |
| Long++ | 12/12 | 11/12 | 11/12 | -1 |
| Ultra-Long | 8/9 | 7/9 | 8/9 | 0 |
| PRO | 78/80 | 74/80 | 79/80 | +1 |
| **Total** | **98/101** | **92/101** | **98/101** | **0** |

The success count is close: the final total is identical and each Benchmark is
within one case of its closest historical reference. Reliability is weaker:
94/101 cases had legal native terminals on the first pass and 100/101 after
retry; one PRO case remained an Agent/API error without a score.

All 61 historical ICL package manifests match the packaged copies. The
validation used Hugging Face dataset commit
`84d9bb295b2f01484875848c0c414381baac5fea`; the three package-tree digests are
unchanged at the current public lock
`e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c`. The tokenizer SHA-256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.

Conclusion: task success is substantially reproduced and no missing image
dependency, GPU OOM, or EGL startup failure was observed. The result does not
claim deterministic per-case reproduction or 101/101 first-pass reliability.
