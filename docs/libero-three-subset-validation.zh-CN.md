# 三套 LIBERO 子集简要验证记录

单镜像复测覆盖 Long++ state 1（12 例）、Ultra-Long state 1（9 例）和 PRO
state 3（80 例），使用 1 张 GPU、并发 16、GPT-6 Astra max reasoning、锁定的
ICL/tokenizer，不使用 VLA 或 SAM3，并遵守官方 1× 物理步数边界。

| Benchmark | 最接近历史结果 | 镜像首轮 | 仅重试异常后 | 差值 |
| --- | ---: | ---: | ---: | ---: |
| Long++ | 12/12 | 11/12 | 11/12 | -1 |
| Ultra-Long | 8/9 | 7/9 | 8/9 | 0 |
| PRO | 78/80 | 74/80 | 79/80 | +1 |
| **合计** | **98/101** | **92/101** | **98/101** | **0** |

成功点数基本接近：最终总点数相同，每个 benchmark 与最接近历史结果最多相差 1 点。
首轮稳定性仍有差距：首轮 94/101 个 case 有合法原生终态，仅重试异常后为 100/101，
仍有 1 个 PRO case 因 Agent/API 异常没有分数。

历史记录中的 61 个 ICL package manifest 与镜像副本全部一致。复测使用 Hugging Face
数据集提交 `84d9bb295b2f01484875848c0c414381baac5fea`；当前公开锁定提交
`e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c` 的三棵 package 目录摘要保持一致。
tokenizer SHA-256 为
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`。

结论：任务成功点数基本复现，未发现镜像缺依赖、GPU OOM 或 EGL 启动错误；但不宣称
逐 case 确定性复现，也不宣称首轮达到 101/101 稳定终态。
