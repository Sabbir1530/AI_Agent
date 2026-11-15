# Assignment 2 summary

Template + system-prompt baseline: **24** input tokens (measured with a 1-character prompt).
Valid pairs: 50/50

| Measure | English | Bangla |
|---|---|---|
| Input tokens (total) | 2056 | 5445 |
| Output tokens (total) | 2555 | 5092 |
| Cost USD (notional) | 0.001841 | 0.003872 |
| Latency mean / median (s) | 16.38 / 13.19 | 25.20 / 24.50 |
| Answers cut by max tokens | 7 | 19 |

- Total cost ratio (Bangla / English): **2.103**
- Pooled input ratio (sum of tokens): 2.648
- Per-pair input ratio: mean 2.41, median 2.12, min 1.20, max 4.45, 95% CI [2.18, 2.65]
- Per-pair ratio after removing the fixed baseline: mean 4.87, median 5.06, min 1.64, max 8.20, 95% CI [4.49, 5.26]

## By category (per-pair input ratio, largest gap first)
| category | pairs | mean | median |
|---|---|---|---|
| long | 8 | 3.73 | 3.91 |
| summarise | 8 | 3.28 | 3.37 |
| instruction | 8 | 2.21 | 2.17 |
| explanation | 8 | 2.10 | 2.12 |
| question | 10 | 1.74 | 1.77 |
| arithmetic | 8 | 1.58 | 1.55 |

Lowest-ratio pair: P30; highest-ratio pair: P47 (look at these for the 'surprising prompt' in your report).
