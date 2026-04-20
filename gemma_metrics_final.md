### `mlx-community/gemma-4-26b-a4b-it-4bit` — Context & Memory Profile

Context depths tested: 512,40000,100000

| Configuration | Context Size | Prefill Speed | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Vanilla | 512 | 1112.28 tok/s | 74.30 tok/s | N/A | 14.7 GB | 20.9 GB |
| Vanilla | 40000 | 1708.89 tok/s | 28.91 tok/s | N/A | 48.7 GB | 57.7 GB |
| Vanilla | 100000 | 1327.82 tok/s | 27.82 tok/s | N/A | 48.5 GB | 57.3 GB |
| SSD Stream | 512 | 457.18 tok/s | 17.11 tok/s | N/A | 4.5 GB | 13.8 GB |
| SSD Stream | 40000 | 1697.18 tok/s | 13.85 tok/s | N/A | 38.5 GB | 47.6 GB |
| SSD Stream | 100000 | 1263.53 tok/s | 10.73 tok/s | N/A | 49.3 GB | 58.5 GB |
| TurboQuant | 512 | 1432.54 tok/s | 66.08 tok/s | N/A | 14.7 GB | 23.7 GB |
| TurboQuant | 40000 | 2881.22 tok/s | 64.41 tok/s | N/A | 18.2 GB | 27.3 GB |
| TurboQuant | 100000 | 2803.57 tok/s | 63.05 tok/s | N/A | 20.7 GB | 29.7 GB |
| SSD + TurboQuant | 512 | 653.75 tok/s | 17.08 tok/s | N/A | 4.5 GB | 13.6 GB |
| SSD + TurboQuant | 40000 | 1972.06 tok/s | 14.90 tok/s | N/A | 7.9 GB | 17.5 GB |
| SSD + TurboQuant | 100000 | 2259.31 tok/s | 16.28 tok/s | N/A | 10.2 GB | 19.2 GB |
| SSD + 16-Worker Prefetch | 512 | 748.49 tok/s | 18.16 tok/s | N/A | 4.6 GB | 13.7 GB |
| SSD + 16-Worker Prefetch | 40000 | 1718.78 tok/s | 13.30 tok/s | N/A | 38.6 GB | 47.6 GB |
| SSD + 16-Worker Prefetch | 100000 | 1277.40 tok/s | 10.43 tok/s | N/A | 49.3 GB | 58.5 GB |
| TurboQuant + Speculative (3) | 512 | 342.91 tok/s | 38.94 tok/s | N/A | 16.9 GB | 25.9 GB |
| TurboQuant + Speculative (3) | 40000 | 2930.71 tok/s | 62.76 tok/s | N/A | 20.4 GB | 29.4 GB |
| TurboQuant + Speculative (3) | 100000 | 2850.21 tok/s | 58.71 tok/s | N/A | 22.9 GB | 31.9 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
