### `mlx-community/Qwen3.6-35B-A3B-4bit` — Context & Memory Profile

Context depths tested: 512,40000,100000

| Configuration | Context Size | Prefill Speed | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Vanilla | 512 | 907.73 tok/s | 27.03 tok/s | N/A | 19.0 GB | 26.8 GB |
| Vanilla | 40000 | 1572.92 tok/s | 24.93 tok/s | N/A | 49.3 GB | 57.2 GB |
| Vanilla | 100000 | 446.06 tok/s | 4.09 tok/s | N/A | 49.3 GB | 58.3 GB |
| SSD Stream | 512 | 280.46 tok/s | 13.34 tok/s | N/A | 4.5 GB | 13.6 GB |
| SSD Stream | 40000 | 1259.90 tok/s | 11.65 tok/s | N/A | 37.5 GB | 46.4 GB |
| SSD Stream | 100000 | 875.68 tok/s | 3.64 tok/s | N/A | 49.4 GB | 58.4 GB |
| TurboQuant | 512 | 1089.10 tok/s | 30.64 tok/s | N/A | 19.0 GB | 27.8 GB |
| TurboQuant | 40000 | 1807.38 tok/s | 1.86 tok/s | N/A | 22.7 GB | 31.9 GB |
| TurboQuant | 100000 | 1540.21 tok/s | 0.16 tok/s | N/A | 27.7 GB | 36.9 GB |
| SSD + TurboQuant | 512 | 298.15 tok/s | 13.28 tok/s | N/A | 4.5 GB | 13.8 GB |
| SSD + TurboQuant | 40000 | 1390.47 tok/s | 5.57 tok/s | N/A | 8.5 GB | 17.5 GB |
| SSD + TurboQuant | 100000 | 1268.80 tok/s | 3.95 tok/s | N/A | 13.4 GB | 22.3 GB |
| SSD + 16-Worker Prefetch | 512 | 544.29 tok/s | 14.89 tok/s | N/A | 4.5 GB | 13.7 GB |
| SSD + 16-Worker Prefetch | 40000 | 1284.39 tok/s | 11.31 tok/s | N/A | 37.5 GB | 46.3 GB |
| SSD + 16-Worker Prefetch | 100000 | 846.98 tok/s | 3.37 tok/s | N/A | 49.4 GB | 58.8 GB |
| TurboQuant + Speculative (0.8B) | 512 | 965.85 tok/s | 27.30 tok/s | N/A | 20.1 GB | 29.6 GB |
| TurboQuant + Speculative (0.8B) | 40000 | 1802.63 tok/s | 2.75 tok/s | N/A | 23.8 GB | 33.3 GB |
| TurboQuant + Speculative (0.8B) | 100000 | 1568.01 tok/s | 4.38 tok/s | N/A | 28.7 GB | 38.0 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
