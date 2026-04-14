### `mlx-community/gemma-4-e4b-it-4bit` — Context & Memory Profile

Context depths tested: 512

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 512 | 0.36s | 52.44 tok/s | N/A | 5.7 GB | 29.2 GB |
| SSD Stream | 512 | 0.42s | 53.62 tok/s | N/A | 7.0 GB | 30.4 GB |
| TurboQuant | 512 | 0.36s | 53.73 tok/s | N/A | 5.7 GB | 29.0 GB |
| SSD + TurboQuant | 512 | 0.42s | 53.65 tok/s | N/A | 7.2 GB | 30.4 GB |
| SSD + 16-Worker Prefetch | 512 | 0.43s | 53.57 tok/s | N/A | 7.1 GB | 30.3 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
