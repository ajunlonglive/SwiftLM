### `mlx-community/Qwen3.6-35B-A3B-4bit` — Context & Memory Profile

Context depths tested: 128,512,1000,2000

| Configuration | Context Size | Prefill Speed | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Vanilla | 128 | 271.30 tok/s | 30.98 tok/s | N/A | 18.6 GB | 26.3 GB |
| Vanilla | 512 | 1276.44 tok/s | 31.85 tok/s | N/A | 19.0 GB | 26.6 GB |
| Vanilla | 1000 | 2530.73 tok/s | 34.20 tok/s | N/A | 19.4 GB | 27.1 GB |
| Vanilla | 2000 | 3039.04 tok/s | 33.25 tok/s | N/A | 20.2 GB | 28.0 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
