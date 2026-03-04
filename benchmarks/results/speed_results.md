# Speed Benchmark Results

### Device: MPS

| Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|-------|----------|------------|---------|---------|
| Face Detection | 4.9 | 4.1 | 9.7 | 23.2 |
| CLIP Encoding | 18.3 | 14.5 | 31.9 | 93.6 |
| FaRL Encoding | 17.3 | 15.1 | 29.7 | 53.8 |
| Similarity+Fusion | 4.6 | 1.8 | 18.0 | 44.1 |
| Total Pipeline | 45.2 | 38.0 | 90.9 | 154.6 |

### Device: CPU

| Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|-------|----------|------------|---------|---------|
| Face Detection | 9.0 | 8.9 | 16.1 | 19.7 |
| CLIP Encoding | 239.6 | 256.5 | 268.4 | 270.7 |
| FaRL Encoding | 71.9 | 60.0 | 135.9 | 188.7 |
| Similarity+Fusion | 7.1 | 8.3 | 12.2 | 17.0 |
| Total Pipeline | 327.6 | 329.4 | 345.3 | 350.9 |

