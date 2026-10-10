200 test episodes (seeds 0-199); mean ± std over training seeds

| Agent | Training steps | Seeds | Goal reached | Collision | Timeout | Mean return |
|---|---|---|---|---|---|---|
| TD3 | 500,000 | 3 | 93 ± 2% | 7 ± 1% | 0 ± 0% | 79.1 ± 3.2 |
| ITD3 (one-step) | 500,000 | 3 | 12 ± 0% | 57 ± 4% | 31 ± 4% | -73.6 ± 2.4 |
| ITD3 (multi-step) | 500,000 | 3 | 26 ± 23% | 63 ± 20% | 11 ± 13% | -50.9 ± 43.8 |
| ITD3 without bonus | 200,000 | 2 | 15 ± 2% | 36 ± 4% | 49 ± 5% | -56.3 ± 0.7 |
| TD3 + one-step bonus | 500,000 | 3 | 96 ± 1% | 3 ± 1% | 0 ± 0% | 86.3 ± 2.3 |
| TD3 + multi-step bonus | 500,000 | 3 | 96 ± 0% | 4 ± 0% | 0 ± 0% | 84.2 ± 0.6 |
| PPO | 500,000 | 3 | 73 ± 25% | 10 ± 1% | 17 ± 24% | 45.4 ± 39.4 |
| Random | - | - | 0% | 62% | 38% | -94.3 |
