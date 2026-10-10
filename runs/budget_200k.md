Evaluation logs, steps 160,000-200,000 (5 evaluations × 20 episodes per seed); mean ± std over seeds

| Agent | Seeds | Goal reached | Collision | Timeout |
|---|---|---|---|---|
| TD3 | 3 | 52 ± 29% | 32 ± 18% | 16 ± 11% |
| ITD3 (one-step) | 3 | 13 ± 2% | 34 ± 7% | 53 ± 6% |
| ITD3 (multi-step) | 3 | 13 ± 2% | 32 ± 8% | 56 ± 6% |
| ITD3 without bonus | 2 | 14 ± 1% | 42 ± 10% | 44 ± 10% |
| TD3 + one-step bonus | 3 | 69 ± 12% | 21 ± 4% | 9 ± 8% |
| TD3 + multi-step bonus | 3 | 77 ± 12% | 16 ± 8% | 8 ± 5% |
| PPO | 3 | 29 ± 16% | 14 ± 7% | 57 ± 23% |
