# DriveVA-lite

DriveVA-lite explores dynamic token selection for autonomous driving planning.

The goal is to reduce the number of history tokens while preserving or improving DriveVA planning performance.

## Current Best Result

| Method | Keep Ratio | PDM |
|---|---:|---:|
| Dense Original DriveVA | 100% | 0.902805 |
| Random50 Dynamic | 50% | 0.909534 |
| Grad-BCE w/o Curriculum | 50% | 0.904764 |
| Gradient + Diversity, λ=0.1 | 50% | 0.870815 |
| **Gradient + Diversity, λ=0.3** | **50%** | **0.925753** |
| Gradient + Diversity, λ=0.5 | 50% | 0.895035 |

Current best:

```text
Gradient × Input + Spatial-Temporal Diversity
λ = 0.3
Keep 50%
PDM = 0.925753
