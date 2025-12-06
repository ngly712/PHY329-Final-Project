# Using the Code
This repository is organized into three main Python modules:

- `map/standardMap.py` — defines the `StandardMap` class, which runs simulations of the Chirikov standard map and stores results in a list of runs (`.runs`).
- `plots/mapEval.py` — defines the `MapEvaluator` class, which provides helper methods to extract data from `StandardMap.runs` for plotting (tails, phase space, I–K diagnostic samples).
- `plots/mapPlot.py` — contains plotting utilities for phase–space plots and \(IK\) diagnostic sweeps.

Scripts should be run from the repository root so that imports (e.g., `from map.standardMap import StandardMap`) resolve correctly.

## Plotting Phase-Space Diagrams

### Import Relevant Modules

```python
from map.standardMap import StandardMap
from plots.mapPlot import plot_phase_generic
```

### Small K: Mostly Invariant Curves (KAM Tori)

```python
K = 0.2
n_iters = 5000
n_sim = 25

aMap = StandardMap(K=K, nIters=n_iters, seed=1)
aMap.simulate(ic=n_sim)
run = aMap.runs[-1]["run"]

plot_phase_generic(
    run=run,
    mode="phase",
    point_size=0.15,
    title=rf"Phase Space for $K = {K}$ ({n_sim} ICs)"
)
```

### Moderate K: Resonance Islands, Periodic Orbits, Cantori

```python
K = 0.7
n_sim = 50

aMap = StandardMap(K=K, nIters=n_iters, seed=1)
aMap.simulate(ic=n_sim)
run = aMap.runs[-1]["run"]

plot_phase_generic(
    run=run,
    mode="phase",
    point_size=0.15,
    title=rf"Phase Space for $K = {K}$ ({n_sim} ICs)"
)
```

### Large K: Mostly Chaotic Sea

```python
K = 2.0
n_sim = 100

aMap = StandardMap(K=K, nIters=n_iters, seed=1)
aMap.simulate(ic=n_sim)
run = aMap.runs[-1]["run"]

plot_phase_generic(
    run=run,
    mode="phase",
    point_size=0.15,
    title=rf"Phase Space for $K = {K}$ ({n_sim} ICs)"
)
```

## Plotting an $IK$ Diagnostic Sweep

### Import Relevant Modules + Adjust Display Settings

```python
# standard Python libraries
import numpy as np
import matplotlib.pyplot as plt
# project modules
from map.standardMap import StandardMap
from plots.mapEval import MapEvaluator
from plots.mapPlot import plot_IK_diagnostic

# matplotlib defaults (optional for nicer plots)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = False
```

### Generate K Values

```python
# parameters for the IK diagnostic sweep
K_min = 0.0
K_max = 5.0
n_K   = 400            # number of K values
Ks    = np.linspace(K_min, K_max, n_K)
```

### Run Simulation

```python
n_sim   = 10          # number of ICs at each K (each set of ICs produces one orbit)
n_iters = 5000        # iterations per orbit
n_tail  = 300         # tail length for diagnostic plot
seed = 1              # seed used inside StandardMap

# generate runs via StandardMap
aMap = StandardMap(K=Ks[0], nIters=n_iters, seed=seed)

for K in Ks:
    aMap.K = K
    aMap.simulate(ic=n_sim)

# wrap runs in a MapEvaluator
evaluator = MapEvaluator(aMap.runs)

print(f"Completed {len(aMap.runs)} runs with K in [{K_min}, {K_max}].")
```

### Plot Diagnostic Sweep

```python
plot_IK_diagnostic(
    evaluator=evaluator,
    n_tail=n_tail,
    K_min=K_min,
    K_max=K_max,
    title="Standard Map I-K Diagnostic Plot",
    max_points=50_000,    # subsample for readability
    point_size=0.1,
    alpha=0.3,
)
```
