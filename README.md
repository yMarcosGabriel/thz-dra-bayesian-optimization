# Bayesian optimization for THz rectangular dielectric resonator antennas

Python implementation of feasibility-aware Bayesian optimization for the
inverse design of micro-scale rectangular dielectric resonator antennas (DRAs).
Ansys HFSS evaluates candidate geometries; a Gaussian-process regressor
with an anisotropic Matérn-5/2 kernel models the optimization objective.
Expected Improvement is weighted by the probability of feasible response,
and the search bounds can expand when promising candidates lie near a boundary.

The optimizer operates on a 0.001 µm fabrication grid. Frequency and relative
permittivity are prescribed for each campaign; the seven geometric variables
are optimized.

## Repository contents

| Path | Purpose |
| --- | --- |
| `pipeline_bo.py` | Simulation coupling, objective, Gaussian-process optimization and output generation |
| `configs/paper_targets.py` | Initial geometry bounds and frequency windows for 10 and 11 THz |
| `experiments/run_manuscript_targets.py` | Independent campaigns over the manuscript's frequency/permittivity targets |
| `experiments/fidelity_protocols.py` | Independent L/M/H campaign comparison for different HFSS mesh settings |
| `tests/test_core.py` | Regression tests that do not require Ansys |

## Requirements

Python 3.11, Ansys Electronics Desktop/HFSS with a valid license, and a
compatible PyAEDT installation. This repository does not provide the HFSS
project or the solver. Install dependencies in a dedicated environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set the path to an existing parametric HFSS project (the exact variable names
must match those defined in the code) and an output directory:

```powershell
$env:HFSS_PROJECT_PATH = "C:\path\to\your-project.aedt"
$env:RESULTS_DIR = "results"
```

## Run the manuscript configurations

```powershell
python -m experiments.run_manuscript_targets --frequencies 10 11 --permittivities 8 9 10
```

For independent fidelity-protocol campaigns at relative permittivity 9:

```powershell
python -m experiments.fidelity_protocols --levels L M H --frequencies 10 11 --permittivity 9
```

| Protocol | Initial-mesh slider | MaxDeltaS | Maximum adaptive passes |
| --- | ---: | ---: | ---: |
| L | 6 | 0.0100 | 12 |
| M | 7 | 0.0050 | 20 |
| H | 8 | 0.0025 | 26 |

Each protocol runs its **own** optimization from the original search bounds.
No observations are shared between protocols. Thus, the experiments assess
numerical fidelity and computational cost; the code does not implement a
joint multi-fidelity surrogate or cost-aware acquisition.

## Reproducibility

The nominal budget is at most 32 Latin-hypercube candidates followed by 40
sequential BO iterations (30 exploration and 10 local refinement). The LHS
stage can stop early; automatic bound expansion can add new evaluations.
Consequently, 72 is not a fixed total evaluation count.

The code contains no run-specific outputs. Complete electromagnetic results
require a compatible external HFSS project, licensed solver and appropriate
material, boundary and mesh definitions. No solver convergence or scientific
result is established by passing the CPU-only tests.

Run tests without starting HFSS:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). A DOI will
be added **after** the new repository's first release is archived on Zenodo.
Do not use the DOI of the earlier, separate repository for this code.

Software source is distributed under the MIT license. Ansys/HFSS and
third-party components remain subject to their own licenses.