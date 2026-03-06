# \# Replication package: Bulk–Boundary Decomposition for Adaptive Windows in Spectral/HAC Estimation

# 

# This repository contains the code, environment information, and output structure needed to reproduce the computational results in the paper:

# 

# \*Bulk–Boundary Decomposition for Adaptive Windows in Spectral/HAC Estimation: Mechanism, Scale, and Certified Negativity\*

# 

# \## Contents

# 

# \- `paper/`: manuscript PDF

# \- `code/`: Python scripts for Monte Carlo simulations, tables, figures, and parameter-map computations

# \- `environment/`: package requirements and environment information

# \- `output/`: generated tables, figures, appendix outputs, and logs

# \- `replication\_metadata/`: seeds, run configuration, software versions, and commit hash

# 

# \## Main computational components

# 

# 1\. Monte Carlo evidence for HAC estimators under fixed and state-dependent truncations

# 2\. Figure and table generation

# 3\. Parameter map for the analytic family \\(H\_{\\alpha,\\beta}\\)

# 4\. Sparse certification for Appendix A

# 

# \## Software

# 

# Tested with:

# \- Python 3.x

# \- NumPy

# \- pandas

# \- SciPy

# \- matplotlib

# 

# See `environment/requirements.txt` for package versions.

# 

# \## Reproducibility

# 

# This package uses a fixed master seed and deterministic derived seeds by DGP, sample size, and replication.  

# A seed log is stored in `replication\_metadata/seeds\_master.csv`.

# 

# \## How to run

# 

# Open a terminal in the repository root.

# 

# \### 1. Create environment

# 

# ```bash

# pip install -r environment/requirements.txt

