# Random Projections for Semidefinite Programming and Polynomial Optimization

This repository contains the implementation of methods and experiments related to using **random projections** for approximating semidefinite programming (SDP) problems and polynomial optimization problems. These methods aim to reduce the size of optimization problems, making them computationally efficient without significant loss of accuracy.


## Overview

Random projection is a dimensionality reduction technique that has gained prominence in recent years. This project explores its application to SDP problems, polynomial optimization, and adversarial machine learning. It proposes a variable projection method for general SDPs, a tailored approach for polynomial optimization problems, and demonstrates its utility in combinatorial and bilevel optimization problems.

Key highlights of the project include:

- Sparse sub-Gaussian random projections to approximate SDPs by reducing matrix variable size.
- Theoretical bounds on feasibility and optimality error. 
- Applications in:
  - Semidefinite relaxations of non-convex quadratically constrained quadratic problems (QCQP).
  - The max-cut problem.
  - The maximum satisfiability problem (Max-2-SAT).
  - Poisoning attack problems in adversarial machine learning.
  - Polynomial optimization problems such as optimization over the unit sphere and stable set problems.


## Repository Structure

| File | Description |
|------|-------------|
| `.gitignore` | Specifies files and directories to ignore in version control. |
| `README.md` | This documentation file. |
| `combinatorics_plot.py` | Script to generate plots related to combinatorial optimization results. |
| `computational_experiments.py` | Main script for running computational experiments for SDPs and optimization problems. |
| `config.yml` | Configuration file for experiments. |
| `cordones.m` | MATLAB script related to generate graphs. |
| `first_level_stable_set.py` | Implementation of the first-level relaxation for stable set problems. |
| `general_sdp.py` | General implementation of the random projection method for SDPs. |
| `generate_graphs.py` | Utility script for generating random graphs for experiments. |
| `maxcut.py` | Implementation for solving the max-cut problem using SDP relaxations. |
| `maxsat.py` | Script for solving the maximum satisfiability (Max-2-SAT) problem using SDP relaxations. |
| `monomials.py` | Monomial generation and manipulation for polynomial optimization problems. |
| `optimization_unit_sphere.py` | Implementation of optimization over the unit sphere with random projections. |
| `plot_spectrahedron.py` | Visualization script for spectrahedrons. |
| `polynomial_generation.py` | Utility for generating polynomial optimization problems with configurable density. |
| `process_DIMACS_data.py` | Preprocessing script for DIMACS datasets. |
| `process_graphs.py` | Graph preprocessing utility. |
| `process_max2sat.py` | Preprocessing script for Max-2-SAT datasets. |
| `random_projections.py` | Core implementation of random projection techniques. |
| `random_qcqp.py` | Script for running QCQP experiments with random projections. |
| `results_to_latex.py` | Converts experimental results into LaTeX tables. |
| `results_to_plots.py` | Generates plots from experimental results. |
| `second_level_stable_set.py` | Second-level relaxation for stable set problems. |
| `setup_python_env.sh` | Script to set up the Python environment for this project. |


## Features

### General Methods:
- Sparse random projection for reducing SDP variable size.

### Polynomial Optimization:
- Tailored projection methods for sum-of-squares SDP relaxations.
- Constraint aggregation techniques.
- Computational experiments for optimization over the unit sphere and stable set problems.

### Applications:
- Semidefinite relaxations for Max-2-SAT and max-cut problems.
- Bilevel polynomial optimization.
- Poisoning attack problems in adversarial machine learning.


## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Set up the Python environment:
   ```bash
   bash setup_python_env.sh
   ```
or

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

- To run computational experiments for a specific problem, use the corresponding script. For example:
  ```bash
  python computational_experiments.py
  ```

- For visualizing results:
  ```bash
  python results_to_plots.py
  ```


## Results

### Key Findings:
- Random projections can effectively approximate SDP problems as long as the number of constraints is manageable.
- Combined constraint aggregation and projection methods yield near-optimal solutions with reduced computational effort.
- Applications in polynomial optimization show promising results for solving challenging problems such as stable set and bilevel optimization.


