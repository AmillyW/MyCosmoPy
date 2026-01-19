# MyCosmoPy

A Python library for cosmology computation, providing computational tools for studying cosmological structure formation, implementing both analytical frameworks and numerical simulations. This project bridges theoretical cosmology with practical computational methods.

**Still Under Development!**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active_Development-green)](https://github.com/AmillyW/MyCosmoPy)

## Features

### ✅ Implemented & Validated

This project follows a rigorous development roadmap aligned with my theoretical research.

- **Linear Growth Theory** (✅ Completed)

  - [X] Numerical solver for Linear Growth Factor ($D_+(z)$) and Growth Rate ($f(z)$) in various cosmological models.
  - [X] Validated against standard Boltzmann codes (CAMB) with $<1\%$ error.
- **Initial Conditions** (✅ Completed)

  - [X] Gaussian Random Field (GRF) generation using FFTW interfaces.
  - [X] Power Spectrum ($P(k)$) estimation and verification.
- **N-Body Simulation (Particle-Mesh)** (🚧 In Progress - will upload after testing is complete)

  - [X] Mass Assignment Schemes: Nearest Grid Point (NGP) & Cloud-in-Cell (CIC).
  - [ ] **Current Focus:** Implementing interlacing to suppress aliasing/shot noise.
  - [ ] Verify results against ``Quijote`` simulations for accuracy.
- **Advanced Theoretical Extensions** (📅 Planned for Feb 2026)

  - [ ] Standard Perturbation Theory (SPT) 1-loop power spectrum calculation.
  - [ ] Comparison between N-body non-linearities and SPT predictions.
  - [ ] Lagrangian Perturbation Theory (LPT) 1LPT/2LPT in the practice of N-body simulations.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/AmillyW/MyCosmoPy.git
cd MyCosmoPy

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```
