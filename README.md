# ML-RTE-Solver: A Machine Learning Approach to Radiative Transfer Equation

This repository contains the official implementation of the paper **"A Machine Learning Approach to Solving Radiative Transfer Equation in Participating Media"**.

## Contributors



## Overview

The Radiative Transfer Equation (RTE) is a fundamental tool for describing light-matter interactions but is notoriously computationally expensive to solve using conventional numerical methods. This project introduces a Machine Learning based framework that combines Residual Networks (ResNet) and Physics-Informed Neural Networks (PINNs) to solve both **forward** and **inverse** RTE problems.
**To launch the demo locally, run: ```streamlit run gui.py``` . Alternatively, you can use the hosted version here: https://nightglow0826-ml-rte-solver-gui-ycpfhy.streamlit.app/ to use it online**.

**Key Capabilities:**
* **Forward Problem:** Predicts transmittance and reflectance spectra from optical properties (optical depth $\tau$, albedo $\omega$).
* **Inverse Problem - Optical Propertis:** Retrieves single-particle level optical properties from spectral responses.
* **Incerse Problem - Material Design:** Implements a post-inverse process using the Nelder-Mead simplex method to map optical properties to material parameters (e.g., particle radius, mass density), enabling inverse material design.
* **Performance:** Achieves a **$10^5 \times$** speedup for forward& finding optical properties and a **$10^2 \times$** speedup for finding material properties compared to conventional RTE solvers.

## Repository Structure

* **`RTE_Truth_Model.py`**: The conventional C-RTE solver (Ground Truth). Implements the discrete ordinate method (Gauss-Legendre quadrature) and finite difference scheme to solve the RTE.
* **`NN.py`**: Implementation of the Forward ML-RTE solver. Contains the `ResNet' architecture and the training loop with PINN loss functions (boundary loss, energy conservation, and analytical constraints).
* **`InverseNN.py`**: Implementation of the Inverse ML-RTE solver. Maps spectral data (Transmittance, Reflectance) back to optical properties.
* **`gen_data.py`**: Scripts for data generation. Uses parallel computing to generate large datasets of optical property pairs and their corresponding spectral responses using the C-RTE solver.
* **`spectrum.py`**: The main application script for **Inverse Material Design**. It integrates Mie theory (using `pymiescatt` or internal logic) with the ML solver to optimize material properties (density, radius) for target spectra, such as those for silica aerogels.

## Installation

### Prerequisites
The code requires Python 3.8+ and the following dependencies.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/NightGlow0826/ML-RTE-Solver.git](https://github.com/NightGlow0826/ML-RTE-Solver.git)
    cd ML-RTE-Solver
    ```
    

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Requirements
Based on `requirements.txt`:
```text
brewer2mpl==1.4.1
joblib==1.5.2
matplotlib==3.10.7
numpy==2.2
pymiescatt==1.8.1.1
scipy==1.13
streamlit==1.51.0
torch==2.9.1
tqdm==4.67.1
tensorboard
```
## Citation

If you find this repository useful, please cite:
```text
@inproceedings{gan2025paper,
  title   = {A Machine Learning Approach to Solving Radiative Transfer Equation in Participating Media},
  author  = {Yuyang Gan, Yang Zhong, Xuanjie Wang, Lenan Zhang*},
  booktitle = {},
  year    = {2025},
  url     = {https://github.com/NightGlow0826/ML-RTE-Solver}
}
```
