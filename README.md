# 🔥 Hotspot Detection in a Tubular Reactor using Physics-Informed Neural Networks (PINNs)

## Overview

This project was developed as part of my **Introduction to AI** course during my third year of Chemical Engineering. It demonstrates the application of **Physics-Informed Neural Networks (PINNs)** for estimating the temperature distribution and detecting hotspot locations in a one-dimensional tubular reactor.

Unlike conventional neural networks, PINNs incorporate the governing physical laws directly into the training process, enabling accurate predictions even when only limited sensor measurements are available.

---

## Problem Statement

Packed-bed and tubular reactors carrying exothermic reactions often develop localized **temperature hotspots**. These hotspots are critical because they can:

- Cause catalyst deactivation
- Reduce reactor efficiency
- Lead to thermal runaway
- Increase operational safety risks

Since placing sensors directly inside the hotspot region is often impractical or unsafe, only sparse temperature measurements are typically available.

The objective of this project is to reconstruct the complete temperature profile using limited sensor data while enforcing the reactor's governing energy balance equation through a Physics-Informed Neural Network.

---

## Governing Equation

The reactor is modeled using the steady-state one-dimensional energy balance equation:

\[
u \frac{dT}{dz}=\frac{k}{\rho C_p}\frac{d^2T}{dz^2}+\frac{-\Delta H_r}{\rho C_p}r_A(z)\]

where the reaction heat source is represented by a **Gaussian function** to simulate a localized hotspot.

### Assumptions

- One-dimensional axial model
- Plug flow reactor
- Steady-state operation
- Negligible radial temperature gradients
- Constant thermophysical properties

### Boundary Conditions

- Fixed inlet temperature (Dirichlet condition)
- Zero temperature gradient at the outlet (Neumann condition)

---

## Project Objectives

- Model the reactor temperature profile
- Generate physics-consistent synthetic data
- Train a Physics-Informed Neural Network
- Reconstruct the complete temperature distribution
- Predict hotspot location using sparse sensor measurements
- Compare PINN predictions with numerical solutions

---

## Data Generation

The governing boundary value problem (BVP) is solved using **SciPy's `solve_bvp()`** to obtain the reference temperature profile.

The generated data is then processed by:

- Selecting sparse temperature sensor locations
- Excluding sensors from the hotspot region
- Adding Gaussian noise to simulate real experimental measurements
- Creating collocation points for enforcing the governing PDE

---

## PINN Architecture

The model is implemented using **TensorFlow**.

**Network Configuration**

- Fully Connected Feedforward Neural Network
- 3 Hidden Layers
- 64 Neurons per Hidden Layer
- **tanh** Activation Function

**Input**

- Axial position (z)

**Output**

- Temperature T(z)

Automatic differentiation is used to compute spatial derivatives required for evaluating the PDE residual.

---

## Training Methodology

The PINN is trained by minimizing a weighted loss function consisting of:

- Data Loss (sensor measurements)
- PDE Residual Loss (energy balance equation)
- Boundary Condition Loss (Dirichlet & Neumann conditions)

The network is optimized using the **Adam Optimizer**, while collocation points sampled throughout the reactor ensure that the governing physics are satisfied.

---

## Technologies Used

- Python
- TensorFlow
- SciPy
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Results

The trained PINN successfully:

- Reconstructed the reactor temperature profile
- Learned the governing physical behavior from sparse data
- Satisfied the PDE and boundary conditions
- Predicted the hotspot location with good accuracy
- Demonstrated the effectiveness of Scientific Machine Learning for inverse engineering problems

The accuracy of hotspot prediction depends on factors such as sensor placement, measurement noise, and model parameter scaling.

---

## Project Structure

```text
PINN-Hotspot-Detection/
│
├── PINN_GROUP17_CODE_FINAL.ipynb
├── Presentation.pptx
├── Report.pdf
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install the required dependencies

```bash
pip install tensorflow scipy numpy matplotlib
```

3. Launch Jupyter Notebook

```bash
jupyter notebook
```

4. Open `PINN_GROUP17_CODE_FINAL.ipynb`

5. Run all cells to:

- Solve the governing BVP
- Generate synthetic sensor data
- Train the PINN
- Predict the reactor temperature profile
- Detect the hotspot location
- Visualize the results

---

## Skills Demonstrated

- Physics-Informed Neural Networks (PINNs)
- Scientific Machine Learning (SciML)
- Deep Learning
- TensorFlow
- Numerical Methods
- Differential Equations
- Heat Transfer
- Chemical Reaction Engineering
- Scientific Computing
- Python Programming

---

## Future Improvements

- Extend the model to two-dimensional reactors
- Incorporate transient (time-dependent) heat transfer
- Use real experimental temperature data
- Simultaneously predict concentration and temperature fields
- Compare PINN performance with finite element and finite difference methods
- Optimize sensor placement for improved hotspot detection

---

## Author

**Badal Sakharkar**

Chemical Engineering Undergraduate, VNIT Nagpur

Interested in **AI for Engineering, Scientific Machine Learning, Data Analytics, and Process Modeling.**
