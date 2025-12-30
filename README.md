Hotspot Detection in a Tubular Reactor using PINNs

This repository contains my final-year project on detecting a temperature hotspot in a 1D tubular reactor using a Physics-Informed Neural Network (PINN).
The aim of the project is to reconstruct the temperature profile inside the reactor using limited temperature sensor data while enforcing the governing energy balance equation.

Problem Description

In tubular reactors carrying exothermic reactions, localized high-temperature regions (hotspots) can form.
Direct placement of sensors inside the hotspot region is often unsafe, so temperature measurements are usually sparse and taken away from the hotspot.

The objective of this project is to:

Reconstruct the full temperature field along the reactor length

Identify the hotspot location

Use physical laws to regularize the learning process when data is limited

Governing Equation

The steady-state one-dimensional energy balance is considered:

u dT/dz = (k / ρCp) d²T/dz² + (−ΔHr / ρCp) rA(z)

The reaction heat source is modeled as a Gaussian function to represent a localized hotspot.
Boundary conditions used are:

Fixed inlet temperature (Dirichlet condition)

Zero temperature gradient at the outlet (Neumann condition)

Data Generation

A boundary value problem (BVP) corresponding to the energy equation is solved using SciPy’s solve_bvp function.
This provides a physics-consistent temperature profile along the reactor length.

From this solution:

A small number of temperature sensors are selected

Sensors are placed outside the hotspot region

Gaussian noise is added to simulate realistic measurements

PINN Architecture

The Physics-Informed Neural Network is implemented using TensorFlow.

Fully connected feedforward neural network

3 hidden layers

64 neurons per hidden layer

tanh activation function

Input: axial position (z)

Output: temperature T(z)

Automatic differentiation is used to compute spatial derivatives required for the PDE residual.

Training Methodology

The network is trained by minimizing a weighted loss function consisting of:

Data loss from sensor measurements

PDE residual loss enforcing the energy balance

Boundary condition losses at inlet and outlet

The Adam optimizer is used for training.
Collocation points are sampled throughout the domain to enforce the governing equation.

Results

The trained PINN is able to:

Reconstruct the temperature distribution along the reactor

Satisfy the governing physics and boundary conditions

Estimate the hotspot location from the reconstructed temperature field

The accuracy of hotspot detection depends on sensor placement, noise level, and parameter scaling.
