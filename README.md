# 1D Transient Steam Flow Simulator with Conjugate Heat Transfer

A 1D transient fluid flow simulator for steam piping networks, fully compliant with the IAPWS-IF97 international standard for steam properties. It solves a conjugate heat transfer problem that accounts for pipe wall thermal capacity, insulation (glass wool), and heat dissipation to the ambient environment using a fully implicit numerical method.

## 1. Governing Equations

This solver models 1D compressible fluid flow by solving the conservation laws (modified Euler equations) using the Finite Difference Method (FDM).

### Mass Conservation (Continuity Equation)
$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0$$

### Momentum Equation
$$\frac{\partial (\rho uQT}{\partial t} + \frac{\partial (\rho u^2 + P)}{\partial x} + f_{\text{loss}} = 0$$
*Note: The friction loss term $f_{\text{loss}}$ is dynamically calculated using the friction factor derived from Haaland's equation, accounting for the pipe's internal roughness.*

### Energy Equation
$$\frac{\partial (\rho E)}{\partial t} + \frac{\partial (\rho u H)}{\partial x} = \frac{Q_{\text{in}}}{A}$$
*Note: Total internal energy is defined as $E = u_{\text{int}} + \frac{1}{2}u^2$, and total enthalpy is defined as $H = h + \frac{1}{2}u^2$.*

## 2. Features

- **High-Precision Property Calculation**: Embedded with the `iapws` library to dynamically update density $\rho$, temperature $T$, viscosity $\mu$, and Prandtl number $Pr$ at each time step based on local pressure $P$ and enthalpy $h$.
- **4-Layer Thermal Resistance Model**: Simulates transient thermal behavior across a conjugate network: Steam boundary $\rightarrow$ Internal convective heat transfer $\rightarrow$ Steel pipe wall thermal capacity $\rightarrow$ Insulation thermal conduction $\rightarrow$ Ambient convective heat dissipation.
- **Robust Numerical Stability**: Employs a fully implicit method via `scipy.optimize.fsolve` to simultaneously solve the non-linear residual equations across all spatial nodes, ensuring robust convergence.

## 3. Future Work

- [ ] Add interactive and animated visualizations of simulation results using Manim / Plotly.
- [ ] **Extend to Physics-Informed Neural Networks (PINNs)** (utilizing this physical solver to generate ground truth training data and validate physics-based loss constraints).
