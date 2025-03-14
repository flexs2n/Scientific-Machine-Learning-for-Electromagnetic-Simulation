<p align="center">
  <a href="https://github.com/[YourUsername]/BEng_project/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python version"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://github.com/[YourUsername]/BEng_project/blob/main/CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome"></a>
  <br>
  <a href="https://github.com/[YourUsername]/BEng_project/stargazers"><img src="https://img.shields.io/github/stars/[YourUsername]/BEng_project.svg?style=social&label=Star" alt="GitHub stars"></a>
  <a href="https://github.com/[YourUsername]/BEng_project/network/members"><img src="https://img.shields.io/github/forks/[YourUsername]/BEng_project.svg?style=social&label=Fork" alt="GitHub forks"></a>
  <a href="https://github.com/[YourUsername]/BEng_project/issues"><img src="https://img.shields.io/github/issues/[YourUsername]/BEng_project.svg" alt="GitHub issues"></a>
  <a href="https://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <a href="https://github.com/[YourUsername]/BEng_project/wiki"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation Status"></a>
</p>

# BEng Project: Evaluating NVIDIA Modulus for 2D Electromagnetic Field Simulations

## Table of Contents
- [Introduction](#-introduction)
- [Why NVIDIA Modulus?](#-why-nvidia-modulus)
- [Features & Capabilities](#-features--capabilities)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Community](#-community)
- [License](#-license)

---

## 📖 Introduction
This repository contains the work for a 4th-year BEng project aimed at evaluating the NVIDIA Modulus framework for simulating 2D electromagnetic fields, specifically focusing on transverse magnetic (TM) and transverse electric (TE) modes. The project compares the performance of Modulus, a physics-informed neural network (PINN) framework, against the classical Finite Difference Time Domain (FDTD) method. The primary focus is on understanding the trade-offs between computational efficiency and solution accuracy as the simulation domain size increases, particularly in domains with inhomogeneous material properties.

---

## 🤔 Why NVIDIA Modulus?
Electromagnetic simulations are critical for designing and optimizing devices in fields such as telecommunications, radar systems, and photonics. Traditional numerical methods like FDTD are widely used but can become computationally expensive for large-scale or complex problems. NVIDIA Modulus offers a promising alternative by leveraging deep learning to solve partial differential equations (PDEs) like Maxwell's equations. This project explores the potential of Modulus for 2D electromagnetic simulations, comparing its accuracy, stability, and computational efficiency with FDTD.

---

## 🚀 Features & Capabilities
- **Physics-Informed Neural Networks (PINNs)**: Solve Maxwell's equations using deep learning.
- **Comparison with FDTD**: Benchmark accuracy and computational efficiency against classical methods.
- **Scalability Analysis**: Evaluate performance as domain size and complexity increase.
- **Inhomogeneous Material Support**: Simulate domains with varying permittivity and permeability.
- **Flexible Setup**: Easily define complex geometries, boundary conditions, and material properties.

## Background
Electromagnetic simulations are critical for designing and optimizing devices in fields such as telecommunications, radar systems, and photonics. Traditional numerical methods like FDTD are widely used but can become computationally expensive for large-scale or complex problems. NVIDIA Modulus offers a promising alternative by leveraging deep learning to solve partial differential equations (PDEs) like Maxwell's equations. This project explores the potential of Modulus for 2D electromagnetic simulations, comparing its accuracy, stability, and computational efficiency with FDTD.

---

## Objectives
1. **Compare Accuracy and Stability of Modulus for TM and TE Fields**  
   - Simulate electromagnetic wave propagation in 2D using Modulus and FDTD.  
   - Evaluate accuracy and temporal evolution of field values (electric and magnetic fields).  
   - Analyze how accuracy scales with increasing domain size.

2. **Benchmark Computational Efficiency with Increasing Domain Sizes**  
   - Measure training time, memory consumption, model convergence, and inference time for Modulus.  
   - Compare computational efficiency with FDTD for small to large-scale domains.  
   - Identify scenarios where Modulus offers advantages over FDTD.

3. **Analyze Performance in Domains with Inhomogeneous Material Properties**  
   - Simulate wave propagation in domains with varying permittivity and permeability.  
   - Compare Modulus and FDTD in handling material inhomogeneities as domain size increases.

4. **Explore Flexibility and Usability of the Modulus Framework**  
   - Evaluate the ease of setting up simulations in Modulus, including defining geometries, boundary conditions, and material properties.  
   - Identify areas where Modulus simplifies problem setup compared to FDTD.

---

## Desirable Outcomes
1. **Detailed Comparison of Accuracy Across Domain Sizes**  
   - Quantitative metrics (e.g., L2 error) comparing Modulus and FDTD results.  
   - Visualizations of field distributions for TM and TE modes in homogeneous and inhomogeneous media.

2. **Scalability Analysis of Computational Efficiency**  
   - Report on computational resources (GPU/CPU usage, memory consumption) and time required by Modulus vs. FDTD.  
   - Insights into scaling behavior and efficiency trade-offs.

3. **Case Studies of Inhomogeneous Domains**  
   - Test cases with obstacles, inclusions, or varying dielectric properties.  
   - Benchmark library for future validation of deep learning-based frameworks.

4. **Conclusions & Recommendations**  
   - Practical insights into adopting Modulus for large-scale electromagnetic simulations.  
   - Recommendations for scenarios where Modulus offers significant advantages.

---

## Resources
- **Simulation Frameworks**:  
  - NVIDIA Modulus for deep learning-based simulations.  
  - Classical FDTD codebase (Python or MATLAB) for comparison.  
- **Computational Resources**:  
  - GPU resources for training Modulus models.  
  - CPU resources for running FDTD simulations.

---

## Suggested Work Plan
1. **Phase 1: Literature Review & Setup**  
   - Review NVIDIA Modulus documentation and research on deep learning for Maxwell's equations.  
   - Set up a basic FDTD simulation code for TM and TE modes.  
   - Prepare test cases for small and large homogeneous and inhomogeneous domains.

2. **Phase 2: Modulus Training & Validation**  
   - Implement electromagnetic problem setups in Modulus and train models for TM and TE fields.  
   - Validate results against FDTD solutions across different domain sizes.  
   - Adjust training parameters to achieve stable and accurate results.

3. **Phase 3: Benchmarking & Analysis**  
   - Conduct systematic comparisons of computational performance and accuracy between Modulus and FDTD.  
   - Analyze behavior in handling larger domains with complex material interfaces.

4. **Phase 4: Report & Presentation**  
   - Compile results, generate visualizations, and create a comprehensive report.  
   - Prepare a presentation summarizing findings and recommendations.

---

## Code Structure
The repository is organized as follows:
- **/FDTD**: Contains the implementation of the classical FDTD method for TM and TE modes.  
- **/Modulus**: Includes scripts for setting up and training PINNs using NVIDIA Modulus.  
- **/CaseStudies**: Test cases for homogeneous and inhomogeneous domains.  
- **/Results**: Outputs, visualizations, and performance metrics.  
- **/Documentation**: Project report, presentation, and supplementary materials.

---

## Daybook Notes
### PINN for 2D Maxwell's Equations in TE Mode
- **Attributes**:  
  - `backbone`: Neural network model for approximating field solutions.  
  - `_loss_residual_weight`: Weight of the residual loss.  
  - `_loss_boundary_weight`: Weight of the boundary loss.  
  - `loss_residual_tracker`, `loss_boundary_tracker`, `mae_tracker`: Trackers for losses and errors.  

- **Initialization**:  
  - `backbone`: Neural network backbone model.  
  - `loss_residual_weight`: Weight of the residual loss.  
  - `loss_boundary_weight`: Weight of the boundary loss.  

- **Forward Pass**:  
  - Computes electric field, magnetic field residuals, and boundary solutions.  

- **Metrics**:  
  - Tracks residual loss, boundary loss, and mean absolute error.  

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or collaborations, please contact:  
Balaji Anandganesh
ba-1@sms.ed.ac.uk 
The University of Edinburgh
