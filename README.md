# BEng_project
BEng 4th year project


Background: This project aims to evaluate the potential of the NVIDIA Modulus framework for simulating 2D electromagnetic fields, specifically focusing on transverse magnetic (TM) and transverse electric (TE) modes. Modulus is a physics-informed neural network (PINN) framework designed to solve partial differential equations (PDEs) through deep learning. We will compare its performance against the classical Finite Difference Time Domain (FDTD) method, a standard numerical technique for solving Maxwell's equations in time-domain simulations.

A key focus of this evaluation is the balance between computational efficiency and solution accuracy as the finite dimensions of the simulation domain increase. This includes exploring how Modulus and FDTD perform in larger domains with complex configurations, particularly when these domains contain multiple inhomogeneities in their electromagnetic properties, such as varying permittivity and permeability. The goal is to understand how agile and scalable the Modulus framework is compared to traditional methods.

Objectives
A). Compare Accuracy and Stability of Modulus for TM and TE Fields: Simulate electromagnetic wave propagation in 2D using the Modulus framework, focusing on both transverse magnetic (TM) and transverse electric (TE) field configurations, and compare the results with those obtained using the classical FDTD method, particularly examining the accuracy of field values (e.g., electric and magnetic field distributions) and their temporal evolution. Evaluate how the accuracy of the Modulus framework scales as the domain size increases, identifying any limitations or advantages over FDTD in larger problem domains.

B). Benchmark Computational Efficiency with Increasing Domain Sizes: Assess the computational performance of the Modulus framework as the size of the domain increases, measuring metrics such as training time, memory consumption, model convergence, and inference time. Compare the computational efficiency of Modulus to that of the FDTD method for solving electromagnetic problems of varying sizes, from small test domains to larger-scale simulations. Evaluate how the computational demand of Modulus scales with domain complexity and identify cases where it offers a computational advantage over FDTD, such as when simulating domains with complex geometries or high-resolution features.

C). Analyse Performance in Domains with Inhomogeneous Material Properties: Simulate wave propagation through domains containing multiple inhomogeneous regions with varying electromagnetic properties (e.g., regions with different permittivity and permeability), and compare how Modulus and FDTD handle these inhomogeneous materials as the size of the domain increases. Should obtain an understanding on the impact of domain size on the ability of Modulus to learn and adapt to these material variations compared to the classical numerical approach.

