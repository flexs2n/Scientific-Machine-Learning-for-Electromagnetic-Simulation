# BEng_project
BEng 4th year project


# Background: 
This project aims to evaluate the potential of the NVIDIA Modulus framework for simulating 2D electromagnetic fields, specifically focusing on transverse magnetic (TM) and transverse electric (TE) modes. Modulus is a physics-informed neural network (PINN) framework designed to solve partial differential equations (PDEs) through deep learning. We will compare its performance against the classical Finite Difference Time Domain (FDTD) method, a standard numerical technique for solving Maxwell's equations in time-domain simulations.

A key focus of this evaluation is the balance between computational efficiency and solution accuracy as the finite dimensions of the simulation domain increase. This includes exploring how Modulus and FDTD perform in larger domains with complex configurations, particularly when these domains contain multiple inhomogeneities in their electromagnetic properties, such as varying permittivity and permeability. The goal is to understand how agile and scalable the Modulus framework is compared to traditional methods.

# Objectives:
A). Compare Accuracy and Stability of Modulus for TM and TE Fields: Simulate electromagnetic wave propagation in 2D using the Modulus framework, focusing on both transverse magnetic (TM) and transverse electric (TE) field configurations, and compare the results with those obtained using the classical FDTD method, particularly examining the accuracy of field values (e.g., electric and magnetic field distributions) and their temporal evolution. Evaluate how the accuracy of the Modulus framework scales as the domain size increases, identifying any limitations or advantages over FDTD in larger problem domains.

B). Benchmark Computational Efficiency with Increasing Domain Sizes: Assess the computational performance of the Modulus framework as the size of the domain increases, measuring metrics such as training time, memory consumption, model convergence, and inference time. Compare the computational efficiency of Modulus to that of the FDTD method for solving electromagnetic problems of varying sizes, from small test domains to larger-scale simulations. Evaluate how the computational demand of Modulus scales with domain complexity and identify cases where it offers a computational advantage over FDTD, such as when simulating domains with complex geometries or high-resolution features.

C). Analyse Performance in Domains with Inhomogeneous Material Properties: Simulate wave propagation through domains containing multiple inhomogeneous regions with varying electromagnetic properties (e.g., regions with different permittivity and permeability), and compare how Modulus and FDTD handle these inhomogeneous materials as the size of the domain increases. Should obtain an understanding on the impact of domain size on the ability of Modulus to learn and adapt to these material variations compared to the classical numerical approach.

D). Explore Flexibility and Usability of the Modulus Framework: Evaluate the ease of setting up and configuring simulations within the Modulus framework, including defining complex geometries, boundary conditions, and material properties, especially as the domain complexity increases. Identify areas where Modulus may simplify problem setup or enable rapid iterations compared to traditional FDTD methods, particularly in scenarios requiring large domain simulations.

# Desirable Outcomes

1. Detailed Comparison of Accuracy Across Domain Sizes: 
   - Generate quantitative metrics comparing the field values obtained through Modulus and FDTD, including error norms (e.g., L2 error) and field distributions over time and space, as a function of domain size.
   - Provide visualizations of electromagnetic field distributions for TM and TE fields in both homogeneous and inhomogeneous media, highlighting any discrepancies between the methods, especially as domain size grows.

2. Scalability Analysis of Computational Efficiency:
   - A comprehensive report on computational resources (GPU/CPU usage, memory consumption) and time required by Modulus to achieve a desired level of accuracy, compared to FDTD, for a range of domain sizes.
   - Identification of the scaling behavior of Modulus relative to FDTD, highlighting scenarios where Modulus maintains accuracy with reduced computational effort or, conversely, where FDTD remains more efficient.
   - Insights into how the computational efficiency of both methods is affected by the presence of high-contrast material interfaces in larger domains.

3. Case Studies of Inhomogeneous Domains:
   - Simulate a set of test cases where the domain contains obstacles, inclusions, or regions with varying dielectric properties, and analyze how Modulus’s accuracy and efficiency scale with domain size compared to FDTD.
   - Create a benchmark library of simulations for electromagnetic wave propagation in complex, large-scale domains that can be used for future validation of other deep learning-based frameworks.

4. Conclusions & Recommendations:
   - Provide insights into the practicality of adopting deep learning frameworks like Modulus for large-scale electromagnetic simulations in practical applications.
   - Identify scenarios where Modulus's computational scaling could offer significant advantages, such as in rapid prototyping of electromagnetic devices, optimizing materials over large domains, or simulations where iterative design adjustments are critical.

# Resources

- Simulation Frameworks: NVIDIA Modulus (for deep learning-based simulations) and a classical FDTD codebase (e.g., implemented in Python or MATLAB) for comparison.
- Computational Resources: Access to GPU resources for training the Modulus models, as well as CPU resources for running FDTD simulations on increasing domain sizes.

# Suggested work plan 

1. Phase 1: Literature Review & Setup:
   - Review relevant documentation for NVIDIA Modulus and research on using deep learning for solving Maxwell’s equations.
   - Set up a basic FDTD simulation code for TM and TE modes.
   - Prepare test cases for small and large homogeneous and inhomogeneous domain simulations.

2. Phase 2: Modulus Training & Validation:
   - Implement the electromagnetic problem setups in Modulus and train models for TM and TE fields.
   - Validate the results against FDTD solutions across different domain sizes and adjust training parameters as needed.
   - Focus on achieving stable and accurate results for a range of domain conditions and sizes.

3. Phase 3: Benchmarking & Analysis:
   - Conduct a systematic comparison of computational performance and accuracy between Modulus and FDTD as domain size increases.
   - Analyze the behavior of both methods in handling larger domains with complex material interfaces.

4. Phase 4: Report & Presentation:
   - Compile the results, generate visualizations, and create a comprehensive report with emphasis on computational efficiency vs. accuracy trade-offs.
   - Prepare a presentation summarizing findings, with recommendations for the use of deep learning in large-scale electromagnetic simulations.
   - 

   DayBook:
   Notes: A PINN for the 2D Maxwell's equations in the Transverse Electric (TE) mode.

    Attributes:
        backbone: The backbone neural network model that approximates the field solutions.
        _loss_residual_weight: The weight of the residual loss.
        _loss_boundary_weight: The weight of the boundary loss.
        loss_residual_tracker: Tracker for residual loss.
        loss_boundary_tracker: Tracker for boundary loss.
        mae_tracker: Tracker for mean absolute error.

   Initializes the Maxwell2DPinn model.
   Args:
            backbone: The neural network backbone model.
            loss_residual_weight: Weight of the residual loss.
            loss_boundary_weight: Weight of the boundary loss.

   Performs a forward pass and computes the fields and residuals for Maxwell's equations.
        Args:
            inputs: A tuple containing field samples and boundary samples.
            training: Boolean indicating training mode.
        Returns:
            Tuple of electric field, magnetic field residuals, and boundary solution.

   Returns the metrics to track.
        Sets the loss weights.

    Args:
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
    
