import tensorflow as tf
from typing import Callable, Tuple
import numpy as np

import tensorflow as tf
from typing import Callable, Tuple

def simulate_maxwell_2d_te(n_samples: int, init_function: Callable, boundary_function: Callable, 
                           epsilon: float, mu: float, x_start: float = 0.0, y_start: float = 0.0, 
                           length_x: float = 1.0, length_y: float = 1.0, time: float = 1.0, 
                           n_init: int = None, n_bndry: int = None, random_seed: int = 42, 
                           dtype: tf.DType = tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
                                                                 Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate Maxwell's equations in 2D TE mode with given initial and boundary conditions.

    Args:
        n_samples (int): Number of samples to generate for the equation.
        init_function (Callable): Function that returns the initial conditions for E and H fields.
        boundary_function (Callable): Function that returns the boundary conditions for E and H fields.
        epsilon (float): Permittivity of the medium.
        mu (float): Permeability of the medium.
        x_start (float, optional): Start of the domain in x-direction. Defaults to 0.0.
        y_start (float, optional): Start of the domain in y-direction. Defaults to 0.0.
        length_x (float, optional): Length of the domain in x-direction. Defaults to 1.0.
        length_y (float, optional): Length of the domain in y-direction. Defaults to 1.0.
        time (float, optional): Time frame of the simulation. Defaults to 1.0.
        n_init (int, optional): Number of initial samples to generate. Defaults to None.
        n_bndry (int, optional): Number of boundary samples to generate. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.DType, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
            Returns samples as (equation_samples, initial_samples, boundary_samples).
    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples

    # Generate grid points (x, y, t) for equation samples
    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), x_start, x_start + length_x, dtype=dtype, seed=random_seed)
    y = tf.random.uniform((n_samples, 1), y_start, y_start + length_y, dtype=dtype, seed=random_seed)
    tx_samples = tf.concat((t, x, y), axis=1)
    
    # Placeholder for the results
    y_eqn = tf.zeros((n_samples, 1), dtype=dtype)

    # Initial samples (t=0)
    t_init = tf.zeros((n_init, 1), dtype=dtype)
    x_init = tf.random.uniform((n_init, 1), x_start, x_start + length_x, dtype=dtype, seed=random_seed)
    y_init = tf.random.uniform((n_init, 1), y_start, y_start + length_y, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init, y_init), axis=1)
    y_init = init_function(tx_init)  # Initial E and H fields

    # Boundary samples
    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary = tf.concat([tf.ones((n_bndry // 2, 1), dtype=dtype) * x_start, 
                            tf.ones((n_bndry // 2, 1), dtype=dtype) * (x_start + length_x)], axis=0)
    y_boundary = tf.concat([
        tf.random.uniform((n_bndry // 2, 1), y_start, y_start + length_y, dtype=dtype, seed=random_seed),
        tf.random.uniform((n_bndry // 2, 1), y_start, y_start + length_y, dtype=dtype, seed=random_seed)
    ], axis=0)
    tx_boundary = tf.concat((t_boundary, x_boundary, y_boundary), axis=1)
    y_boundary = boundary_function(tx_boundary)

    # Return all samples in the expected format
    return (tx_samples, y_eqn), (tx_init, y_init), (tx_boundary, y_boundary)



