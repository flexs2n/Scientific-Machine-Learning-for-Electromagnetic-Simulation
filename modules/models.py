'''
This file contains the PINN models for the Advection, Burgers, Schrodinger, Poisson, Heat, and Wave equations.
'''

from typing import Tuple, List, Union, Callable
import tensorflow as tf
import numpy as np
import sys

LOSS_TOTAL = "loss_total"
LOSS_BOUNDARY = "loss_boundary"
LOSS_INITIAL = "loss_initial"
LOSS_RESIDUAL = "loss_residual"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"

def create_history_dictionary() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_BOUNDARY: [],
        LOSS_INITIAL: [],
        LOSS_RESIDUAL: [],
        MEAN_ABSOLUTE_ERROR: []
    }

def create_dense_model(layers: List[Union[int, "tf.keras.layers.Layer"]], activation: "tf.keras.activations.Activation", \
    initializer: "tf.keras.initializers.Initializer", n_inputs: int, n_outputs: int, **kwargs) -> "tf.keras.Model":
    """
    Creates a dense model with the given layers, activation, and input and output sizes.

    Args:
        layers: The layers to use. Elements can be either an integer or a Layer instance. If an integer, a Dense layer with that many units will be used.
        activation: The activation function to use.
        initializer: The initializer to use.
        n_inputs: The number of inputs.
        n_outputs: The number of outputs.
        **kwargs: Additional arguments to pass to the Model constructor.

    Returns:
        The dense model.
    """
    inputs = tf.keras.Input(shape=(n_inputs,))
    x = inputs
    for layer in layers:
        if isinstance(layer, int):
            x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer=initializer)(x)
        else:
            x = layer(x)
    outputs = tf.keras.layers.Dense(n_outputs, kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)

class WavePinn(tf.keras.Model):
    """
    A model that solves the wave equation.
    """

    def __init__(self, backbone: "tf.keras.Model", c: float, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            c: The wave speed.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.c = c
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)

            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
        second_order = tape2.batch_jacobian(first_order, tx_samples)
        d2u_dt2 = second_order[..., 0, 0]
        d2u_dx2 = second_order[..., 1, 1]
        lhs_samples = d2u_dt2 - (self.c ** 2) * d2u_dx2

        u_bnd = self.backbone(tx_bnd, training=training)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(tx_init)
            u_initial = self.backbone(tx_init, training=training)
        du_dt_init = tape.batch_jacobian(u_initial, tx_init)[..., 0]

        return u_samples, lhs_samples, u_initial, du_dt_init, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. The outputs are the exact solutions for the samples, \
                the exact rhs for the samples, the exact solution for the initial, the exact derivative for the initial, \
                and the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, du_dt_init_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, du_dt_init, u_bnd = self(inputs, training=True)
            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial_neumann = self.init_loss(du_dt_init_exact, du_dt_init)
            loss_initial_dirichlet = self.init_loss(u_initial_exact, u_initial)
            loss_initial = loss_initial_neumann + loss_initial_dirichlet
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.
        """
        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, du_dt_init_exact, u_bnd_exact = outputs

        u_samples, lhs_samples, u_initial, du_dt_init, u_bnd = self(inputs, training=False)
        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial_neumann = self.init_loss(du_dt_init_exact, du_dt_init)
        loss_initial_dirichlet = self.init_loss(u_initial_exact, u_initial)
        loss_initial = loss_initial_neumann + loss_initial_dirichlet
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]

class Maxwell2DPinn(tf.keras.Model):
    """
    A PINN for the 2D Maxwell's equations in the Transverse Electric (TE) mode.

    Attributes:
        backbone: The backbone neural network model that approximates the field solutions.
        _loss_residual_weight: The weight of the residual loss.
        _loss_boundary_weight: The weight of the boundary loss.
        loss_residual_tracker: Tracker for residual loss.
        loss_boundary_tracker: Tracker for boundary loss.
        mae_tracker: Tracker for mean absolute error.
    """

    def __init__(self, backbone: tf.keras.Model, loss_residual_weight: float = 1.0, loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the Maxwell2DPinn model.

        Args:
            backbone: The neural network backbone model.
            loss_residual_weight: Weight of the residual loss.
            loss_boundary_weight: Weight of the boundary loss.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.loss_total_tracker = tf.keras.metrics.Mean(name='loss_total')
        self.loss_residual_tracker = tf.keras.metrics.Mean(name='loss_residual')
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name='loss_boundary')
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name='mae')
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass and computes the fields and residuals for Maxwell's equations.

        Args:
            inputs: A tuple containing field samples and boundary samples.
            training: Boolean indicating training mode.

        Returns:
            Tuple of electric field, magnetic field residuals, and boundary solution.
        """
        tx_samples, tx_bnd = inputs

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(tx_samples)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)
                E_z, H_x, H_y = u_samples[..., 0], u_samples[..., 1], u_samples[..., 2]

            # First-order derivatives
            E_z_x = tape.gradient(E_z, tx_samples)[..., 1]
            E_z_y = tape.gradient(E_z, tx_samples)[..., 2]
            H_x_t = tape.gradient(H_x, tx_samples)[..., 0]
            H_y_t = tape.gradient(H_y, tx_samples)[..., 0]

            # Second-order derivatives
            E_z_tt = tape2.gradient(E_z, tx_samples)[..., 0]
            H_x_y = tape2.gradient(H_x, tx_samples)[..., 2]
            H_y_x = tape2.gradient(H_y, tx_samples)[..., 1]

        # Maxwell's TE mode residuals
        residual_Ez = E_z_tt - (H_y_x - H_x_y)
        residual_Hx = H_x_t - E_z_y
        residual_Hy = H_y_t + E_z_x

        # Boundary conditions
        u_bnd = self.backbone(tx_bnd, training=training)

        return u_samples, residual_Ez, residual_Hx, residual_Hy, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_bnd]
        """
        inputs, outputs = data
        E_z_exact, H_x_exact, H_y_exact, rhs_Ez_exact, rhs_Hx_exact, rhs_Hy_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, residual_Ez, residual_Hx, residual_Hy, u_bnd = self(inputs, training=True)

            # Calculate residual and boundary losses
            loss_residual = (self.res_loss(rhs_Ez_exact, residual_Ez) + 
                             self.res_loss(rhs_Hx_exact, residual_Hx) + 
                             self.res_loss(rhs_Hy_exact, residual_Hy))
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(E_z_exact, u_samples[..., 0])
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.
        """
        inputs, outputs = data
        E_z_exact, H_x_exact, H_y_exact, rhs_Ez_exact, rhs_Hx_exact, rhs_Hy_exact, u_bnd_exact = outputs

        u_samples, residual_Ez, residual_Hx, residual_Hy, u_bnd = self(inputs, training=False)
        loss_residual = (self.res_loss(rhs_Ez_exact, residual_Ez) + 
                         self.res_loss(rhs_Hx_exact, residual_Hx) + 
                         self.res_loss(rhs_Hy_exact, residual_Hy))
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(E_z_exact, u_samples[..., 0])
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_boundary_tracker, self.mae_tracker]

    def set_loss_weights(self, loss_residual_weight, loss_boundary_weight):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)



class Helmholtz2DPinn(tf.keras.Model):
    """
    A PINN for the Helmholtz equation in 2D.

    Attributes:
        backbone: The backbone model.
        _k: The wave number.
        _loss_residual_weight: The weight of the residual loss.
        _loss_boundary_weight: The weight of the boundary loss.
        loss_residual_tracker: The residual loss tracker.
        loss_boundary_tracker: The boundary loss tracker.
        mae_tracker: The mean absolute error tracker.
    """

    def __init__(self, backbone: tf.keras.Model, loss_residual_weight: float = 1.0, loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
        """

        super().__init__(**kwargs)
        self.backbone = backbone
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight",
                                                 dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight,
                                                 trainable=False, name="loss_boundary_weight",
                                                 dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, and the solution for the boundary samples.
        """
        tx_samples, tx_bnd = inputs

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)
            
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)
            first_order = tape.batch_jacobian(u_samples, tx_samples)[:, 0, :] # (N, 2)
        second_order = tape2.batch_jacobian(first_order, tx_samples)
        u_xx = second_order[:, 0, 0:1]
        u_yy = second_order[:, 1, 1:2]
        residuals = u_xx + u_yy + u_samples

        u_bnd = self.backbone(tx_bnd, training=training)

        return u_samples, residuals, u_bnd
    
    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_bnd]
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, residuals, u_bnd = self(inputs, training=True)
            loss_residual = self.res_loss(rhs_samples_exact, residuals)
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.
        """
        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_bnd_exact = outputs

        u_samples, residuals, u_bnd = self(inputs, training=False)
        loss_residual = self.res_loss(rhs_samples_exact, residuals)
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, \
                         MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_boundary_tracker, \
                self.mae_tracker]
    
    def set_loss_weights(self, loss_residual_weight, loss_boundary_weight):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss. Defaults to 1.
            loss_initial_weight: The weight of the initial loss. Defaults to 1.
            loss_boundary_weight: The weight of the boundary loss. Defaults to 1.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
