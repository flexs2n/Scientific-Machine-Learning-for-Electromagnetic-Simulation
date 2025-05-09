"""
Utility module for plotting models and losses.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
import numpy as np
import tensorflow as tf
from modules.models import LOSS_RESIDUAL, LOSS_BOUNDARY, LOSS_INITIAL, MEAN_ABSOLUTE_ERROR


def plot_wave_model_slider(model, x_start, length, time, figsize=(10,5), show=True) -> None:
    '''
    plot the solution of the wave equation for a given model with a time slider. If in a notebook, use %matplotlib notebook before calling this function.
    The returned object must be kept in memory to keep the slider working.
    '''
    t, x = np.meshgrid(np.linspace(0, time, 100), np.linspace(x_start, x_start + length, 100))
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=1000)
    u = u.reshape(t.shape)

    init_time = 0
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot(x[:, 0].flatten(), u[init_time])

    ax.set_xlim(x_start, x_start + length)
    ax.set_xlabel('x')
    ax.set_ylim([-1, 1])

    fig.subplots_adjust(bottom=0.25)

    axtime = plt.axes([0.25, 0.1, 0.65, 0.03])
    stime = Slider(axtime, 'Time', 0, 99, valinit=init_time, valstep=1)

    def update(val):
        time = stime.val
        line.set_ydata(u[time:time+1])
        fig.canvas.draw_idle()

    stime.on_changed(update)

    if show:
        plt.show()
    return stime


def plot_wave_at_x(model, x, time, save_path = None, show=True) -> None:
    """
    Plot the solution of the wave equation for a given model at a given x coordinate.
    Args:
        model (tf.keras.Model): Model that predicts the solution of the wave equation.
        x (float): x coordinate of the plot.
        time (float): Time frame of the simulation.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    t = np.linspace(0, time, 100)
    u = model.predict(np.stack([t, np.full(t.shape, x)], axis=-1), batch_size=1000)
    plt.plot(t, u)
    plt.xlabel('t')
    plt.ylabel('u')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_wave_model(model, x_start, length, time, save_path = None, show=True) -> None:
    """
    Plot the solution of the wave equation for a given model.
    Args:
        model (tf.keras.Model): Model that predicts the solution of the wave equation.
        x_start (float): Start of the domain.
        length (float): Length of the domain.
        time (float): Time frame of the simulation.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    t, x = np.meshgrid(np.linspace(0, time, 100), np.linspace(x_start, x_start + length, 100))  
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=1000)

    fig = plt.figure(figsize=(30, 90))
    ax = fig.add_subplot(311, projection='3d')
    surf = ax.scatter(t, x, np.reshape(u, t.shape), cmap='viridis', alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.azim = 5
    ax.elev = 20
    # fig.colorbar(surf)

    ax = fig.add_subplot(312, projection='3d')
    surf = ax.scatter(t, x, np.reshape(u, t.shape), cmap='viridis', alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.azim = 45
    ax.elev = 20
    # fig.colorbar(surf)

    ax = fig.add_subplot(313, projection='3d')
    surf = ax.scatter(t, x, np.reshape(u, t.shape), cmap='viridis', alpha=0.6)
    ax.azim = 85
    ax.elev = 20
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    # fig.colorbar(surf)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_burgers_model(model, save_path = None, show=True) -> None:
    """
    Plot the model predictions for the Burgers equation.

    Args:
        model: A trained BurgersPinn model.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    t_flat = np.linspace(0, 1, num_test_samples)
    x_flat = np.linspace(-1, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 5)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0,0.25, 0.5,0.75,1]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = model.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_heat_model(model, length, time, save_path = None, show=True) -> None:
    """
    Plot the model predictions for the heat equation.
    Args:
        model: A trained HeatPinn model.
        length: The length of the domain.
        time: The time frame of the simulation.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    t_flat = np.linspace(0, time, num_test_samples)
    x_flat = np.linspace(0, length, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 5)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0, time/4, time/2, 3*time/4, time]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = model.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_schrodinger_model(model, x_start, length, time, fig_size = (7, 4), dpi = 100, save_path = None, show=True) -> None:
    """
    Plot the model predictions for the Schrodinger equation.
    Args:
        model: A trained SchrodingerPinn model.
        length: The length of the domain.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    t_flat = np.linspace(0, time, num_test_samples)
    x_flat = np.linspace(x_start, x_start + length, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    h = model.predict(tx, batch_size=num_test_samples)
    u = tf.abs(tf.complex(h[:,0:1], h[:,1:2]))
    u = tf.reshape(u, t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    gs = GridSpec(2, 5)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('|h(t,x)|')
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0, time/4, time/2, 3*time/4, time]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        h = model.predict(tx, batch_size=num_test_samples)
        u = tf.abs(tf.complex(h[:,0:1], h[:,1:2]))
        plt.plot(x_flat, u)
        plt.title(f't={t_cs:.3f}')
        plt.xlabel('x')
        plt.ylabel('|h(t,x)|')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_poisson_model(model, x_start, length, save_path = None, show=True) -> None:
    """
    Plot the model predictions for the Poisson equation.
    Args:
        model: A trained PoissonPinn model.
        length: The length of the domain.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    x = np.linspace(x_start, x_start + length, num_test_samples)[:, np.newaxis]
    u = model.predict(x, batch_size=num_test_samples)

    # plot u(x) distribution as a color-map
    fig, ax = plt.subplots(figsize = (7,4))
    ax.plot(x.flatten(), u.flatten())
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_advection_model(model, x_start = 0.0, length = 1.0, save_path = None, show=True) -> None:
    """
    Plot the model predictions for the advection equation.
    Args:
        model: A trained AdvectionPinn model.
        length: The length of the domain.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    x = np.linspace(x_start, x_start + length, num_test_samples)[:, np.newaxis]
    u = model.predict(x, batch_size=num_test_samples)

    # plot u(x) distribution as a color-map
    fig, ax = plt.subplots(figsize = (7,4))
    ax.plot(x.flatten(), u.flatten())
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_training_loss(history, x_scale = "linear", y_scale = "linear", save_path=None, show=True):
    """
    Plot the training residual, initial, and boundary losses separately.
    Args:
        history: The history object returned by the model.fit() method.
        x_scale: The scale of the x-axis.
        y_scale: The scale of the y-axis.
    """
    plt.figure(figsize=(10, 5), dpi = 100)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    if LOSS_INITIAL in history:
        if len(history[LOSS_INITIAL]) > 0:
            plt.plot(history[LOSS_INITIAL], label='initial loss', alpha=0.8)
    if LOSS_BOUNDARY in history:
        if len(history[LOSS_BOUNDARY]) > 0:
            plt.plot(history[LOSS_BOUNDARY], label='boundary loss', alpha=0.8)
    if LOSS_RESIDUAL in history:
        if len(history[LOSS_RESIDUAL]) > 0:
            plt.plot(history[LOSS_RESIDUAL], label='residual loss', alpha=0.8)
    if MEAN_ABSOLUTE_ERROR in history:
        if len(history[MEAN_ABSOLUTE_ERROR]) > 0:
            plt.plot(history[MEAN_ABSOLUTE_ERROR], label='mean absolute error', alpha = 0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_training_loss_linlog(history, save_path=None, show=True):
    """
        plot training loss with both linear and log y scales

        Args:
            history: The history object returned by the model.fit() method.
            save_path: The path to save the plot to.
    """
    plt.figure(figsize=(12, 5), dpi = 100)
    plt.subplot(1, 2, 1)
    plt.xscale("linear")
    plt.yscale("linear")
    if LOSS_INITIAL in history:
        if len(history[LOSS_INITIAL]) > 0:
            plt.plot(history[LOSS_INITIAL], label='initial loss', alpha=0.8)
    if LOSS_BOUNDARY in history:
        if len(history[LOSS_BOUNDARY]) > 0:
            plt.plot(history[LOSS_BOUNDARY], label='boundary loss', alpha=0.8)
    if LOSS_RESIDUAL in history:
        if len(history[LOSS_RESIDUAL]) > 0:
            plt.plot(history[LOSS_RESIDUAL], label='residual loss', alpha=0.8)
    if MEAN_ABSOLUTE_ERROR in history:
        if len(history[MEAN_ABSOLUTE_ERROR]) > 0:
            plt.plot(history[MEAN_ABSOLUTE_ERROR], label='mean absolute error', alpha = 0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xscale("linear")
    plt.yscale("log")
    if LOSS_INITIAL in history:
        if len(history[LOSS_INITIAL]) > 0:
            plt.plot(history[LOSS_INITIAL], label='initial loss', alpha=0.8)
    if LOSS_BOUNDARY in history:
        if len(history[LOSS_BOUNDARY]) > 0:
            plt.plot(history[LOSS_BOUNDARY], label='boundary loss', alpha=0.8)
    if LOSS_RESIDUAL in history:
        if len(history[LOSS_RESIDUAL]) > 0:
            plt.plot(history[LOSS_RESIDUAL], label='residual loss', alpha=0.8)
    if MEAN_ABSOLUTE_ERROR in history:
        if len(history[MEAN_ABSOLUTE_ERROR]) > 0:
            plt.plot(history[MEAN_ABSOLUTE_ERROR], label='mean absolute error', alpha = 0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_pointwise_error(y_true, y_pred, x, figsize = (7, 4), dpi = 100, y_scale = "linear", save_path=None, show=True):
    """
    Plot the pointwise error between the true and predicted values.
    Args:
        y_true: The true values.
        y_pred: The predicted values.
        x: The x-values.
        figsize: The size of the figure.
        dpi: The resolution of the figure.
        y_scale: The scale of the y-axis. Either "linear" or "log".
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, np.abs(y_true - y_pred))
    plt.xscale("linear")
    plt.yscale(y_scale)
    plt.xlabel('x')
    plt.ylabel('Absolute error')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_pointwise_error_mesh(u_true_flat, u_pred_flat, t_mesh, x_mesh, cbar_limit=None, figsize=(7, 4), dpi=100, colormap = 'viridis', title="", save_path=None, show=True):
    """
    Plot the pointwise error between the true and predicted values.
    Args:
        u_true_flat: The true values. A 1D array.
        u_pred_flat: The predicted values. A 1D array.
        t_mesh: The t-values. A 2D mesh.
        x_mesh: The x-values. A 2D mesh.
        figsize: The size of the figure. Default is (7, 4).
        cb_limit: The limits of the colorbar. Default is None.
        dpi: The resolution of the figure. Default is 100.
        colormap: The colormap to use. Default is 'viridis'.
        title: The title of the figure. Default is "".
        save_path: The path to save the figure to. Default is None. If None, the figure is not saved.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.pcolormesh(t_mesh, x_mesh, np.abs(u_true_flat - u_pred_flat).reshape(t_mesh.shape), cmap=colormap)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('|h(t,x)|')
    if cbar_limit is not None:
        cbar.mappable.set_clim(cbar_limit[0], cbar_limit[1])
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()