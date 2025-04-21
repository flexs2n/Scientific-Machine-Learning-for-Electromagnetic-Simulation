import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import imageio
from tqdm import tqdm

def plot_predictions_tmz(
    pred,
    true,
    inputs,
    index_t=-1,
    names=['Ez', 'Hx', 'Hy'],
    save_path=None,
    save_suffix=None,
    font_size=12,
    sci_limits=(-3, 3),
    shading="auto",
    cmap="viridis",
):
    """Plots 2D heatmaps of TMz predictions, ground truth, initial conditions, and errors.

    Args:
        pred (torch.Tensor): Predicted fields [nt, nx, ny, 3] (Ez, Hx, Hy).
        true (torch.Tensor): Ground truth fields [nt, nx, ny, 3].
        inputs (torch.Tensor): Input data [nt, nx, ny, 7] (t, x, y, Ez0, Hx0, Hy0, src_field).
        index_t (int): Time index to plot (default: -1, last timestep).
        names (list): Field names ['Ez', 'Hx', 'Hy'].
        save_path (str): Directory to save PNG (optional).
        save_suffix (str): Suffix for saved file (optional).
        font_size (int): Font size for plot text.
        sci_limits (tuple): Scientific notation limits for colorbar.
        shading (str): Shading mode for pcolormesh.
        cmap (str): Colormap for heatmaps.
    """
    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})
    if sci_limits is not None:
        plt.rcParams.update({"axes.formatter.limits": sci_limits})

    # Normalize fields (Ez ~1.2e4, Hx, Hy ~1e0)
    output_norm = torch.tensor([1.2e4, 1.0, 1.0], device=pred.device)
    pred = pred / output_norm
    true = true / output_norm

    # Extract coordinates
    x = inputs[0, :, 0, 1].cpu()  # [nx]
    y = inputs[0, 0, :, 2].cpu()  # [ny]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    t = inputs[index_t, 0, 0, 0].cpu().item()  # Scalar time

    # Initial conditions
    initial_data = inputs[0, ..., 3:6] / output_norm  # [nx, ny, 3]

    # Plot setup
    fig = plt.figure(figsize=(24, 5 * len(names)))

    for index, name in enumerate(names):
        u_pred = pred[index_t, ..., index].cpu()
        u_true = true[index_t, ..., index].cpu()
        u_err = u_pred - u_true
        u_ic = initial_data[..., index].cpu()

        # Clip color scale to avoid outliers
        zmin = min(u_true.min(), u_pred.min(), u_ic.min()).item()
        zmax = max(u_true.max(), u_pred.max(), u_ic.max()).item()
        if abs(zmax - zmin) < 1e-6:
            zmin, zmax = -1, 1

        # Initial condition
        plt.subplot(len(names), 4, index * 4 + 1)
        plt.pcolormesh(X, Y, u_ic, cmap=cmap, shading=shading, vmin=zmin, vmax=zmax)
        plt.colorbar()
        plt.title(f"Initial ${name}_0(x,y)$")
        plt.axis("square")
        plt.axis("off")

        # Ground truth
        plt.subplot(len(names), 4, index * 4 + 2)
        plt.pcolormesh(X, Y, u_true, cmap=cmap, shading=shading, vmin=zmin, vmax=zmax)
        plt.colorbar()
        plt.title(f"Exact ${name}(x,y,t={t:.2e})$")
        plt.axis("square")
        plt.axis("off")

        # Prediction
        plt.subplot(len(names), 4, index * 4 + 3)
        plt.pcolormesh(X, Y, u_pred, cmap=cmap, shading=shading, vmin=zmin, vmax=zmax)
        plt.colorbar()
        plt.title(f"Predicted ${name}(x,y,t={t:.2e})$")
        plt.axis("square")
        plt.axis("off")

        # Error
        plt.subplot(len(names), 4, index * 4 + 4)
        plt.pcolormesh(X, Y, u_err, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Error ${name}(x,y,t={t:.2e})$")
        plt.axis("square")
        plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        figure_path = f"{save_path}_{save_suffix}.png" if save_suffix else f"{save_path}.png"
        plt.savefig(figure_path, bbox_inches="tight", dpi=300)

    plt.close()

def plot_predictions_tmz_plotly(
    pred,
    true,
    inputs,
    index=0,
    index_t=-1,
    name="Ez",
    save_path=None,
    font_size=12,
    shading="auto",
    cmap="viridis",
):
    """Generates plotly figures for TMz fields for wandb logging.

    Args:
        pred (torch.Tensor): Predicted fields [nt, nx, ny, 3].
        true (torch.Tensor): Ground truth fields [nt, nx, ny, 3].
        inputs (torch.Tensor): Input data [nt, nx, ny, 7].
        index (int): Field index (0: Ez, 1: Hx, 2: Hy).
        index_t (int): Time index to plot.
        name (str): Field name ('Ez', 'Hx', 'Hy').
        save_path (str): Not used (for compatibility).
        font_size (int): Font size for titles.
        shading (str): Not used (for compatibility).
        cmap (str): Colormap for heatmaps.

    Returns:
        tuple: Four plotly figures (ic, pred, true, error).
    """
    # Normalize fields
    output_norm = torch.tensor([1.2e4, 1.0, 1.0], device=pred.device)
    pred = pred / output_norm
    true = true / output_norm

    # Extract data
    u_pred = pred[index_t, ..., index].cpu().numpy()
    u_true = true[index_t, ..., index].cpu().numpy()
    u_ic = inputs[0, ..., 3 + index].cpu().numpy() / output_norm[index]
    u_err = u_pred - u_true

    # Coordinates
    x = inputs[0, :, 0, 1].cpu().numpy()  # [nx]
    y = inputs[0, 0, :, 2].cpu().numpy()  # [ny]
    t = inputs[index_t, 0, 0, 0].cpu().item()

    # Color scale
    zmin = min(u_true.min(), u_pred.min(), u_ic.min())
    zmax = max(u_true.max(), u_pred.max(), u_ic.max())
    if abs(zmax - zmin) < 1e-6:
        zmin, zmax = -1, 1

    labels = {"color": name}

    # Initial condition
    fig_ic = px.imshow(
        u_ic,
        x=y,
        y=x,
        color_continuous_scale=cmap,
        labels=labels,
        title=f"{name}_0",
        zmin=zmin,
        zmax=zmax,
    )
    fig_ic.update_xaxes(showticklabels=False)
    fig_ic.update_yaxes(showticklabels=False)
    fig_ic.update_layout(font=dict(size=font_size))

    # Ground truth
    fig_true = px.imshow(
        u_true,
        x=y,
        y=x,
        color_continuous_scale=cmap,
        labels=labels,
        title=f"Exact {name}: t={t:.2e}",
        zmin=zmin,
        zmax=zmax,
    )
    fig_true.update_xaxes(showticklabels=False)
    fig_true.update_yaxes(showticklabels=False)
    fig_true.update_layout(font=dict(size=font_size))

    # Prediction
    fig_pred = px.imshow(
        u_pred,
        x=y,
        y=x,
        color_continuous_scale=cmap,
        labels=labels,
        title=f"Predicted {name}: t={t:.2e}",
        zmin=zmin,
        zmax=zmax,
    )
    fig_pred.update_xaxes(showticklabels=False)
    fig_pred.update_yaxes(showticklabels=False)
    fig_pred.update_layout(font=dict(size=font_size))

    # Error
    fig_err = px.imshow(
        u_err,
        x=y,
        y=x,
        color_continuous_scale=cmap,
        labels=labels,
        title=f"Error {name}: t={t:.2e}",
    )
    fig_err.update_xaxes(showticklabels=False)
    fig_err.update_yaxes(showticklabels=False)
    fig_err.update_layout(font=dict(size=font_size))

    return fig_ic, fig_pred, fig_true, fig_err

def generate_movie_2D_tmz(
    preds_y,
    test_y,
    test_x,
    key=0,
    plot_title="Ez",
    field=0,
    val_cbar_index=-1,
    err_cbar_index=-1,
    val_clim=None,
    err_clim=None,
    font_size=12,
    movie_dir="movies",
    movie_name="tmz_movie.gif",
    frame_basename="tmz_frame",
    frame_ext="jpg",
    cmap="viridis",
    shading="gouraud",
    remove_frames=True,
):
    """Generates a movie of TMz exact, predicted, and error fields over time.

    Args:
        preds_y (torch.Tensor): Predicted fields [nt, nx, ny, 3].
        test_y (torch.Tensor): Ground truth fields [nt, nx, ny, 3].
        test_x (torch.Tensor): Input data [nt, nx, ny, 7].
        key (int): Batch index.
        plot_title (str): Field name for title ('Ez', 'Hx', 'Hy').
        field (int): Field index (0: Ez, 1: Hx, 2: Hy).
        val_cbar_index (int): Time index for colorbar scaling.
        err_cbar_index (int): Time index for error colorbar.
        val_clim (tuple): Color scale for true/pred (min, max).
        err_clim (tuple): Color scale for error.
        font_size (int): Font size for titles.
        movie_dir (str): Directory to save movie.
        movie_name (str): Name of GIF file.
        frame_basename (str): Base name for frame files.
        frame_ext (str): Extension for frame files.
        cmap (str): Colormap.
        shading (str): Shading mode for pcolormesh.
        remove_frames (bool): Delete frame files after movie creation.
    """
    frame_files = []
    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    # Normalize fields
    output_norm = torch.tensor([1.2e4, 1.0, 1.0], device=preds_y.device)
    pred = preds_y[key][..., field].cpu() / output_norm[field]
    true = test_y[key][..., field].cpu() / output_norm[field]
    error = pred - true

    Nt, Nx, Ny = pred.shape
    t = test_x[key][:, 0, 0, 0].cpu()
    x = test_x[key][0, :, 0, 1].cpu()
    y = test_x[key][0, 0, :, 2].cpu()
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Initialize plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axs

    # Set color scales
    if val_clim is None:
        val_clim = (
            min(true[val_cbar_index].min(), pred[val_cbar_index].min()).item(),
            max(true[val_cbar_index].max(), pred[val_cbar_index].max()).item(),
        )
        if abs(val_clim[1] - val_clim[0]) < 1e-6:
            val_clim = (-1, 1)
    if err_clim is None:
        err_clim = (error[err_cbar_index].min().item(), error[err_cbar_index].max().item())
        if abs(err_clim[1] - err_clim[0]) < 1e-6:
            err_clim = (-1, 1)

    # Initial frames for colorbar
    pcm1 = ax1.pcolormesh(X, Y, true[val_cbar_index], cmap=cmap, shading=shading)
    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.axis("square")
    ax1.set_axis_off()

    pcm2 = ax2.pcolormesh(X, Y, pred[val_cbar_index], cmap=cmap, shading=shading)
    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.axis("square")
    ax2.set_axis_off()

    pcm3 = ax3.pcolormesh(X, Y, error[err_cbar_index], cmap=cmap, shading=shading)
    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.axis("square")
    ax3.set_axis_off()

    plt.tight_layout()

    # Generate frames
    for i in tqdm(range(Nt), desc="Generating movie frames"):
        ax1.clear()
        pcm1 = ax1.pcolormesh(X, Y, true[i], cmap=cmap, shading=shading)
        pcm1.set_clim(val_clim)
        ax1.set_title(f"Exact {plot_title}: t={t[i]:.2e}")
        ax1.axis("square")
        ax1.set_axis_off()

        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[i], cmap=cmap, shading=shading)
        pcm2.set_clim(val_clim)
        ax2.set_title(f"Predicted {plot_title}: t={t[i]:.2e}")
        ax2.axis("square")
        ax2.set_axis_off()

        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[i], cmap=cmap, shading=shading)
        pcm3.set_clim(err_clim)
        ax3.set_title(f"Error {plot_title}: t={t[i]:.2e}")
        ax3.axis("square")
        ax3.set_axis_off()

        fig.canvas.draw()

        if movie_dir:
            frame_path = os.path.join(movie_dir, f"{frame_basename}-{i:03}.{frame_ext}")
            frame_files.append(frame_path)
            plt.savefig(frame_path, bbox_inches="tight", dpi=300)

    # Create movie
    if movie_dir and frame_files:
        movie_path = os.path.join(movie_dir, movie_name)
        with imageio.get_writer(movie_path, mode="I", fps=10) as writer:
            for frame in frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

        if remove_frames:
            for frame in frame_files:
                try:
                    os.remove(frame)
                except:
                    pass

    plt.close()