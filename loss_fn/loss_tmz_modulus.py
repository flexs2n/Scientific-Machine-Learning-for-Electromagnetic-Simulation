import numpy as np
import torch
import torch.nn.functional as F
from .losses import LpLoss


class LossTMz(object):
    """Calculate loss for TMz Maxwell's equations without PhysicsNeMo sym, with dynamic weighting."""

    def __init__(
        self,
        epsilon_0=8.854187817e-12,
        mu_0=1.2566370614359173e-6,
        relative_permittivity=1.0,
        sigma=0.0,
        initial_data_weight = 0.5, 
        initial_ic_weight = 0.3, 
        initial_pde_weight = 0.2,
        final_data_weight=0.5,
        final_ic_weight=0.3,
        final_pde_weight=0.2,
        schedule_epochs=100,
        use_adaptive_weighting=True,
        smoothing_alpha=0.9,
        use_data_loss=True,
        use_ic_loss=True,
        use_pde_loss=True,
        Ez_weight=1e-4,
        Hx_weight=1.0,
        Hy_weight=1.0,
        DEz_weight=1e-4,
        DHx_weight=1.0,
        DHy_weight=1.0,
        Lx=0.1,
        Ly=0.1,
        tend=3.333e-10,
        use_weighted_mean=False,
        **kwargs
    ):
        self.epsilon_0 = epsilon_0
        self.mu_0 = mu_0
        self.relative_permittivity = relative_permittivity
        self.sigma = sigma
        self.epsilon = self.epsilon_0 * self.relative_permittivity
        self.initial_data_weight = initial_data_weight
        self.initial_ic_weight = initial_ic_weight
        self.initial_pde_weight = initial_pde_weight
        self.final_data_weight = final_data_weight
        self.final_ic_weight = final_ic_weight
        self.final_pde_weight = final_pde_weight
        self.schedule_epochs = schedule_epochs
        self.use_adaptive_weighting = use_adaptive_weighting
        self.smoothing_alpha = smoothing_alpha
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.Ez_weight = Ez_weight
        self.Hx_weight = Hx_weight
        self.Hy_weight = Hy_weight
        self.DEz_weight = DEz_weight
        self.DHx_weight = DHx_weight
        self.DHy_weight = DHy_weight
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend
        self.use_weighted_mean = use_weighted_mean

        # Initialize dynamic weights
        self.data_weight = initial_data_weight
        self.ic_weight = initial_ic_weight
        self.pde_weight = initial_pde_weight
        self.epoch = 0
        self.data_loss_avg = 0.0
        self.ic_loss_avg = 0.0
        self.pde_loss_avg = 0.0

        if not self.use_data_loss:
            self.data_weight = 0
        if not self.use_ic_loss:
            self.ic_weight = 0
        if not self.use_pde_loss:
            self.pde_weight = 0

    def step(self):
        """Update dynamic loss weights after each epoch."""
        self.epoch += 1
        if self.epoch <= self.schedule_epochs:
            # Linear interpolation from initial to final weights
            t = self.epoch / self.schedule_epochs
            self.data_weight = (
                (1 - t) * self.initial_data_weight + t * self.final_data_weight
            )
            self.ic_weight = (
                (1 - t) * self.initial_ic_weight + t * self.final_ic_weight
            )
            self.pde_weight = (
                (1 - t) * self.initial_pde_weight + t * self.final_pde_weight
            )

        if self.use_adaptive_weighting:
            # Normalize weights based on EMA of losses
            total_loss = (
                self.data_loss_avg + self.ic_loss_avg + self.pde_loss_avg + 1e-10
            )
            if total_loss > 0:
                self.data_weight = self.data_loss_avg / total_loss
                self.ic_weight = self.ic_loss_avg / total_loss
                self.pde_weight = self.pde_loss_avg / total_loss
                # Ensure weights sum to 1
                weight_sum = self.data_weight + self.ic_weight + self.pde_weight
                if weight_sum > 0:
                    self.data_weight /= weight_sum
                    self.ic_weight /= weight_sum
                    self.pde_weight /= weight_sum

    def __call__(self, pred, true, inputs, return_loss_dict=False):
        if not return_loss_dict:
            loss = self.compute_loss(pred, true, inputs)
            return loss
        else:
            loss, loss_dict = self.compute_losses(pred, true, inputs)
            return loss, loss_dict

    def compute_loss(self, pred, true, inputs):
        """Compute weighted loss."""
        pred = pred.reshape(true.shape)
        Ez = pred[..., 0]
        Hx = pred[..., 1]
        Hy = pred[..., 2]

        # Data loss
        loss_data = self.data_loss(pred, true) if self.use_data_loss else 0

        # IC loss
        loss_ic = self.ic_loss(pred, inputs) if self.use_ic_loss else 0

        # PDE loss
        if self.use_pde_loss:
            DEz, DHx, DHy = self.tmz_pde(Ez, Hx, Hy, inputs)
            loss_pde = self.tmz_pde_loss(DEz, DHx, DHy)
        else:
            loss_pde = 0

        if self.use_weighted_mean:
            weight_sum = self.data_weight + self.ic_weight + self.pde_weight
        else:
            weight_sum = 1.0

        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
        ) / max(weight_sum, 1e-10)
        return loss

    def compute_losses(self, pred, true, inputs):
        """Compute weighted loss and dictionary."""
        pred = pred.reshape(true.shape)
        Ez = pred[..., 0]
        Hx = pred[..., 1]
        Hy = pred[..., 2]

        loss_dict = {}

        # Data loss
        if self.use_data_loss:
            loss_data, loss_Ez, loss_Hx, loss_Hy = self.data_loss(
                pred, true, return_all_losses=True
            )
            loss_dict["loss_data"] = loss_data
            loss_dict["loss_Ez"] = loss_Ez
            loss_dict["loss_Hx"] = loss_Hx
            loss_dict["loss_Hy"] = loss_Hy
            # Update EMA for adaptive weighting
            self.data_loss_avg = (
                self.smoothing_alpha * self.data_loss_avg
                + (1 - self.smoothing_alpha) * loss_data.item()
            )
        else:
            loss_data = 0

        # IC loss
        if self.use_ic_loss:
            loss_ic, loss_Ez_ic, loss_Hx_ic, loss_Hy_ic = self.ic_loss(
                pred, inputs, return_all_losses=True
            )
            loss_dict["loss_ic"] = loss_ic
            loss_dict["loss_Ez_ic"] = loss_Ez_ic
            loss_dict["loss_Hx_ic"] = loss_Hx_ic
            loss_dict["loss_Hy_ic"] = loss_Hy_ic
            self.ic_loss_avg = (
                self.smoothing_alpha * self.ic_loss_avg
                + (1 - self.smoothing_alpha) * loss_ic.item()
            )
        else:
            loss_ic = 0

        # PDE loss
        if self.use_pde_loss:
            DEz, DHx, DHy = self.tmz_pde(Ez, Hx, Hy, inputs)
            loss_pde, loss_DEz, loss_DHx, loss_DHy = self.tmz_pde_loss(
                DEz, DHx, DHy, return_all_losses=True
            )
            loss_dict["loss_pde"] = loss_pde
            loss_dict["loss_DEz"] = loss_DEz
            loss_dict["loss_DHx"] = loss_DHx
            loss_dict["loss_DHy"] = loss_DHy
            self.pde_loss_avg = (
                self.smoothing_alpha * self.pde_loss_avg
                + (1 - self.smoothing_alpha) * loss_pde.item()
            )
        else:
            loss_pde = 0

        if self.use_weighted_mean:
            weight_sum = self.data_weight + self.ic_weight + self.pde_weight
        else:
            weight_sum = 1.0

        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
        ) / max(weight_sum, 1e-10)
        loss_dict["loss"] = loss
        return loss, loss_dict

    def data_loss(self, pred, true, return_all_losses=False):
        """Compute data loss."""
        lploss = LpLoss(size_average=True)
        Ez_pred = pred[..., 0]
        Hx_pred = pred[..., 1]
        Hy_pred = pred[..., 2]
        Ez_true = true[..., 0]
        Hx_true = true[..., 1]
        Hy_true = true[..., 2]

        loss_Ez = lploss(Ez_pred, Ez_true)
        loss_Hx = lploss(Hx_pred, Hx_true)
        loss_Hy = lploss(Hy_pred, Hy_true)

        if self.use_weighted_mean:
            weight_sum = self.Ez_weight + self.Hx_weight + self.Hy_weight
        else:
            weight_sum = 1.0

        loss_data = (
            self.Ez_weight * loss_Ez
            + self.Hx_weight * loss_Hx
            + self.Hy_weight * loss_Hy
        ) / max(weight_sum, 1e-10)

        if return_all_losses:
            return loss_data, loss_Ez, loss_Hx, loss_Hy
        else:
            return loss_data

    def ic_loss(self, pred, inputs, return_all_losses=False):
        """Compute initial condition loss."""
        lploss = LpLoss(size_average=True)
        ic_pred = pred[:, 0]
        ic_true = inputs[:, 0, ..., 3:6]  # Ez0, Hx0, Hy0
        Ez_ic_pred = ic_pred[..., 0]
        Hx_ic_pred = ic_pred[..., 1]
        Hy_ic_pred = ic_pred[..., 2]
        Ez_ic_true = ic_true[..., 0]
        Hx_ic_true = ic_true[..., 1]
        Hy_ic_true = ic_true[..., 2]

        loss_Ez_ic = lploss(Ez_ic_pred, Ez_ic_true)
        loss_Hx_ic = lploss(Hx_ic_pred, Hx_ic_true)
        loss_Hy_ic = lploss(Hy_ic_pred, Hy_ic_true)

        if self.use_weighted_mean:
            weight_sum = self.Ez_weight + self.Hx_weight + self.Hy_weight
        else:
            weight_sum = 1.0

        loss_ic = (
            self.Ez_weight * loss_Ez_ic
            + self.Hx_weight * loss_Hx_ic
            + self.Hy_weight * loss_Hy_ic
        ) / max(weight_sum, 1e-10)

        if return_all_losses:
            return loss_ic, loss_Ez_ic, loss_Hx_ic, loss_Hy_ic
        else:
            return loss_ic

    def tmz_pde_loss(self, DEz, DHx, DHy, return_all_losses=False):
        """Compute PDE loss."""
        DEz_val = torch.zeros_like(DEz)
        DHx_val = torch.zeros_like(DHx)
        DHy_val = torch.zeros_like(DHy)

        Ez_scale = 1e4  # Adjust based on Ez field magnitude
        Hx_scale = 1e0  # Adjust based on Hx field magnitude
        Hy_scale = 1e0  # Adjust based on Hy field magnitude
        loss_DEz = F.mse_loss(DEz / Ez_scale, DEz_val / Ez_scale)
        loss_DHx = F.mse_loss(DHx / Hx_scale, DHx_val / Hx_scale)
        loss_DHy = F.mse_loss(DHy / Hy_scale, DHy_val / Hy_scale)
        # loss_pde = loss_DEz + loss_DHx + loss_DHy

        if self.use_weighted_mean:
            weight_sum = self.DEz_weight + self.DHx_weight + self.DHy_weight
        else:
            weight_sum = 1.0

        loss_pde = (
            self.DEz_weight * loss_DEz
            + self.DHx_weight * loss_DHx
            + self.DHy_weight * loss_DHy
        ) / max(weight_sum, 1e-10)

        if return_all_losses:
            return loss_pde, loss_DEz, loss_DHx, loss_DHy
        else:
            return loss_pde

    def tmz_pde(self, Ez, Hx, Hy, inputs):
        """Compute TMz PDE residuals."""
        batchsize = Ez.size(0)
        nt = Ez.size(1)
        nx = Ez.size(2)
        ny = Ez.size(3)
        device = Ez.device
        dt = self.tend / (nt - 1)
        dx = self.Lx / nx
        dy = self.Ly / ny
        k_max = nx // 2

        # Wavenumbers for Fourier derivatives
        k_x = (
            2
            * np.pi
            / self.Lx
            * torch.cat(
                [
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ],
                0,
            )
            .reshape(nx, 1)
            .repeat(1, ny)
            .reshape(1, 1, nx, ny)
        )
        k_y = (
            2
            * np.pi
            / self.Ly
            * torch.cat(
                [
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ],
                0,
            )
            .reshape(1, ny)
            .repeat(nx, 1)
            .reshape(1, 1, nx, ny)
        )

        # Source term
        Jz = inputs[..., 6]  # src_field

        # Fourier transforms
        Ez_h = torch.fft.fftn(Ez, dim=[2, 3])
        Hx_h = torch.fft.fftn(Hx, dim=[2, 3])
        Hy_h = torch.fft.fftn(Hy, dim=[2, 3])

        # Spatial derivatives
        Ez_x_h = self.Du_i(Ez_h, k_x)
        Ez_y_h = self.Du_i(Ez_h, k_y)
        Hx_y_h = self.Du_i(Hx_h, k_y)
        Hy_x_h = self.Du_i(Hy_h, k_x)

        Ez_x = torch.fft.irfftn(Ez_x_h[..., : k_max + 1], dim=[2, 3])
        Ez_y = torch.fft.irfftn(Ez_y_h[..., : k_max + 1], dim=[2, 3])
        Hx_y = torch.fft.irfftn(Hx_y_h[..., : k_max + 1], dim=[2, 3])
        Hy_x = torch.fft.irfftn(Hy_x_h[..., : k_max + 1], dim=[2, 3])

        # Time derivatives
        Ez_t = self.Du_t(Ez, dt)
        Hx_t = self.Du_t(Hx, dt)
        Hy_t = self.Du_t(Hy, dt)

        # TMz PDE right-hand sides
        Ez_rhs = (1 / self.epsilon) * (Hy_x - Hx_y - self.sigma * Ez - Jz)
        Hx_rhs = -(1 / self.mu_0) * Ez_y
        Hy_rhs = (1 / self.mu_0) * Ez_x

        # PDE residuals
        DEz = Ez_t - Ez_rhs[:, 1:-1]
        DHx = Hx_t - Hx_rhs[:, 1:-1]
        DHy = Hy_t - Hy_rhs[:, 1:-1]

        return DEz, DHx, DHy

    def Du_t(self, u, dt):
        """Compute time derivative."""
        u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        return u_t

    def Du_i(self, u_h, k_i):
        """Compute spatial derivative in Fourier space."""
        u_i_h = (1j * k_i) * u_h
        return u_i_h