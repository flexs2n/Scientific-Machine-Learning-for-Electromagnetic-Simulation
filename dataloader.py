import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import glob
import h5py

try:
    from .datasets import FDTD2DDataset
except:
    from datasets import FDTD2DDataset


class EMTMzDataloader(Dataset):
    """Dataloader for 2D TMz Electromagnetics Dataset from MATLAB .mat files"""

    def __init__(
        self, dataset: FDTD2DDataset, sub_x=1, sub_y=1, sub_t=1, ind_x=None, ind_y=None, ind_t=None
    ):
        self.dataset = dataset
        self.sub_x = sub_x
        self.sub_y = sub_y
        self.sub_t = sub_t
        self.ind_x = ind_x
        self.ind_y = ind_y
        self.ind_t = ind_t
        t, x, y = dataset.get_coords(0)
        self.x = x[:ind_x:sub_x]
        self.y = y[:ind_y:sub_y]
        self.t = t[:ind_t:sub_t]
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nt = len(self.t)
        self.num = len(self.dataset)
        self.x_slice = slice(0, self.ind_x, self.sub_x)
        self.y_slice = slice(0, self.ind_y, self.sub_y)
        self.t_slice = slice(0, self.ind_t, self.sub_t)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """Gets input and output tensors for the dataloader"""
        fields = self.dataset[index]

        # Fields: Ez, Hx, Hy
        Ez = fields["Ez"]
        Hx = fields["Hx"]
        Hy = fields["Hy"]

        # Subsample and convert to tensors, aligning grids
        # Note: Hx and Hy are staggered; interpolate to Ez grid (nx, ny)
        Ez_tensor = torch.from_numpy(
            Ez[:self.ind_t:self.sub_t, :self.ind_x:self.sub_x, :self.ind_y:self.sub_y]
        )
        Hx_tensor = torch.from_numpy(
            Hx[:self.ind_t:self.sub_t, :self.ind_x:self.sub_x, :-1:self.sub_y]
        )  # NY+1 -> NY
        Hy_tensor = torch.from_numpy(
            Hy[:self.ind_t:self.sub_t, :-1:self.sub_x, :self.ind_y:self.sub_y]
        )  # NX+1 -> NX

        # Simple interpolation to align Hx, Hy with Ez grid (could refine later)
        Hx_tensor = (Hx_tensor[:, :, :-1] + Hx_tensor[:, :, 1:]) / 2  # Average y-direction
        Hy_tensor = (Hy_tensor[:, :-1, :] + Hy_tensor[:, 1:, :]) / 2  # Average x-direction

        # Stack outputs: (nt, nx, ny, 3)
        data = torch.stack([Ez_tensor, Hx_tensor, Hy_tensor], dim=-1)
        data0 = data[0].reshape(1, self.nx, self.ny, -1).repeat(self.nt, 1, 1, 1)

        # Grid tensors
        grid_t = (
            torch.from_numpy(self.t)
            .reshape(self.nt, 1, 1, 1)
            .repeat(1, self.nx, self.ny, 1)
        )
        grid_x = (
            torch.from_numpy(self.x)
            .reshape(1, self.nx, 1, 1)
            .repeat(self.nt, 1, self.ny, 1)
        )
        grid_y = (
            torch.from_numpy(self.y)
            .reshape(1, 1, self.ny, 1)
            .repeat(self.nt, self.nx, 1, 1)
        )

        # Inputs: (nt, nx, ny, 6) - grid + initial conditions
        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data

        return inputs, outputs

    def create_dataloader(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        distributed=False,
    ):
        """Creates dataloader and sampler based on whether distributed training is on"""
        if distributed:
            sampler = torch.utils.data.DistributedSampler(self)
            dataloader = DataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            sampler = None
            dataloader = DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return dataloader, sampler


if __name__ == "__main__":
    dataset = FDTD2DDataset(
        data_path="em_data",
        output_names="output_*.mat",
        field_names=["Ez", "Hx", "Hy"],
    )
    em_dataloader = EMTMzDataloader(dataset)
    inputs, outputs = em_dataloader[0]
    print(f"Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}")
