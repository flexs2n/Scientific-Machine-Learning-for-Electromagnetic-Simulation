import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import glob
import h5py
from IPython.display import display

try:
    from .datasets import FDTD2DDataset
except:
    from datasets import FDTD2DDataset


class EMTMzDataloader(Dataset):
    """Dataloader for 2D TMz Electromagnetics Dataset from HDF5 files"""

    def __init__(
        self, dataset: FDTD2DDataset, sub_x=2, sub_y=2, sub_t=1, ind_x=None, ind_y=None, ind_t=None
    ):
        self.dataset = dataset
        self.sub_x = sub_x
        self.sub_y = sub_y
        self.sub_t = sub_t
        self.ind_x = ind_x or 100  # NX=100
        self.ind_y = ind_y or 100  # NY=100
        self.ind_t = ind_t or 100  # Focus on first 100 steps (non-zero fields)
        t, x, y = dataset.get_coords(0)
        self.x = x[:self.ind_x:self.sub_x]  # e.g., [50] if sub_x=2
        self.y = y[:self.ind_y:self.sub_y]  # e.g., [50] if sub_y=2
        self.t = t[:self.ind_t:self.sub_t]  # e.g., [100] if sub_t=1, ind_t=100
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

        # Fields
        Ez = fields["Ez_out"]  # [208, 100, 100]
        Hx = fields["Hx_out"]  # [208, 100, 101]
        Hy = fields["Hy_out"]  # [208, 101, 100]
        Sx = fields["Sx"]  # Scalar
        Sy = fields["Sy"]  # Scalar
        src = fields["src"]  # [208]

        # Interpolate Hx, Hy to Ez grid [208, 100, 100]
        Hx = (Hx[:, :, :-1] + Hx[:, :, 1:]) / 2  # Average y-direction
        Hy = (Hy[:, :-1, :] + Hy[:, 1:, :]) / 2  # Average x-direction

        # Subsample and convert to tensors
        Ez_tensor = torch.from_numpy(
            Ez[:self.ind_t:self.sub_t, :self.ind_x:self.sub_x, :self.ind_y:self.sub_y]
        ).float()
        Hx_tensor = torch.from_numpy(
            Hx[:self.ind_t:self.sub_t, :self.ind_x:self.sub_x, :self.ind_y:self.sub_y]
        ).float()
        Hy_tensor = torch.from_numpy(
            Hy[:self.ind_t:self.sub_t, :self.ind_x:self.sub_y, :self.ind_y:self.sub_y]
        ).float()

        # Stack outputs: [nt, nx, ny, 3]
        data = torch.stack([Ez_tensor, Hx_tensor, Hy_tensor], dim=-1)  # e.g., [100, 50, 50, 3]

        # Initial conditions: [1, nx, ny, 3]
        data0 = data[0:1].repeat(self.nt, 1, 1, 1)  # Repeat t=0 over nt

        # Grid tensors
        grid_t = (
            torch.from_numpy(self.t)
            .reshape(self.nt, 1, 1, 1)
            .repeat(1, self.nx, self.ny, 1)
            .float()
        )
        grid_x = (
            torch.from_numpy(self.x)
            .reshape(1, self.nx, 1, 1)
            .repeat(self.nt, 1, self.ny, 1)
            .float()
        )
        grid_y = (
            torch.from_numpy(self.y)
            .reshape(1, 1, self.ny, 1)
            .repeat(self.nt, self.nx, 1, 1)
            .float()
        )

        # Source term: Create a 2D mask at Sx, Sy
        src_tensor = torch.from_numpy(src[:self.ind_t:self.sub_t]).reshape(self.nt, 1, 1, 1).float()
        src_mask = torch.zeros(1, self.nx, self.ny, 1).float()
        src_idx_x = min(int(Sx * self.nx / 100), self.nx - 1)  # Scale Sx to subsampled grid
        src_idx_y = min(int(Sy * self.ny / 100), self.ny - 1)  # Scale Sy to subsampled grid
        src_mask[:, src_idx_x, src_idx_y, :] = 1.0
        src_field = src_tensor * src_mask  # [nt, nx, ny, 1]

        # Inputs: [nt, nx, ny, 7] - grid + initial conditions + source
        inputs = torch.cat([grid_t, grid_x, grid_y, data0, src_field], dim=-1)  # e.g., [100, 50, 50, 7]
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
        data_path="hdf5_files",
        output_names="results_*.h5",
    )
    em_dataloader = EMTMzDataloader(dataset, sub_t=1, sub_x=2, sub_y=2, ind_t=100)
    inputs, outputs = em_dataloader[0]
    print(f"Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}") 
    
