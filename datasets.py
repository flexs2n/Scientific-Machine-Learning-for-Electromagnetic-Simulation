import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import h5py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FDTD2DDataset(Dataset):
    """Dataset for 2D TMz FDTD data from HDF5 files converted from MATLAB .mat"""

    def __init__(
        self,
        data_path,
        output_names="results_*.h5",
        field_names=[
            "Ez_out",
            "Hx_out",
            "Hy_out",
            "Sx",
            "Sy",
            "CenterX",
            "CenterY",
            "Radius",
            "relative_permittivity",
            "sigma",
        ],
        num_train=None,
        num_test=None,
        use_train=True,
    ):
        self.data_path = data_path
        self.output_names = output_names
        raw_path = os.path.join(data_path, output_names)
        files_raw = sorted(glob.glob(raw_path))
        if not files_raw:
            raise FileNotFoundError(f"No files found at {raw_path}")
        self.files_raw = files_raw
        self.num_files_raw = len(files_raw)
        self.field_names = field_names
        self.use_train = use_train

        if num_train is None or num_train > self.num_files_raw:
            num_train = int(0.8 * self.num_files_raw)  # 80% train
        self.num_train = num_train
        self.train_files = self.files_raw[:num_train]
        if num_test is None or num_test > (self.num_files_raw - num_train):
            num_test = self.num_files_raw - num_train
        self.num_test = num_test
        self.test_end = num_train + num_test
        self.test_files = self.files_raw[num_train:self.test_end]
        self.files = self.train_files if self.use_train else self.test_files
        self.num_files = len(self.files)

        # Simulation parameters
        self.NX = 100
        self.NY = 100
        self.Iter = 207
        self.dx = 0.001
        self.dy = 0.001
        self.co = 2.997925e8
        self.dt = 1 / (self.co * np.sqrt(1.0 / (self.dx**2) + 1.0 / (self.dy**2)))
        self.eo = 8.854187817e-12

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Gets item for dataloader"""
        file = self.files[index]
        fields = {}
        with h5py.File(file, "r") as h5file:
            for field_name in self.field_names:
                if field_name in h5file:
                    fields[field_name] = h5file[field_name][...]  # Load as numpy array
                    # Check for invalid values
                    if np.any(np.isnan(fields[field_name])) or np.any(np.isinf(fields[field_name])):
                        logger.warning(f"NaN or Inf in {field_name} for file {file}")
                else:
                    logger.warning(f"Field name {field_name} not found in {file}")

            # Derive material fields
            relative_permittivity = fields.get("relative_permittivity", 1.0)
            sigma = fields.get("sigma", 0.0)
            fields["ER"] = np.ones((self.NX, self.NY)) * relative_permittivity * self.eo
            fields["SIGMA"] = np.ones((self.NX, self.NY)) * sigma

            # Validate cylinder parameters
            CenterX = fields.get("CenterX", 50)
            CenterY = fields.get("CenterY", 50)
            Radius = fields.get("Radius", 10)

            # Convert to scalar and validate
            try:
                CenterX = float(CenterX)
                CenterY = float(CenterY)
                Radius = float(Radius)
            except (TypeError, ValueError):
                logger.error(f"Invalid CenterX, CenterY, or Radius in {file}: {CenterX}, {CenterY}, {Radius}")
                CenterX, CenterY, Radius = 50, 50, 10  # Fallback values

            if not (0 <= CenterX <= self.NX and 0 <= CenterY <= self.NY and 0 < Radius <= min(self.NX, self.NY)/2):
                logger.error(f"Out-of-bounds cylinder parameters in {file}: CenterX={CenterX}, CenterY={CenterY}, Radius={Radius}")
                CenterX, CenterY, Radius = 50, 50, 10

            if np.isnan(CenterX) or np.isnan(CenterY) or np.isnan(Radius):
                logger.error(f"NaN in cylinder parameters in {file}: CenterX={CenterX}, CenterY={CenterY}, Radius={Radius}")
                CenterX, CenterY, Radius = 50, 50, 10

            # Reconstruct cylinder
            for i in range(int(CenterX - Radius - 1), int(CenterX + Radius + 2)):
                for j in range(int(CenterY - Radius - 1), int(CenterY + Radius + 2)):
                    if i >= 0 and i < self.NX and j >= 0 and j < self.NY:
                        dist = np.sqrt((i - CenterX) ** 2 + (j - CenterY) ** 2)
                        if np.isnan(dist) or np.isinf(dist):
                            logger.warning(f"Invalid distance calculation at ({i}, {j}) in {file}")
                            continue
                        if dist <= Radius:
                            fields["ER"][i, j] = relative_permittivity * self.eo
                            fields["SIGMA"][i, j] = sigma

            # Reconstruct source
            tw = 26.53e-12
            to = 4 * tw
            src = np.zeros(self.Iter + 1)
            for i in range(self.Iter + 1):
                t = i * self.dt
                src[i] = -2.0 * ((t - to) / tw) * np.exp(-((t - to) / tw) ** 2)
            fields["src"] = src

        return fields

    def get_coords(self, index):
        """Gets coordinates of t, x, y"""
        t = np.arange(self.Iter + 1) * self.dt  # [208]
        x = np.arange(self.NX) * self.dx  # [100]
        y = np.arange(self.NY) * self.dy  # [100]
        return t, x, y


if __name__ == "__main__":
    dataset = FDTD2DDataset(
        data_path="hdf5_files",
        output_names="results_*.h5",
    )
    print(f"Found {len(dataset.files)} files: {dataset.files[:5]}...")
    data = dataset[0]
    print(f"Fields: {list(data.keys())}")
    t, x, y = dataset.get_coords(0)
    print(f"t shape: {t.shape}, x shape: {x.shape}, y shape: {y.shape}")