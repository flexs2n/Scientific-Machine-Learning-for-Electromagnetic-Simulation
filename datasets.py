import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import h5py


class FDTD2DDataset(Dataset):
    """Dataset for 2D TMz FDTD data from MATLAB .mat files"""

    def __init__(
        self,
        data_path,
        output_names="output_*.mat",
        field_names=["Ez", "Hx", "Hy"],
        num_train=None,
        num_test=None,
        use_train=True,
    ):
        self.data_path = data_path
        self.output_names = output_names
        raw_path = os.path.join(data_path, output_names)
        files_raw = sorted(glob.glob(raw_path))
        self.files_raw = files_raw
        self.num_files_raw = len(files_raw)
        self.field_names = field_names
        self.use_train = use_train

        if (num_train is None) or (num_train > self.num_files_raw):
            num_train = self.num_files_raw
        self.num_train = num_train
        self.train_files = self.files_raw[:num_train]
        if (num_test is None) or (num_test > (self.num_files_raw - num_train)):
            num_test = self.num_files_raw - num_train
        self.num_test = num_test
        self.test_end = num_train + num_test
        self.test_files = self.files_raw[num_train:self.test_end]
        self.files = self.train_files if self.use_train else self.test_files
        self.num_files = len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Gets item for dataloader"""
        file = self.files[index]
        fields = {}
        with h5py.File(file, "r") as h5file:
            for field_name in self.field_names:
                if field_name in h5file:
                    fields[field_name] = h5file[field_name][:]  # Load as numpy array
                else:
                    print(f"Field name {field_name} not found in {file}")
        return fields

    def get_coords(self, index):
        """Gets coordinates of t, x, y from the .mat file"""
        file = self.files[index]
        with h5py.File(file, "r") as h5file:
            t = h5file["t"][:] if "t" in h5file else np.arange(415) * 2.357e-12  # Default dt
            x = h5file["x"][:] if "x" in h5file else np.arange(200) * 0.001  # Default dx
            y = h5file["y"][:] if "y" in h5file else np.arange(200) * 0.001  # Default dy
        return t, x, y


if __name__ == "__main__":
    dataset = FDTD2DDataset(data_path="em_data")
    data = dataset[0]
    print(data.keys())
