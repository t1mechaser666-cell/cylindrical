import torch
from torchsparse import SparseTensor

import kit.io as io
from typing import Optional
from cylindrical1 import CylindricalGridCfg, cart_to_cyl_grid


class PCDataset:
    """RENO dataset wrapper with cylindrical-coordinate quantization."""

    def __init__(
        self,
        file_path_ls,
        posQ: int = 4,
        is_pre_quantized: bool = False,
        theta_scale_mm: float = 4096.0,
        shift_mm: float = 131072.0,
        q_r_mm: Optional[float] = None,
        q_a_mm: Optional[float] = None,
        q_z_mm: Optional[float] = None,
        use_log_r: bool = True,
        log_r_scale_mm: float = 1000.0,
        do_unique: bool = True,
    ):
        self.files = io.read_point_clouds(file_path_ls)
        self.is_pre_quantized = is_pre_quantized
        self.cfg = CylindricalGridCfg(
            posQ=float(posQ),
            theta_scale_mm=float(theta_scale_mm),
            shift_mm=float(shift_mm),
            q_r_mm=q_r_mm,
            q_a_mm=q_a_mm,
            q_z_mm=q_z_mm,
            use_log_r=bool(use_log_r),
            log_r_scale_mm=float(log_r_scale_mm),
            do_unique=bool(do_unique),
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        xyz = torch.tensor(self.files[idx], dtype=torch.float32)
        coords = cart_to_cyl_grid(xyz, self.cfg, is_pre_quantized=self.is_pre_quantized)

        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        input_tensor = SparseTensor(coords=coords, feats=feats)
        return {"input": input_tensor}
