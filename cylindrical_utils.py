import torch
from torchsparse import SparseTensor

import kit.io as io
from cylindrical import CylindricalGridCfg, cart_to_cyl_grid


class PCDataset:
    """
    RENO 数据集（柱坐标版本）
    只改输入坐标系： (x,y,z) -> (r, a, z) 柱坐标网格
    核心网络不变
    """

    def __init__(
        self,
        file_path_ls,
        posQ: int = 4,
        is_pre_quantized: bool = False,
        theta_scale_mm: float = 2000.0,
        shift_mm: float = 131072.0,
        do_unique: bool = True,
    ):
        self.files = io.read_point_clouds(file_path_ls)
        self.is_pre_quantized = is_pre_quantized
        self.cfg = CylindricalGridCfg(
            posQ=float(posQ),
            theta_scale_mm=float(theta_scale_mm),
            shift_mm=float(shift_mm),
            do_unique=bool(do_unique),
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        xyz = torch.tensor(self.files[idx], dtype=torch.float32)  # (N,3)

        # (x,y,z) -> (r,a,z) grid coords (int32)
        coords = cart_to_cyl_grid(xyz, self.cfg, is_pre_quantized=self.is_pre_quantized)

        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        input_tensor = SparseTensor(coords=coords, feats=feats)
        return {"input": input_tensor}