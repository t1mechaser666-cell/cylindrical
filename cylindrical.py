"""
cylindrical_coords.py

将点云从笛卡尔坐标 (x,y,z) 映射到“柱坐标网格” (r, a, z)，并且保证：
1) 输出是 TorchSparse 可用的 int grid coords；
2) 不改动 RENO/FOG/FCG 等核心网络逻辑，只改坐标系；
3) 沿用原工程的 posQ 与 shift=131072 的量化风格。

说明：
- a 不是直接存储弧度 theta，而是 a_mm = theta * theta_scale_mm
  把角度线性化成“伪长度”(mm)轴，从而仍可用 dyadic 2×2×2 层级。
- 解码时 theta = a_mm / theta_scale_mm，再还原 (x,y,z)。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CylindricalGridCfg:
    """柱坐标网格参数（长度单位：mm）"""
    posQ: float
    theta_scale_mm: float = 2000.0
    shift_mm: float = 131072.0
    # 强烈建议 True：去掉重复 voxel，避免稀疏卷积/FOG 出现重复坐标导致异常
    do_unique: bool = True


def _to_mm(xyz: torch.Tensor, is_pre_quantized: bool) -> torch.Tensor:
    """
    统一到 mm：
    - is_pre_quantized=False：输入通常是 meters（如 KITTI .bin），乘 1000。
    - is_pre_quantized=True：默认输入已是 mm（如 Ford 或你自己的预量化数据）。
    """
    xyz = xyz.to(torch.float32)
    return xyz if is_pre_quantized else (xyz / 0.001)


def cart_to_cyl_grid(
    xyz: torch.Tensor,
    cfg: CylindricalGridCfg,
    *,
    is_pre_quantized: bool = False,
) -> torch.Tensor:
    """
    (x,y,z) -> 柱坐标网格 int coords

    Args:
        xyz: (N,3) float tensor. meters 或 mm 取决于 is_pre_quantized
        cfg: CylindricalGridCfg
        is_pre_quantized: True 表示 xyz 已是 mm

    Returns:
        coords: (M,3) int32 tensor，TorchSparse 可直接作为 SparseTensor.coords
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz should be (N,3), got {tuple(xyz.shape)}")

    xyz_mm = _to_mm(xyz, is_pre_quantized=is_pre_quantized)

    x_mm = xyz_mm[:, 0]
    y_mm = xyz_mm[:, 1]
    z_mm = xyz_mm[:, 2]

    r_mm = torch.sqrt(x_mm * x_mm + y_mm * y_mm + 1e-12)  # r>=0
    theta = torch.atan2(y_mm, x_mm)  # [-pi, pi]

    # 角度线性化为“伪长度轴”（mm）
    a_mm = theta * float(cfg.theta_scale_mm)

    cyl_mm = torch.stack((r_mm, a_mm, z_mm), dim=1)

    # shift + quantize -> int coords
    coords = torch.round((cyl_mm + float(cfg.shift_mm)) / float(cfg.posQ)).to(torch.int32)

    if cfg.do_unique:
        coords = torch.unique(coords, dim=0)

    return coords


def cyl_grid_to_cart(
    coords: torch.Tensor,
    cfg: CylindricalGridCfg,
    *,
    is_pre_quantized: bool = False,
) -> torch.Tensor:
    """
    柱坐标网格 int coords -> (x,y,z)

    Args:
        coords: (N,3) 或 (N,4)（含 batch idx），若 (N,4) 自动取 [:,1:]
        cfg: CylindricalGridCfg
        is_pre_quantized: False 输出 meters；True 输出 mm

    Returns:
        xyz: (N,3) float32
    """
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coords should be (N,3) or (N,4), got {tuple(coords.shape)}")

    coords3 = coords[:, 1:] if coords.shape[1] == 4 else coords

    cyl_mm = coords3.to(torch.float32) * float(cfg.posQ) - float(cfg.shift_mm)

    r_mm = torch.clamp(cyl_mm[:, 0], min=0.0)
    a_mm = cyl_mm[:, 1]
    z_mm = cyl_mm[:, 2]

    theta = a_mm / float(cfg.theta_scale_mm)
    x_mm = r_mm * torch.cos(theta)
    y_mm = r_mm * torch.sin(theta)

    xyz_mm = torch.stack((x_mm, y_mm, z_mm), dim=1)

    return xyz_mm if is_pre_quantized else (xyz_mm * 0.001)