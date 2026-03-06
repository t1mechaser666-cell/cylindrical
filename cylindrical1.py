"""
Cylindrical coordinate quantization helpers for LiDAR point clouds.

This module keeps RENO's sparse-octree style processing untouched and only
changes coordinate quantization:
1) Cartesian (x, y, z) -> cylindrical (r, a, z)
2) optional logarithmic radial mapping r -> log(1 + r / s) * s (paper-inspired)
3) independent quantization step per axis (Q_r, Q_a, Q_z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class CylindricalGridCfg:
    """Quantization config in millimeter domain."""

    posQ: float
    theta_scale_mm: float = 4096.0
    shift_mm: float = 131072.0

    # Optional per-axis quantization steps. If None or <=0, falls back to posQ.
    q_r_mm: Optional[float] = None
    q_a_mm: Optional[float] = None
    q_z_mm: Optional[float] = None

    # Paper-inspired option: partition radial axis logarithmically.
    use_log_r: bool = True
    log_r_scale_mm: float = 1000.0

    do_unique: bool = True

    def quant_steps(self) -> Tuple[float, float, float]:
        base = float(self.posQ)

        def _resolve(v: Optional[float]) -> float:
            if v is None:
                return base
            v = float(v)
            return v if v > 0 else base

        return _resolve(self.q_r_mm), _resolve(self.q_a_mm), _resolve(self.q_z_mm)


def _to_mm(xyz: torch.Tensor, is_pre_quantized: bool) -> torch.Tensor:
    """Normalize xyz units to millimeters."""
    xyz = xyz.to(torch.float32)
    return xyz if is_pre_quantized else (xyz * 1000.0)


def _radius_forward(r_mm: torch.Tensor, cfg: CylindricalGridCfg) -> torch.Tensor:
    """Optional log-radius mapping used before quantization."""
    r_mm = torch.clamp(r_mm, min=0.0)
    if not cfg.use_log_r:
        return r_mm

    s = max(float(cfg.log_r_scale_mm), 1e-6)
    return torch.log1p(r_mm / s) * s


def _radius_inverse(r_enc_mm: torch.Tensor, cfg: CylindricalGridCfg) -> torch.Tensor:
    """Inverse of optional log-radius mapping."""
    r_enc_mm = torch.clamp(r_enc_mm, min=0.0)
    if not cfg.use_log_r:
        return r_enc_mm

    s = max(float(cfg.log_r_scale_mm), 1e-6)
    return torch.expm1(r_enc_mm / s) * s


def cart_to_cyl_grid(
    xyz: torch.Tensor,
    cfg: CylindricalGridCfg,
    *,
    is_pre_quantized: bool = False,
) -> torch.Tensor:
    """(x, y, z) -> int cylindrical grid coordinates."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz should be (N,3), got {tuple(xyz.shape)}")

    xyz_mm = _to_mm(xyz, is_pre_quantized=is_pre_quantized)

    x_mm = xyz_mm[:, 0]
    y_mm = xyz_mm[:, 1]
    z_mm = xyz_mm[:, 2]

    r_mm = torch.sqrt(x_mm * x_mm + y_mm * y_mm + 1e-12)
    theta = torch.atan2(y_mm, x_mm)  # [-pi, pi]

    r_enc_mm = _radius_forward(r_mm, cfg)
    a_mm = theta * float(cfg.theta_scale_mm)

    cyl_mm = torch.stack((r_enc_mm, a_mm, z_mm), dim=1)

    q_r_mm, q_a_mm, q_z_mm = cfg.quant_steps()
    q = torch.tensor([q_r_mm, q_a_mm, q_z_mm], device=cyl_mm.device, dtype=torch.float32)

    coords = torch.round((cyl_mm + float(cfg.shift_mm)) / q).to(torch.int32)

    if cfg.do_unique:
        coords = torch.unique(coords, dim=0)

    return coords


def cyl_grid_to_cart(
    coords: torch.Tensor,
    cfg: CylindricalGridCfg,
    *,
    is_pre_quantized: bool = False,
) -> torch.Tensor:
    """int cylindrical grid coordinates -> (x, y, z)."""
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coords should be (N,3) or (N,4), got {tuple(coords.shape)}")

    coords3 = coords[:, 1:] if coords.shape[1] == 4 else coords

    q_r_mm, q_a_mm, q_z_mm = cfg.quant_steps()
    q = torch.tensor([q_r_mm, q_a_mm, q_z_mm], device=coords3.device, dtype=torch.float32)

    cyl_mm = coords3.to(torch.float32) * q - float(cfg.shift_mm)

    r_enc_mm = torch.clamp(cyl_mm[:, 0], min=0.0)
    a_mm = cyl_mm[:, 1]
    z_mm = cyl_mm[:, 2]

    r_mm = _radius_inverse(r_enc_mm, cfg)
    theta = a_mm / float(cfg.theta_scale_mm)

    x_mm = r_mm * torch.cos(theta)
    y_mm = r_mm * torch.sin(theta)

    xyz_mm = torch.stack((x_mm, y_mm, z_mm), dim=1)
    return xyz_mm if is_pre_quantized else (xyz_mm * 0.001)
