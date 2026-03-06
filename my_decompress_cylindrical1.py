import os
import time
import random
import argparse

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchac

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

from my_network_cylindrical import Network

import kit.io as io
import kit.op as op

from cylindrical1 import CylindricalGridCfg, cyl_grid_to_cart

random.seed(1)
np.random.seed(1)
device = 'cuda'

# set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)


def load_ckpt(net: torch.nn.Module, ckpt_path: str):
    """Compat with both weight-only and full-state checkpoints."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        net.load_state_dict(ckpt['model'], strict=True)
    else:
        net.load_state_dict(ckpt, strict=True)


parser = argparse.ArgumentParser(
    prog='decompress.py',
    description='Decompress point cloud geometry (Cylindrical coords, network unchanged).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', default='./my_test/cy1_KITTI14999_compressed/*.bin', help='Glob pattern for input bin files.')
parser.add_argument('--output_folder', default='./my_test/cy1_KITTI14999_decompressed/', help='Folder to save decompressed ply files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the original point cloud is pre quantized.")

# Fallback options for legacy bin format (old header only stores posQ/theta_scale_mm)
parser.add_argument('--legacy_shift_mm', type=float, default=131072.0, help='Shift for legacy files.')
parser.add_argument('--legacy_use_log_r', type=int, default=0, choices=[0, 1], help='Legacy files radial mode fallback.')
parser.add_argument('--legacy_log_r_scale_mm', type=float, default=1000.0, help='Legacy files log-r scale fallback.')

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)
parser.add_argument('--ckpt', help='Checkpoint load path.', default='./my_model/cy1_KITTI14999/best_model.pt')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

file_path_ls = glob(args.input_glob)

# network
net = Network(channels=args.channels, kernel_size=args.kernel_size)
load_ckpt(net, args.ckpt)
net.cuda().eval()

# warm up
random_coords = torch.randint(low=0, high=65536, size=(4096, 3)).int().cuda()
net(SparseTensor(
        coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
        feats=torch.ones((4096, 1), device='cuda')
    ).cuda()
)

dec_time_ls = []

with torch.no_grad():
    for file_path in tqdm(file_path_ls):
        file_name = os.path.split(file_path)[-1]
        decompressed_file_path = os.path.join(args.output_folder, file_name + '.ply')

        # ==============================
        # 1) Read bin
        # ==============================
        with open(file_path, 'rb') as f:
            head4 = f.read(4)

            if head4 == b'CYL2':
                hdr = np.frombuffer(f.read(24), dtype=np.float32)
                if hdr.size != 6:
                    raise RuntimeError(f'Invalid CYL2 header in {file_path}')

                q_r_mm, q_a_mm, q_z_mm, theta_scale_mm, shift_mm, log_r_scale_mm = [float(v) for v in hdr]
                use_log_r = bool(np.frombuffer(f.read(1), dtype=np.uint8)[0])
            else:
                # legacy format: first 4 bytes are [posQ(float16), theta_scale_mm(float16)]
                legacy = np.frombuffer(head4, dtype=np.float16)
                if legacy.size != 2:
                    raise RuntimeError(f'Invalid legacy header in {file_path}')
                posQ = float(legacy[0])
                theta_scale_mm = float(legacy[1])

                q_r_mm = posQ
                q_a_mm = posQ
                q_z_mm = posQ
                shift_mm = float(args.legacy_shift_mm)
                log_r_scale_mm = float(args.legacy_log_r_scale_mm)
                use_log_r = bool(args.legacy_use_log_r)

            base_x_len = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            base_x_coords = np.frombuffer(f.read(base_x_len * 4 * 3), dtype=np.int32)
            base_x_feats = np.frombuffer(f.read(base_x_len * 1), dtype=np.uint8)
            byte_stream = f.read()

        cyl_cfg = CylindricalGridCfg(
            posQ=float(q_z_mm),
            theta_scale_mm=float(theta_scale_mm),
            shift_mm=float(shift_mm),
            q_r_mm=float(q_r_mm),
            q_a_mm=float(q_a_mm),
            q_z_mm=float(q_z_mm),
            use_log_r=bool(use_log_r),
            log_r_scale_mm=float(log_r_scale_mm),
            do_unique=False,
        )

        dec_time_start = time.time()

        # ==============================
        # 2) Decompress (still in cyl-grid)
        # ==============================
        base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=device, dtype=torch.int32)
        base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=device, dtype=torch.uint8)

        x = SparseTensor(
            coords=torch.cat((base_x_feats * 0, base_x_coords), dim=-1),
            feats=base_x_feats
        ).cuda()

        byte_stream_ls = op.unpack_byte_stream(byte_stream)

        for byte_stream_idx in range(0, len(byte_stream_ls), 2):
            byte_stream_s0 = byte_stream_ls[byte_stream_idx]
            byte_stream_s1 = byte_stream_ls[byte_stream_idx + 1]

            # prior embedding
            x_O = x.feats.long()
            x.feats = net.prior_embedding(x_O).view(-1, net.channels)
            x = net.prior_resnet(x)

            # target embedding
            x_up_C, x_up_F = net.fcg(x.coords, x_O, x_F=x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            # stage0 decode
            x_up_O_prob_s0 = net.pred_head_s0(x_up.feats)
            x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1] * 0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
            x_up_O_cdf_s0_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True).cpu()
            x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm, byte_stream_s0).cuda()

            # stage1 decode
            x_up_O_prob_s1 = net.pred_head_s1(x_up.feats + net.pred_head_s1_emb(x_up_O_s0.long()))
            x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1] * 0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
            x_up_O_cdf_s1_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s1, True).cpu()
            x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm, byte_stream_s1).cuda()

            x_up_O = x_up_O_s1 * 16 + x_up_O_s0
            x = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1)).cuda()

        # decode the last layer -> full-res cyl-grid coords (N,4)
        scan_grid = net.fcg(x.coords, x.feats)

        # ==============================
        # 3) cyl-grid -> cartesian xyz
        # ==============================
        scan_xyz = cyl_grid_to_cart(scan_grid, cyl_cfg, is_pre_quantized=args.is_data_pre_quantized)

        dec_time_end = time.time()
        dec_time_ls.append(dec_time_end - dec_time_start)

        io.save_ply_ascii_geo(scan_xyz.float().cpu().numpy(), decompressed_file_path)

print('Total: {total_n:d} | Decode Time:{dec_time:.3f} | Max GPU Memory:{memory:.2f}MB'.format(
    total_n=len(dec_time_ls),
    dec_time=np.array(dec_time_ls).mean() if len(dec_time_ls) else 0.0,
    memory=torch.cuda.max_memory_allocated()/1024/1024
))
