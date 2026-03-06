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

from cylindrical import CylindricalGridCfg, cart_to_cyl_grid

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
    prog='compress.py',
    description='Compress point cloud geometry (Cylindrical coords, network unchanged).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', default='/home/zyn/pccdata/KITTI14999_preprocess/testing/*.ply', help='Glob pattern for input point clouds.')
parser.add_argument('--output_folder', default='./my_test/cy1_KITTI14999_compressed/', help='Folder to save compressed bin files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the input data is pre quantized.")

# Base quantization step (fallback for all axes)
parser.add_argument('--posQ', default=8.0, type=float, help='Base quantization step in mm.')

# Cylindrical quantization parameters
parser.add_argument('--theta_scale_mm', type=float, default=4096.0, help='a_mm = theta(rad) * theta_scale_mm.')
parser.add_argument('--shift_mm', type=float, default=131072.0, help='Shift before quantization.')
parser.add_argument('--q_r_mm', type=float, default=None, help='Quantization step for radial axis (after optional log mapping).')
parser.add_argument('--q_a_mm', type=float, default=None, help='Quantization step for angular mapped axis a_mm.')
parser.add_argument('--q_z_mm', type=float, default=None, help='Quantization step for z axis.')
parser.add_argument('--use_log_r', type=int, default=1, choices=[0, 1], help='Use logarithmic radial partitioning (paper style).')
parser.add_argument('--log_r_scale_mm', type=float, default=1000.0, help='Scale s in r_enc = log(1 + r/s) * s.')

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)
parser.add_argument('--ckpt', help='Checkpoint load path.', default='./my_model/cy1_KITTI14999/best_model.pt')

parser.add_argument('--num_samples', default=-1, type=int, help='Random choose some samples for quick test. [-1 means test all data]')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

file_path_ls = glob(args.input_glob, recursive=True)

# random choose some samples for quick test
if args.num_samples != -1:
    np.random.shuffle(file_path_ls)
    file_path_ls = file_path_ls[:args.num_samples]

# reading point cloud using multithread
xyz_ls = io.read_point_clouds(file_path_ls)

# cylindrical grid config
cyl_cfg = CylindricalGridCfg(
    posQ=float(args.posQ),
    theta_scale_mm=float(args.theta_scale_mm),
    shift_mm=float(args.shift_mm),
    q_r_mm=args.q_r_mm,
    q_a_mm=args.q_a_mm,
    q_z_mm=args.q_z_mm,
    use_log_r=bool(args.use_log_r),
    log_r_scale_mm=float(args.log_r_scale_mm),
    do_unique=True,
)

q_r_mm, q_a_mm, q_z_mm = cyl_cfg.quant_steps()
header_f32 = np.array(
    [
        q_r_mm,
        q_a_mm,
        q_z_mm,
        float(cyl_cfg.theta_scale_mm),
        float(cyl_cfg.shift_mm),
        float(cyl_cfg.log_r_scale_mm),
    ],
    dtype=np.float32,
)
header_log_flag = np.array([1 if cyl_cfg.use_log_r else 0], dtype=np.uint8)

# network
net = Network(channels=args.channels, kernel_size=args.kernel_size)
load_ckpt(net, args.ckpt)
net.cuda().eval()

# warm up (kernel-map init)
random_coords = torch.randint(low=0, high=65536, size=(4096, 3)).int().cuda()
net(SparseTensor(
        coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
        feats=torch.ones((4096, 1), device='cuda')
    ).cuda()
)

enc_time_ls, bpp_ls = [], []

with torch.no_grad():
    for file_idx in tqdm(range(len(file_path_ls))):
        file_path = file_path_ls[file_idx]
        file_name = os.path.split(file_path)[-1]
        compressed_file_path = os.path.join(args.output_folder, file_name + '.bin')

        # ==============================
        # 1) (x,y,z) -> cylindrical grid coords
        # ==============================
        xyz = torch.tensor(xyz_ls[file_idx], dtype=torch.float32)
        coords3 = cart_to_cyl_grid(xyz, cyl_cfg, is_pre_quantized=args.is_data_pre_quantized)
        N = int(coords3.shape[0])
        coords4 = torch.cat((coords3[:, 0:1]*0, coords3), dim=-1).int()
        feats = torch.ones((coords4.shape[0], 1), dtype=torch.float32)
        x = SparseTensor(coords=coords4, feats=feats).cuda()

        torch.cuda.synchronize()
        enc_time_start = time.time()

        # ==============================
        # 2) Preprocessing (FOG multi-scale)
        # ==============================
        data_ls = []
        while True:
            x = net.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone()))
            if x.coords.shape[0] < 64:
                break
        data_ls = data_ls[::-1]

        # ==============================
        # 3) NN Inference + Arithmetic Coding
        # ==============================
        byte_stream_ls = []
        for depth in range(len(data_ls)-1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth+1]
            gt_x_up_C, gt_x_up_O = op.sort_CF(gt_x_up_C, gt_x_up_O)

            # embedding prior scale feats
            x_F = net.prior_embedding(x_O.int()).view(-1, net.channels)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = net.prior_resnet(x)

            # target embedding
            x_up_C, x_up_F = net.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            # bit-wise two-stage coding
            gt_x_up_O_s0 = torch.remainder(gt_x_up_O, 16)
            gt_x_up_O_s1 = torch.div(gt_x_up_O, 16, rounding_mode='floor')

            x_up_O_prob_s0 = net.pred_head_s0(x_up.feats)
            x_up_O_prob_s1 = net.pred_head_s1(
                x_up.feats + net.pred_head_s1_emb(gt_x_up_O_s0[:, 0].long())
            )

            # AE
            x_up_O_prob = torch.cat((x_up_O_prob_s0, x_up_O_prob_s1), dim=0)
            gt_x_up_O_cat = torch.cat((gt_x_up_O_s0, gt_x_up_O_s1), dim=0)

            # get cdf
            x_up_O_cdf = torch.cat((x_up_O_prob[:, 0:1]*0, x_up_O_prob.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf = torch.clamp(x_up_O_cdf, min=0, max=1)
            x_up_O_cdf_norm = op._convert_to_int_and_normalize(x_up_O_cdf, True)

            # cdf to cpu
            x_up_O_cdf_norm = x_up_O_cdf_norm.cpu()
            gt_x_up_O_cat = gt_x_up_O_cat[:, 0].to(torch.int16).cpu()

            # coding
            half = gt_x_up_O_cat.shape[0] // 2
            byte_stream_s0 = torchac.encode_int16_normalized_cdf(
                x_up_O_cdf_norm[:half],
                gt_x_up_O_cat[:half]
            )
            byte_stream_s1 = torchac.encode_int16_normalized_cdf(
                x_up_O_cdf_norm[half:],
                gt_x_up_O_cat[half:]
            )
            byte_stream_ls.append(byte_stream_s0)
            byte_stream_ls.append(byte_stream_s1)

        byte_stream = op.pack_byte_stream_ls(byte_stream_ls)

        torch.cuda.synchronize()
        enc_time_end = time.time()

        # ==============================
        # 4) Write bin
        # ==============================
        base_x_coords, base_x_feats = data_ls[0]
        base_x_len = base_x_coords.shape[0]
        base_x_coords = base_x_coords[:, 1:].cpu().numpy().astype(np.int32)
        base_x_feats = base_x_feats.cpu().numpy().astype(np.uint8)

        with open(compressed_file_path, 'wb') as f:
            # new header format (quantization config)
            f.write(b'CYL2')
            f.write(header_f32.tobytes())
            f.write(header_log_flag.tobytes())

            f.write(np.array(base_x_len, dtype=np.int32).tobytes())
            f.write(base_x_coords.tobytes())
            f.write(base_x_feats.tobytes())
            f.write(byte_stream)

        enc_time_ls.append(enc_time_end - enc_time_start)
        bpp_ls.append(op.get_file_size_in_bits(compressed_file_path) / max(1, N))

print('Total: {total_n:d} | Avg. Bpp:{bpp:.3f} | Encode time:{enc_time:.3f} | Max GPU Memory:{memory:.2f}MB'.format(
    total_n=len(enc_time_ls),
    bpp=np.array(bpp_ls).mean() if len(bpp_ls) else 0.0,
    enc_time=np.array(enc_time_ls).mean() if len(enc_time_ls) else 0.0,
    memory=torch.cuda.max_memory_allocated()/1024/1024
))
