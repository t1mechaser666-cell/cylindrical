import os
import re
import math
import random
import argparse
import datetime
import logging
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn

from cylindrical_utils import PCDataset
from my_network_cylindrical import Network

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------
# Helper: Learning Rate Scheduler (Warmup + Cosine)
# ----------------------
def get_lr(current_step, total_steps, warmup_steps, base_lr, min_lr=5e-4):
    if current_step < warmup_steps:
        return base_lr * (current_step / max(1, warmup_steps))
    else:
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0) 
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ----------------------
# Logger Setup
# ----------------------
def setup_logger(save_path):
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'training.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

# ----------------------
# Load Data Resources
# ----------------------
def get_data_resources(args, logger):
    train_files = np.array(glob(args.training_data, recursive=True))
    test_files = np.array(glob(args.testing_data, recursive=True))

    if args.valid_samples != '':
        valid_sample_names = np.loadtxt(args.valid_samples, dtype=str)
        train_files = [f for f in train_files if os.path.splitext(os.path.basename(f))[0] in valid_sample_names]
        train_files = np.array(train_files)
        test_files = [f for f in test_files if os.path.splitext(os.path.basename(f))[0] in valid_sample_names]
        test_files = np.array(test_files)

    logger.info(f"[DATA] Training files: {len(train_files)}")
    logger.info(f"[DATA] Testing files (Full Pool): {len(test_files)}")

    train_dataset = PCDataset(train_files, is_pre_quantized=args.is_data_pre_quantized, posQ=args.posQ, 
                              theta_scale_mm=args.theta_scale_mm, shift_mm=args.shift_mm)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sparse_collate_fn,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, test_files

# ----------------------
# Build Model
# ----------------------
def build_model(args, device):
    conv_config = F.conv_config.get_default_conv_config()
    conv_config.kmap_mode = "hashmap"
    F.conv_config.set_global_conv_config(conv_config)

    model = Network(channels=args.channels, kernel_size=args.kernel_size).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    return model, optimizer, scaler

# ----------------------
# Validate Function
# ----------------------
def validate(model, val_loader, device):
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        # mininterval=2.0 减少刷新频率
        for data in tqdm(val_loader, desc="Validating", leave=False, mininterval=2.0):
            x = data['input'].to(device)
            with autocast():
                loss = model(x)
            val_losses.append(loss.item())
    
    model.train()
    return np.mean(val_losses)

# ----------------------
# Find Latest Checkpoint
# ----------------------
def find_latest_checkpoint(save_folder):
    # 优先寻找 epoch 结尾的 checkpoint
    epoch_ckpts = glob(os.path.join(save_folder, 'ckpt_epoch*.pt'))
    if epoch_ckpts:
        epoch_ckpts = sorted(epoch_ckpts, key=lambda x: int(re.findall(r'epoch(\d+)', x)[0]))
        return epoch_ckpts[-1]
    
    # 其次寻找 step 结尾的 checkpoint
    step_ckpts = glob(os.path.join(save_folder, 'ckpt_step*.pt'))
    if step_ckpts:
        step_ckpts = sorted(step_ckpts, key=lambda x: int(re.findall(r'step(\d+)', x)[0]))
        return step_ckpts[-1]

    return None

# ----------------------
# Training Loop
# ----------------------
def train(model, optimizer, scaler, train_loader, test_files_pool, args, logger, device):
    global_step = 0
    start_epoch = 1
    best_val_loss = float('inf')
    
    warmup_steps = int(args.max_steps * 0.05)
    logger.info(f"[SCHEDULER] Warmup steps: {warmup_steps}, Total steps: {args.max_steps}")

    # ================= Checkpoint Loading Logic =================
    ckpt_path = find_latest_checkpoint(args.model_save_folder)
    if ckpt_path:
        logger.info(f"[RESUME] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 情况 A: 这是一个包含完整状态的新版 Checkpoint (字典形式)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            logger.info("[RESUME] Detected full state checkpoint.")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            global_step = checkpoint['global_step']
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
        
        # 情况 B: 这是一个只包含权重的旧版 Checkpoint (state_dict)
        else:
            logger.info("[RESUME] Detected legacy weight-only checkpoint.")
            model.load_state_dict(checkpoint) # checkpoint 本身就是 state_dict
            start_epoch = 1
            global_step = 0
    else:
        logger.info("[RESUME] No checkpoint found. Starting from scratch.")
    # ============================================================

    try:
        for epoch in range(start_epoch, 9999):
            current_lr = get_lr(global_step, args.max_steps, warmup_steps, args.learning_rate)
            logger.info(f"Epoch {epoch} started. (Global Step: {global_step}, Start LR: {current_lr:.9f})")
            
            # --- Training Phase ---
            model.train()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="it", mininterval=2.0)
            epoch_train_losses = []

            for data in pbar:
                current_lr = get_lr(global_step, args.max_steps, warmup_steps, args.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                x = data['input'].to(device)

                optimizer.zero_grad()
                with autocast():
                    loss = model(x)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                global_step += 1
                loss_val = loss.item()
                epoch_train_losses.append(loss_val)
                
                # 更新进度条信息
                pbar.set_postfix(loss=loss_val, lr=current_lr)

                if global_step >= args.max_steps:
                    raise StopIteration

            # --- Epoch End: Logging ---
            avg_train_loss = np.mean(epoch_train_losses)
            
            # --- Validation Phase ---
            logger.info(f"Preparing validation set for Epoch {epoch}...")
            
            num_val = max(1, int(len(test_files_pool) * 0.1))
            current_val_files = np.random.choice(test_files_pool, size=num_val, replace=False)
            
            val_dataset = PCDataset(current_val_files, is_pre_quantized=args.is_data_pre_quantized, posQ=args.posQ, 
                                    theta_scale_mm=args.theta_scale_mm, shift_mm=args.shift_mm)
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=sparse_collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
            avg_val_loss = validate(model, val_loader, device)
            
            logger.info(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.9f}")

            # --- Saving Logic (Updated to save FULL STATE) ---
            checkpoint_dict = {
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_loss': best_val_loss
            }

            ckpt_save_path = os.path.join(args.model_save_folder, f'ckpt_epoch{epoch}.pt')
            torch.save(checkpoint_dict, ckpt_save_path) # 🔥 保存字典
            logger.info(f"[CKPT] Saved checkpoint: {ckpt_save_path}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(args.model_save_folder, 'best_model.pt')
                torch.save(checkpoint_dict, best_path) # 🔥 保存字典
                logger.info(f"✅ [BEST] New Best Model Saved! Val Loss: {best_val_loss:.6f}")
                
            del val_loader, val_dataset

    except StopIteration:
        logger.info(f"Reached max training step: {args.max_steps}. Stopping.")

    except Exception as e:
        logger.exception(f"Training interrupted by error: {e}")

    finally:
        final_model_path = os.path.join(args.model_save_folder, 'final_model.pt')
        final_dict = {
            'epoch': epoch if 'epoch' in locals() else 0,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }
        torch.save(final_dict, final_model_path)
        logger.info(f"[FINAL] Final model saved to {final_model_path}")
        logger.info(f"[BEST] Best Val Loss achieved: {best_val_loss:.6f}")

# ----------------------
# Main Entry
# ----------------------
def main():
    parser = argparse.ArgumentParser(description='Point Cloud Compression Training')
    
    # Data parameters
    parser.add_argument('--training_data', default='/home/zyn/pccdata/KITTI14999/training/velodyne/*.bin')
    parser.add_argument('--testing_data', default='/home/zyn/pccdata/KITTI14999/testing/velodyne/*.bin')
    
    # Save & checkpoint
    parser.add_argument('--model_save_folder', default='./my_model/cy1_KITTI14999')
    parser.add_argument('--is_data_pre_quantized', type=bool, default=False)
    parser.add_argument('--valid_samples', type=str, default='')
    
    # Network parameters
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3) 
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_steps', type=int, default=666666)
    
    # Quantization parameters
    parser.add_argument('--posQ', type=float, default=8.0, help='Quantization step in mm for spatial grid')
    parser.add_argument('--theta_scale_mm', type=float, default=4096.0, help='Scaling factor for theta (angular quantization)')
    parser.add_argument('--shift_mm', type=float, default=131072.0, help='Shift for quantization')
    
    args = parser.parse_args()

    # Set random seed
    seed = 11
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = setup_logger(args.model_save_folder)
    logger.info(f"[INFO] Device = {device}")
  
    logger.info("[INFO] Loading data resources...")
    train_loader, test_files_pool = get_data_resources(args, logger)

    logger.info("[INFO] Building model...")
    model, optimizer, scaler = build_model(args, device)

    # ------------------------------------------------------------------
    logger.info("="*30 + " Model Architecture: Prior ResNet " + "="*30)
    if hasattr(model, 'prior_resnet'):
        logger.info(str(model.prior_resnet))
    else:
        logger.info("Attribute 'prior_resnet' not found in model.")
    logger.info("="*30 + " Model Architecture: Target ResNet " + "="*30)
    if hasattr(model, 'target_resnet'):
        logger.info(str(model.target_resnet))
    else:
        logger.info("Attribute 'target_resnet' not found in model.")
    logger.info("="*80)
    logger.info(f"[SCHEDULER] Using Warmup + Cosine Annealing Strategy.")
    # ------------------------------------------------------------------

    logger.info("[INFO] Starting training...")
    train(model, optimizer, scaler, train_loader, test_files_pool, args, logger, device)

if __name__ == '__main__':
    main()