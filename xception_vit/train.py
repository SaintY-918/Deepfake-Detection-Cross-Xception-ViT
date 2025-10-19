# train.py (適用於單分支 Xception-ViT)

import torch
import torch.nn as nn
import os
import argparse
import yaml
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import math
import collections
import torch.optim as optim
import time # 用於建立獨特的日誌名稱

# 導入 TensorBoard 的 SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# --- 1. 匯入新的單分支模型 ---
from model import XceptionViT 
from deepfakes_dataset import DeepFakesDataset, collate_fn_filter_none
from utils import get_n_params, check_correct

def read_frames_from_videos(video_paths, frames_per_video, label):
    all_frames = []
    for video_path in video_paths:
        frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path)])
        if not frame_files:
            continue
        step = len(frame_files) // frames_per_video if len(frame_files) > frames_per_video else 1
        sampled_files = frame_files[::step][:frames_per_video]
        for frame_path in sampled_files:
            all_frames.append((frame_path, float(label)))
    return all_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 Xception-ViT 單分支模型")
    parser.add_argument('--config', type=str, default='architecture.yaml', help="設定檔路徑")
    parser.add_argument('--dataset', type=str, default='All', help="偽造資料集類型")
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--patience', type=int, default=10)
    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用設備: {device} ---")

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # --- 2. 更改 TensorBoard 日誌路徑 ---
    log_dir = os.path.join("runs", f"xception_vit_bs{config['training']['bs']}_lr{config['training']['lr']}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)
    print(f"--- TensorBoard 日誌將儲存到: {log_dir} ---")

    MODELS_PATH = "models"
    BASE_DIR = '../prepared_dataset/'
    TRAINING_DIR = os.path.join(BASE_DIR, "training_set")
    VALIDATION_DIR = os.path.join(BASE_DIR, "validation_set")
    
    print("--- 準備資料集 (1:1 平衡策略) ---")
    frames_per_video = config['training']['frames-per-video']
    
    # (資料集讀取邏輯保持不變)
    fake_types_in_train = [d for d in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, d)) and d != 'Original']
    if opt.dataset != 'All':
        fake_types_in_train = [opt.dataset]
    num_real_train_videos = len(os.listdir(os.path.join(TRAINING_DIR, 'Original')))
    total_real_train_frames = num_real_train_videos * frames_per_video
    num_fake_train_videos = sum(len(os.listdir(os.path.join(TRAINING_DIR, fake_type))) for fake_type in fake_types_in_train)
    frames_per_fake_train = max(1, round(total_real_train_frames / num_fake_train_videos)) if num_fake_train_videos > 0 else 0
    train_paths = []
    original_train_videos = [os.path.join(TRAINING_DIR, 'Original', v) for v in os.listdir(os.path.join(TRAINING_DIR, 'Original'))]
    train_paths.extend(read_frames_from_videos(original_train_videos, frames_per_video, 0.0))
    for fake_type in fake_types_in_train:
        fake_video_paths = [os.path.join(TRAINING_DIR, fake_type, v) for v in os.listdir(os.path.join(TRAINING_DIR, fake_type))]
        train_paths.extend(read_frames_from_videos(fake_video_paths, frames_per_fake_train, 1.0))
    
    fake_types_in_val = [d for d in os.listdir(VALIDATION_DIR) if os.path.isdir(os.path.join(VALIDATION_DIR, d)) and d != 'Original']
    if opt.dataset != 'All':
        fake_types_in_val = [opt.dataset]
    num_real_val_videos = len(os.listdir(os.path.join(VALIDATION_DIR, 'Original')))
    total_real_val_frames = num_real_val_videos * frames_per_video
    num_fake_val_videos = sum(len(os.listdir(os.path.join(VALIDATION_DIR, fake_type))) for fake_type in fake_types_in_val)
    frames_per_fake_val = max(1, round(total_real_val_frames / num_fake_val_videos)) if num_fake_val_videos > 0 else 0
    val_paths = []
    original_val_videos = [os.path.join(VALIDATION_DIR, 'Original', v) for v in os.listdir(os.path.join(VALIDATION_DIR, 'Original'))]
    val_paths.extend(read_frames_from_videos(original_val_videos, frames_per_video, 0.0))
    for fake_type in fake_types_in_val:
        fake_video_paths = [os.path.join(VALIDATION_DIR, fake_type, v) for v in os.listdir(os.path.join(VALIDATION_DIR, fake_type))]
        val_paths.extend(read_frames_from_videos(fake_video_paths, frames_per_fake_val, 1.0))
    
    train_samples = len(train_paths)
    val_samples = len(val_paths)
    print(f"訓練樣本數 (影格): {train_samples}, 驗證樣本數 (影格): {val_samples}")
    print("訓練集統計:", collections.Counter(label for _, label in train_paths))
    print("驗證集統計:", collections.Counter(label for _, label in val_paths))
    
    # --- 3. 實例化新的單分支模型 ---
    model = XceptionViT(config, pretrained=True)
    print("--- 已載入單分支模型 (XceptionViT) ---")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    
    # (排程器邏輯保持不變)
    scheduler_name = config['training'].get('scheduler', 'cosine')
    if scheduler_name.lower() == 'cosine':
        t_max = config['training'].get('t_max', opt.num_epochs) 
        eta_min = config['training'].get('eta_min', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        print(f"--- 使用 CosineAnnealingLR 排程器 (T_max={t_max}, eta_min={eta_min}) ---")
    elif scheduler_name.lower() == 'steplr':
        step_size = config['training'].get('step-size', 15)
        gamma = config['training'].get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"--- 使用 StepLR 排程器 (step_size={step_size}, gamma={gamma}) ---")
    else:
        scheduler = None
        print("--- 未設定或設定了無效的 Scheduler，將不使用學習率排程 ---")
    
    scaler = GradScaler()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    train_dataset = DeepFakesDataset(train_paths, image_size=config['model']['image-size'], mode='train')
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=opt.workers, collate_fn=collate_fn_filter_none)
    
    val_dataset = DeepFakesDataset(val_paths, image_size=config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=opt.workers, collate_fn=collate_fn_filter_none)
    
    print("模型總參數數量:", get_n_params(model))
    model.to(device)
    
    # (寫入模型架構圖邏輯保持不變)
    try:
        images_batch, _ = next(iter(dl))
        writer.add_graph(model, images_batch.to(device))
        print("--- 模型架構圖已成功寫入 TensorBoard ---")
    except Exception as e:
        print(f"無法寫入模型架構圖 (這通常不影響訓練): {e}")

    not_improved_loss = 0
    previous_loss = math.inf
    
    # (訓練迴圈邏輯保持不變)
    for t in range(opt.num_epochs):
        if not_improved_loss >= opt.patience:
            print(f"Early stopping at epoch {t+1}")
            break
        
        total_loss, train_correct = 0, 0
        model.train()
        loop = tqdm(dl, desc=f"EPOCH #{t+1}/{opt.num_epochs} [TRAIN]")
        for index, (images, labels) in enumerate(loop):
            if not images.numel(): continue
            images, labels = images.to(device), labels.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            with autocast():
                y_pred = model(images)
                loss = loss_fn(y_pred, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            corrects, _, _ = check_correct(y_pred, labels)
            train_correct += corrects
            loop.set_postfix(loss=loss.item(), acc=f"{(train_correct / ((index + 1) * config['training']['bs'])):.4f}")

        if scheduler:
            scheduler.step()
        
        total_val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_dl:
                if not val_images.numel(): continue
                val_images, val_labels = val_images.to(device), val_labels.unsqueeze(1).to(device)
                with autocast():
                    val_pred = model(val_images)
                    val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += val_loss.item()
                corrects, _, _ = check_correct(val_pred, val_labels)
                val_correct += corrects
        
        avg_loss = total_loss / len(dl) if len(dl) > 0 else 0
        avg_acc = train_correct / train_samples if train_samples > 0 else 0
        avg_val_loss = total_val_loss / len(val_dl) if len(val_dl) > 0 else 0
        avg_val_acc = val_correct / val_samples if val_samples > 0 else 0
        current_lr = scheduler.get_last_lr()[0] if scheduler else config['training']['lr']
        
        # (TensorBoard 寫入邏輯保持不變)
        writer.add_scalar('LearningRate', current_lr, t + 1)
        writer.add_scalars('Loss', { 'train': avg_loss, 'validation': avg_val_loss }, t + 1)
        writer.add_scalars('Accuracy', { 'train': avg_acc, 'validation': avg_val_acc }, t + 1)
        
        print(f"EPOCH #{t+1}/{opt.num_epochs} -> loss: {avg_loss:.4f}, acc: {avg_acc:.4f} || val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f} || lr: {current_lr:.6f}")
        
        if avg_val_loss < previous_loss:
            previous_loss = avg_val_loss
            not_improved_loss = 0
            if not os.path.exists(MODELS_PATH):
                os.makedirs(MODELS_PATH)
            
            # --- 4. 更改模型儲存路徑 ---
            model_save_path = os.path.join(MODELS_PATH, f"xception_vit_best_{opt.dataset}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saving best model to {model_save_path}")
        else:
            not_improved_loss += 1
            print(f"Validation loss did not improve. Counter: {not_improved_loss}/{opt.patience}")

    writer.close()
    print("--- TensorBoard 日誌寫入完畢 ---")