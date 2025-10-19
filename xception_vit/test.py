# test.py (適用於單分支 Xception-ViT)

import torch
import os
import argparse
import yaml
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import collections

# --- 1. 匯入新的單分支模型 ---
from model import XceptionViT 
from deepfakes_dataset import DeepFakesDataset, collate_fn_filter_none


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="測試 Xception-ViT 單分支模型")
    
    parser.add_argument('--config', type=str, default='architecture.yaml', help="設定檔路徑")
    
    # 其他參數
    parser.add_argument('--dataset', type=str, default='All', help="要評估的偽造資料集類型 (e.g., FaceSwap, All)")
    parser.add_argument('--workers', default=8, type=int, help="DataLoader 的工作執行緒數量")
    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用設備: {device} ---")

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # --- 2. 實例化新的單分支模型 ---
    model = XceptionViT(config, pretrained=False)
    
    # --- 3. 載入對應的權重檔案 ---
    model_path = os.path.join("models", f"xception_vit_best_{opt.dataset}.pth")
    model.load_state_dict(torch.load(model_path))
    print(f"--- 已從 {model_path} 載入 Xception-ViT 模型權重 ---")
    
    model.to(device)
    model.eval()

    TEST_DIR = '../prepared_dataset/test_set'
    frames_per_video = config['training']['frames-per-video']
    
    print("--- 準備測試資料 ---")
    all_video_paths = []
    video_types_to_process = []

    if opt.dataset == 'All':
        video_types_to_process = os.listdir(TEST_DIR)
    else:
        video_types_to_process = ['Original', opt.dataset]
    
    for video_type in video_types_to_process:
        type_path = os.path.join(TEST_DIR, video_type)
        if os.path.isdir(type_path):
            all_video_paths.extend([os.path.join(type_path, v) for v in os.listdir(type_path)])

    # (測試迴圈和報告生成邏輯保持不變)
    results = []
    with torch.no_grad():
        for video_path in tqdm(all_video_paths, desc="正在測試"):
            video_name = os.path.basename(video_path)
            fake_type = os.path.basename(os.path.dirname(video_path))
            true_label = 0. if fake_type == 'Original' else 1.
            
            frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path)])
            if not frame_files: continue

            step = len(frame_files) // frames_per_video if len(frame_files) > frames_per_video else 1
            sampled_paths = frame_files[::step][:frames_per_video]
            
            if not sampled_paths: continue

            video_data_paths = [(p, true_label) for p in sampled_paths]
            video_dataset = DeepFakesDataset(video_data_paths, image_size=config['model']['image-size'], mode='test')
            dl = torch.utils.data.DataLoader(video_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=opt.workers, collate_fn=collate_fn_filter_none)
            
            video_preds = []
            for images, _ in dl:
                if not images.numel(): continue
                images = images.to(device)
                preds = model(images)
                preds = torch.sigmoid(preds)
                video_preds.extend(preds.cpu().numpy().flatten())
            
            avg_pred = sum(video_preds) / len(video_preds) if video_preds else 0.5
            results.append({'video': video_name, 'label': true_label, 'pred': avg_pred, 'type': fake_type})

    df = pd.DataFrame(results)
    all_fake_types = df[df['label'] == 1]['type'].unique()
    
    for fake_type in all_fake_types:
        print(f"\n--- 評估報告: Original vs {fake_type} ---")
        
        real_df = df[df['type'] == 'Original']
        fake_df = df[df['type'] == fake_type]
        
        if len(real_df) == 0 or len(fake_df) == 0:
            print(f"缺少 Original 或 {fake_type} 的樣本，無法進行評估。")
            continue
            
        min_count = min(len(real_df), len(fake_df))
        balanced_df = pd.concat([real_df.sample(min_count, random_state=42), fake_df.sample(min_count, random_state=42)])
        
        y_true = balanced_df['label']
        y_pred_proba = balanced_df['pred']
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred_proba]
        
        acc = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"樣本數 (平衡後): {len(balanced_df)}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")

    print("\n--- 最終總體性能評估 (1:1 平衡抽樣) ---")
    real_df = df[df['type'] == 'Original']
    fake_df = df[df['label'] == 1]
    
    if len(real_df) == 0 or len(fake_df) == 0:
        print("缺少 Original 或 Fake 的樣本，無法進行總體評估。")
    else:
        min_count = min(len(real_df), len(fake_df))
        balanced_df = pd.concat([real_df.sample(min_count, random_state=42), fake_df.sample(n=min_count, random_state=42)])

        y_true = balanced_df['label']
        y_pred_proba = balanced_df['pred']
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred_proba]
        
        acc = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)

        print(f"總樣本數 (平衡後): {len(balanced_df)}")
        print(f"Overall Accuracy:  {acc:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall:    {recall:.4f}")
        print(f"Overall F1-Score:  {f1:.4f}")
        print(f"Overall AUC:       {auc:.4f}")