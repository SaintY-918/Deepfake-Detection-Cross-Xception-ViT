
import json
import os
import shutil
import argparse

def organize_files(source_dir, fake_type):
    """
    根據指定的來源資料夾和偽造類型，自動整理檔案。
    """
    print(f"\n{'='*20}")
    print(f"--- 開始處理偽造類型: {fake_type} ---")
    print(f"--- 從來源資料夾: {source_dir} ---")
    print(f"{'='*20}")

    if not os.path.exists(source_dir):
        print(f"錯誤：找不到來源資料夾 '{source_dir}'。請確認路徑是否正確。")
        return

    # 定義分割檔案和目標路徑
    splits_dir = "splits"
    output_dir = "prepared_dataset"
    split_mapping = {
        "train.json": "training_set",
        "val.json": "validation_set",
        "test.json": "test_set"
    }

    # 遍歷 train, val, test
    for split_file, split_name in split_mapping.items():
        split_path = os.path.join(splits_dir, split_file)
        if not os.path.exists(split_path):
            print(f"警告：找不到分割檔案 {split_path}，跳過 '{split_name}'。")
            continue

        # 建立最終的目標資料夾，例如: prepared_dataset/training_set/Face2Face/
        dest_dir = os.path.join(output_dir, split_name, fake_type)
        os.makedirs(dest_dir, exist_ok=True)

        with open(split_path, 'r') as f:
            split_data = json.load(f)

        moved_count = 0
        for pair in split_data:
            # 偽造影片的命名規則是 id1_id2 或 id2_id1
            for fake_name_template in [f"{pair[0]}_{pair[1]}", f"{pair[1]}_{pair[0]}"]:
                source_folder_path = os.path.join(source_dir, fake_name_template)
                
                # 如果在來源資料夾中找到了對應的影片資料夾，就移動它
                if os.path.exists(source_folder_path):
                    destination_path = os.path.join(dest_dir, fake_name_template)
                    shutil.move(source_folder_path, destination_path)
                    moved_count += 1
        
        print(f"在 '{split_name}' 中，成功移動了 {moved_count} 個 '{fake_type}' 資料夾到 {dest_dir}")

# --- 主程式區塊：解析指令行參數 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動化整理 DeepFake 資料集")
    parser.add_argument('--source_dir', type=str, required=True, help="包含已切割臉部圖片的來源資料夾 (例如: face_crops_face2face)")
    parser.add_argument('--fake_type', type=str, required=True, help="偽造的類型名稱 (例如: Face2Face)，這將作為目標資料夾的名稱")
    
    args = parser.parse_args()
    
    organize_files(args.source_dir, args.fake_type)

    print("\n--- 所有指定的整理任務已完成！ ---")