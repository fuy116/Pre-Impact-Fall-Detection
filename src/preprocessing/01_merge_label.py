import pandas as pd
import os


BASE_DIR = os.path.dirname(__file__)
sensor_data_root = os.path.join(BASE_DIR, '../../data/KFall/Sensor_Data')
label_data_root = os.path.join(BASE_DIR, '../../data/KFall/Label_Data')
output_path = os.path.join(BASE_DIR, '../..', 'output', 'All_Sensor_Data.csv')

# 若輸出檔案已存在則先刪除（避免重複附加）
if os.path.exists(output_path):
    os.remove(output_path)

header_written = False  # 控制是否寫入 header

# 處理每位受測者 
for sid in range(6, 39):  # SA06 ~ SA38
    subject_folder = f'SA{sid:02d}'
    subject_path = os.path.join(sensor_data_root, subject_folder)
    label_file = os.path.join(label_data_root, f'{subject_folder}_Label.xlsx')

    if not os.path.exists(subject_path):
        print(f' 缺少 Sensor 資料夾：{subject_path}')
        continue
    if not os.path.exists(label_file):
        print(f'缺少 Label 檔案：{label_file}')
        continue

    # === 讀取 Label 檔並補齊合併儲存格 ===
    label_df = pd.read_excel(label_file)
    label_df['Task Code (Task ID)'] = label_df['Task Code (Task ID)'].fillna(method='ffill')
    label_df['Task_ID'] = label_df['Task Code (Task ID)'].astype(str).str.extract(r'\((\d+)\)').astype(int)

    for file in sorted(os.listdir(subject_path)):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(subject_path, file)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f'無法讀取 {file_path} -> {e}')
            continue

        if df.empty:
            print(f'空檔案：{file_path}')
            continue

        if 'FrameCounter' not in df.columns:
            print(f'檔案 {file} 缺少 FrameCounter 欄位，跳過')
            continue

        df['FrameCounter'] = pd.to_numeric(df['FrameCounter'], errors='coerce')
        df['Label'] = 0
        df['RecordName'] = os.path.splitext(file)[0]

        # 解析 Task ID 與 Trial ID
        try:
            base = os.path.splitext(file)[0]  # e.g., S06T20R02
            task_id = int(base[4:6])
            trial_id = int(base[7:9])
        except:
            print(f'無法解析檔名格式：{file}')
            continue

        matched = label_df[
            (label_df['Task_ID'] == task_id) &
            (label_df['Trial ID'] == trial_id)
        ]

        for _, row in matched.iterrows():
            onset = int(row['Fall_onset_frame'])
            impact = int(row['Fall_impact_frame'])
            df.loc[
                (df['FrameCounter'] >= onset) & (df['FrameCounter'] <= impact),
                'Label'
            ] = 1

        # 寫入 CSV，第一筆寫 header，之後 append
        df.to_csv(output_path, mode='a', index=False, header=not header_written)
        header_written = True

        print(f'已處理：{file_path} -> 共 {len(df)} 筆')

print(f'\n合併與跌倒標記完成！儲存於：{output_path}')
