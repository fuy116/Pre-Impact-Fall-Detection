import pandas as pd
import numpy as np
import os
from scipy.linalg import inv

class KalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.n_states = 6  # 六軸資料
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states)
        self.F = np.eye(self.n_states)
        self.H = np.eye(self.n_states)
        self.Q = np.eye(self.n_states) * process_noise
        self.R = np.eye(self.n_states) * measurement_noise
        self.is_initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P

    def filter_point(self, measurement):
        if not self.is_initialized:
            self.x = measurement.copy()
            self.is_initialized = True
            return self.x.copy()
        self.predict()
        self.update(measurement)
        return self.x.copy()

def apply_kalman_filter_to_group(group, process_noise=0.01, measurement_noise=0.1):
    kf = KalmanFilter(process_noise, measurement_noise)
    sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    sensor_data = group[sensor_cols].values
    filtered_data = [kf.filter_point(measurement) for measurement in sensor_data]
    filtered_df = group.copy()
    filtered_df[sensor_cols] = np.array(filtered_data)
    return filtered_df

def main():
    BASE_DIR = os.path.dirname(__file__)
    input_csv = os.path.join(BASE_DIR, '../..', 'output', 'All_Sensor_Data.csv')
    output_csv = os.path.join(BASE_DIR, '../..', 'output', 'All_Sensor_Data_Kalman.csv')

    print("開始卡爾曼濾波處理（6軸）...")
    df = pd.read_csv(input_csv)
    print(f"原始資料: {len(df)} 筆記錄")

    required_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'RecordName']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"錯誤：缺少必要欄位: {missing}")
        return

    PROCESS_NOISE = 0.01
    MEASUREMENT_NOISE = 0.1
    print(f"濾波參數: 過程噪音={PROCESS_NOISE}, 測量噪音={MEASUREMENT_NOISE}")

    filtered_groups = []
    total_groups = df['RecordName'].nunique()

    for i, (name, group) in enumerate(df.groupby('RecordName'), 1):
        filtered = apply_kalman_filter_to_group(group, PROCESS_NOISE, MEASUREMENT_NOISE)
        filtered_groups.append(filtered)
        if i % 10 == 0 or i == total_groups:
            print(f"進度: {i}/{total_groups} ({i/total_groups*100:.1f}%)")

    print("合併濾波結果")
    filtered_df = pd.concat(filtered_groups, ignore_index=True)

    print("儲存濾波後的資料")
    filtered_df.to_csv(output_csv, index=False)

    print("\n濾波前後比較（加速度）:")
    print(df[['AccX', 'AccY', 'AccZ']].describe())
    print("\n濾波後（加速度）:")
    print(filtered_df[['AccX', 'AccY', 'AccZ']].describe())

    print("\n濾波前後比較（陀螺儀）:")
    print(df[['GyrX', 'GyrY', 'GyrZ']].describe())
    print("\n濾波後（陀螺儀）:")
    print(filtered_df[['GyrX', 'GyrY', 'GyrZ']].describe())

    print(f"\n卡爾曼濾波完成！")
    print(f"輸出檔案: {output_csv}")
    print(f"處理筆數: {len(filtered_df)} 筆")

if __name__ == "__main__":
    main()

