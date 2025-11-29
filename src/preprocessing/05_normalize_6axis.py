import numpy as np
import os
from sklearn.preprocessing import RobustScaler
import joblib   # 用來存 scaler

#  路徑設定 
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, '../..', 'output')
OUTPUT_DIR = INPUT_DIR  # 正常儲存於 output/

#  載入 sliding window 資料 
X_train = np.load(os.path.join(INPUT_DIR, 'X_train_6axis.npy'))
y_train = np.load(os.path.join(INPUT_DIR, 'y_train_6axis.npy'))
X_test = np.load(os.path.join(INPUT_DIR, 'X_test_6axis.npy'))
y_test = np.load(os.path.join(INPUT_DIR, 'y_test_6axis.npy'))

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# 重新 reshape 為 2D (samples*timesteps, features) 做 scaler 擬合 
N_train, T, F = X_train.shape
N_test = X_test.shape[0]

X_train_2d = X_train.reshape(-1, F)
X_test_2d = X_test.reshape(-1, F)

# RobustScaler 擬合於訓練資料 
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_2d)
X_test_scaled = scaler.transform(X_test_2d)

# 還原成 3D tensor 
X_train_scaled = X_train_scaled.reshape(N_train, T, F)
X_test_scaled = X_test_scaled.reshape(N_test, T, F)

# 儲存 normalized 資料 
np.save(os.path.join(OUTPUT_DIR, 'X_train_6axis_norm.npy'), X_train_scaled)
np.save(os.path.join(OUTPUT_DIR, 'X_test_6axis_norm.npy'), X_test_scaled)

# 儲存 RobustScaler 
scaler_path = os.path.join(OUTPUT_DIR, 'robust_scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"RobustScaler 已儲存：{scaler_path}")


print("Normalization 完成！資料儲存為：")
print(f"  X_train_6axis_norm.npy, shape: {X_train_scaled.shape}")
print(f"  X_test_6axis_norm.npy,  shape: {X_test_scaled.shape}")

