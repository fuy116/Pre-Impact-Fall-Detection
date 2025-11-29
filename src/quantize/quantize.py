import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 添加模型路徑到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/model'))
from cnn_dense import CNNDense

BASE_PATH = os.path.join(os.path.dirname(__file__), '../', 'output')
ESPDL_MODEL_PATH = os.path.join(BASE_PATH, "cnn_dense.espdl")
MODEL_WEIGHTS_PATH = os.path.join(BASE_PATH, "cnn_dense.pth")
TARGET = "esp32s3"
NUM_OF_BITS = 8
DEVICE = "cpu"
BATCH_SIZE = 32


X_train = np.load(os.path.join(BASE_PATH, 'X_train_6axis_norm.npy'))
y_train = np.load(os.path.join(BASE_PATH, 'y_train_6axis.npy'))
X_test = np.load(os.path.join(BASE_PATH, 'X_test_6axis_norm.npy'))
y_test = np.load(os.path.join(BASE_PATH, 'y_test_6axis.npy'))

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

INPUT_SHAPE = [1, X_train.shape[1], X_train.shape[2]]  # 以訓練資料為準


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#取x
def collate_fn(batch):
    return batch[0].to(DEVICE)

# 載入模型結構與權重 
timesteps, features = X_train.shape[1], X_train.shape[2]
model = CNNDense(input_dim=features, time_steps=timesteps).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model.eval()

quant_ppq_graph = espdl_quantize_torch(
    model=model,
    espdl_export_file=ESPDL_MODEL_PATH,
    calib_dataloader=train_loader,   
    input_shape=INPUT_SHAPE,
    inputs=None,
    target=TARGET,
    num_of_bits=NUM_OF_BITS,
    collate_fn=collate_fn,
    dispatching_override={"quantize_input": True},
    device=DEVICE,
    error_report=True,
    skip_export=False,
    export_test_values=True,
    verbose=1,
)


y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        probs = model(xb).cpu()
        y_true.append(yb)
        y_pred.append(probs)

y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()
thr = 0.6
y_pred_label = (y_pred > thr).astype(int)

print("\n原始模型指標:")
print(f"Accuracy:  {accuracy_score(y_true, y_pred_label):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred_label):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_label):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred_label):.4f}")

executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)

y_true_q, y_pred_q = [], []
for xb, yb in test_loader:
    xb = xb.to(DEVICE)
    y_hat = executor(xb)[0].cpu()
    y_true_q.append(yb)
    y_pred_q.append(y_hat)

y_true_q = torch.cat(y_true_q).numpy()
y_pred_q = torch.cat(y_pred_q).numpy()
y_pred_label_q = (y_pred_q > thr).astype(int)

print("\n量化後模型指標:")
print(f"Accuracy:  {accuracy_score(y_true_q, y_pred_label_q):.4f}")
print(f"Recall:    {recall_score(y_true_q, y_pred_label_q):.4f}")
print(f"Precision: {precision_score(y_true_q, y_pred_label_q):.4f}")
print(f"F1 Score:  {f1_score(y_true_q, y_pred_label_q):.4f}")
print(f"\nESPDL 檔案已儲存至：{ESPDL_MODEL_PATH}")