import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

base_path = os.path.join(os.path.dirname(__file__), '../..', 'output')
X_train = np.load(os.path.join(base_path, 'X_train_6axis_norm.npy'))
y_train = np.load(os.path.join(base_path, 'y_train_6axis.npy'))
X_test  = np.load(os.path.join(base_path, 'X_test_6axis_norm.npy'))
y_test  = np.load(os.path.join(base_path, 'y_test_6axis.npy'))

# 轉成 torch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

# === 模型輸入形狀 ===
timesteps, features = X_train.shape[1], X_train.shape[2]
print(f"模型輸入形狀：({timesteps}, {features})")

# === Initial bias ===
neg, pos = np.bincount(y_train.numpy().astype(int))
initial_bias = np.log(pos / neg)
print(f"Initial bias: {initial_bias:.4f} (pos={pos}, neg={neg})")

# === Class Weights ===
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train.numpy()),
    y=y_train.numpy().astype(int)
)
class_weights = torch.tensor(class_weights_array, dtype=torch.float32)
print(f"Class weights -> {dict(enumerate(class_weights_array))}")

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        modulating = (1.0 - pt) ** self.gamma
        loss = alpha_factor * modulating * BCE_loss
        return loss.mean()

USE_FOCAL = False

class CNNDense(nn.Module):
    def __init__(self, input_dim, time_steps):
        super(CNNDense, self).__init__()

        # --- CNN ---
        self.cnn = nn.Sequential(
            # Temporal CNN Block 1
            nn.Conv1d(input_dim, 32, kernel_size=15, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            # Temporal CNN Block 2
            nn.Conv1d(32, 64, kernel_size=7, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            # Temporal CNN Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
        )

        self.dropout1 = nn.Dropout(0.3)

        # --- 動態計算 fc 輸入維度 ---
        # 先模擬一個 batch=1, timestep=time_steps
        with torch.no_grad():
            dummy = torch.zeros(1, time_steps, input_dim)
            dummy = dummy.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]
            cnn_out = self.cnn(dummy)
            self.flattened_dim = cnn_out.shape[1] * cnn_out.shape[2]

        # --- FC ---
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, F] → [B, F, T]
        x = self.cnn(x)
        x = self.dropout1(x)
        x = self.fc_layers(x)
        return x.squeeze(1)

# === DataLoader 和 Validation split ===
dataset = TensorDataset(X_train, y_train)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

# === 模型與優化器 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNDense(features, timesteps).to(device)


# 使用 class weight 的 WeightedBCELoss
if USE_FOCAL:
    criterion = FocalLoss()
else:

    def weighted_bce_loss(pred, target, weights):
        loss = nn.BCELoss(reduction='none')(pred, target)
        weight_tensor = torch.where(target == 1, weights[1], weights[0])
        return (loss * weight_tensor).mean()
    
    criterion = lambda pred, target: weighted_bce_loss(pred, target, class_weights.to(device))

optimizer = optim.Adam(model.parameters())

# === EarlyStopping 模擬 ===
best_recall = 0
patience = 10
no_improve = 0
best_state = None

for epoch in range(200):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 驗證 recall
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            y_true.append(yb)
            y_pred.append(pred)

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_pred_label = (y_pred > 0.5).astype(int)
    recall = recall_score(y_true, y_pred_label)

    print(f"Epoch {epoch+1} | Recall: {recall:.4f} | Loss: {loss.item():.4f}")

    if recall > best_recall:
        best_recall = recall
        best_state = model.state_dict()
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break


if best_state:
    model.load_state_dict(best_state)

# === 測試評估 ===
model.eval()
y_proba, y_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu()
        y_proba.append(pred)
        y_true.append(yb)

y_proba = torch.cat(y_proba).numpy()
y_true = torch.cat(y_true).numpy()

print("\nThreshold Sweep:")
for t in np.arange(0.1, 0.9, 0.1):
    y_pred = (y_proba > t).astype("int32")
    print(f"Thr={t:.1f} | "
          f"Rec={recall_score(y_true, y_pred):.4f} | "
          f"Prec={precision_score(y_true, y_pred):.4f} | "
          f"F1={f1_score(y_true, y_pred):.4f}")

thr = 0.6
y_pred = (y_proba > thr).astype("int32")
print(f"\nClassification Report (Thr={thr}):")
print(classification_report(y_true, y_pred, digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

torch.save(model.state_dict(), os.path.join(base_path, 'cnn_dense.pth'))
print("模型已儲存為 cnn_dense.pth")