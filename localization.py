import local_package as lp
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np




#-------------------------------------------------
#把資料讀進來，使用dataloader和GCC_Data把資料變成可用的模樣

train_dir = Path(r"C:\Users\howardhow\Desktop\sound detection\data\localization_data\label\training")
temp_data_dir = Path(r"C:\Users\howardhow\Desktop\sound detection\data\localization_data\label\temp")

train_samples = []
samples = []
for json_file in temp_data_dir.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        part = json.load(f)
        print(json_file, "裡面有", len(part), "筆")
        
        samples.extend(part)

for json_file in train_dir.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        part = json.load(f)
        print(json_file, "裡面有", len(part), "筆")
        train_samples.extend(part)

def is_valid_position(s):
    pos = s.get("Position")

    if pos is None:
        return False
    if pos == "" or pos == "nan":
        return False

    return True

print("Before filter:", len(samples), len(train_samples))

samples = [s for s in samples if is_valid_position(s)]
train_samples = [s for s in train_samples if is_valid_position(s)]

print("After filter:", len(samples), len(train_samples))
#------------------------------------------------------------------------------------
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("Device:", devices)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

#之後資料量夠要修改成使用不同音檔來分切tain、test、vali

batch_size = 16


valid_samples, test_samples = train_test_split(
    samples,
    test_size=0.5,   # 30% 的一半 → 15%
    random_state=42,
)

MAX_TAU = 512
GCC_BINS = 2 * MAX_TAU + 1
n_fft = 2048
win_length = 1024
hop_length = 512
segment_sec = 0.5
pre_buffer = 0.08
sample_rates = 8000

train_dataset = lp.GCCPhatPathDataset(
    samples=train_samples,
    sample_rate=sample_rates,
    segment_sec=segment_sec,
    pre_buffer_sec=pre_buffer,
    n_fft = n_fft,
    win_length = win_length,
    hop_length = hop_length,
    max_tau = MAX_TAU
)

test_dataset = lp.GCCPhatPathDataset(
    samples=test_samples,
    sample_rate=sample_rates,
    segment_sec=segment_sec,
    pre_buffer_sec=pre_buffer,
    n_fft = n_fft,
    win_length = win_length,
    hop_length = hop_length,
    max_tau = MAX_TAU
)

valid_dataset = lp.GCCPhatPathDataset(
    samples=valid_samples,
    sample_rate=sample_rates,
    segment_sec=segment_sec,
    pre_buffer_sec=pre_buffer,
    n_fft = n_fft,
    win_length = win_length,
    hop_length = hop_length,
    max_tau = MAX_TAU
)

# lp.precompute_gcc_features(train_dataset, "data/localization_data/cache_gcc/train")
# lp.precompute_gcc_features(valid_dataset, "data/localization_data/cache_gcc/valid")
# lp.precompute_gcc_features(test_dataset, "data/localization_data/cache_gcc/test")

train_dataset_cached = lp.CachedGCCDataset("data/localization_data/cache_gcc/train")
valid_dataset_cached = lp.CachedGCCDataset("data/localization_data/cache_gcc/valid")
test_dataset_cached = lp.CachedGCCDataset("data/localization_data/cache_gcc/test")



train_data_loader = DataLoader(
    train_dataset_cached,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

test_data_loader = DataLoader(
    test_dataset_cached,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

valid_data_loader = DataLoader(
    valid_dataset_cached,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

gcc_model = lp.GCC_CRNN(
    n_pairs=3,
    gcc_bins=GCC_BINS,
    num_classes=3,
    gru_hidden=32,
    dropout=0.5
).to(devices)
print(next(gcc_model.parameters()).device)

optimizers = torch.optim.Adam(gcc_model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val = float("inf")

num_epochs = 25
history = {
    "train_loss": [],
    "train_acc": [],
    "valid_loss": [],
    "valid_acc": [],
}


for epoch in range(num_epochs):
    print(f'第{epoch}輪開始')
    train_loss, train_acc = lp.train_one_epoch(model=gcc_model, device=devices, optimizer=optimizers, loader=train_data_loader)
    print(f'Training完')
    valid_loss, valid_acc = lp.valid_one_epoch(model=gcc_model, device=devices, loader=valid_data_loader)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)

    print(f'第{epoch}輪:')
    print(f'Train_loss:{train_loss}, Train_acc:{train_acc}')
    print(f'Validation_loss:{valid_loss},Validation_acc:{valid_acc}')
    if valid_loss < best_val:
        best_val = valid_loss
        torch.save(gcc_model.state_dict(), "best_model.pth")

epochs = range(1, num_epochs + 1)
# loss 曲線
plt.figure()
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["valid_loss"], label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300)
plt.show()

# accuracy 曲線
plt.figure()
plt.plot(epochs, history["train_acc"], label="Train Accuracy")
plt.plot(epochs, history["valid_acc"], label="Valid Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

best_model = lp.GCC_CRNN(
    n_pairs=3,
    gcc_bins=GCC_BINS,
    num_classes=3,
    gru_hidden=32,
    dropout=0.5
).to(devices)

best_model.load_state_dict(torch.load("best_model.pth", map_location=devices))

best_model.eval()

test_loss, test_acc = lp.valid_one_epoch(
    model=best_model,
    device=devices,
    loader=test_data_loader
)

print("===== FINAL TEST RESULT =====")
print(f"Test_loss: {test_loss}")
print(f"Test_acc: {test_acc}")

all_preds = []
all_labels = []

best_model.eval()
with torch.no_grad():
    for batch in test_data_loader:
        x = batch["x"].to(devices)
        # pair_mask = batch["pair_mask"].to(devices)
        y = batch["y"].to(devices)

        logits = best_model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(all_labels, all_preds))


class_names = ["欄位1", "欄位2", "欄位3"]

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

plt.xticks(np.arange(len(class_names)), class_names)
plt.yticks(np.arange(len(class_names)), class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


per_class_acc = np.divide(
    cm.diagonal(),
    cm.sum(axis=1),
    out=np.zeros_like(cm.diagonal(), dtype=float),
    where=cm.sum(axis=1) != 0
)

for i, acc in enumerate(per_class_acc):
    print(f"{class_names[i]} accuracy: {acc:.4f}")