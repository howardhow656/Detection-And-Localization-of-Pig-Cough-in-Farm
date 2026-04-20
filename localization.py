import local_package as lp
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



class GCC_CRNN(nn.Module):
    def __init__(
        self,
        n_pairs: int = 3,
        gcc_bins: int = 41,
        num_classes: int = 3,
        gru_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        # CNN 部分：把 pair 當成 channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_pairs, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),   # 只壓縮 GCC 軸
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # 推導 conv 後的 GCC 維度
        reduced_gcc_bins = gcc_bins
        for _ in range(3):
            reduced_gcc_bins = math.floor(reduced_gcc_bins / 2)

        if reduced_gcc_bins < 1:
            raise ValueError("gcc_bins 太小，經過 pooling 後變 0 了")

        self.feature_dim = 128 * reduced_gcc_bins

        self.bigru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, P, T, G)
        CNN expects (B, C, H, W)
        這裡:
           C = Pairs
           H = Time
           W = GCC bins
        """

        x = self.conv1(x)   # (B, 32, T, G/2)
        x = self.conv2(x)   # (B, 64, T, G/4)
        x = self.conv3(x)   # (B,128, T, G/8)

        # 轉成給 GRU 的格式：(B, T, feature_dim)
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, T, C, G')
        b, t, c, g = x.shape
        x = x.view(b, t, c * g)                  # (B, T, feature_dim)

        x, _ = self.bigru(x)                     # (B, T, 2*hidden)

        # clip-level 分類：時間平均池化
        x = x.mean(dim=1)                        # (B, 2*hidden)

        out = self.classifier(x)                # (B, num_classes)
        return out


#-------------------------------------------------
#將資料變成json格式
samples = {

}
#把資料讀進來，使用dataloader和GCC_Data把資料變成可用的模樣


devices = "cuda" if torch.cuda.is_available() else "cpu"

#之後資料量夠要修改成使用不同音檔來分切tain、test、vali
train_samples, temp_samples = train_test_split(
    samples,
    test_size=0.3,     # 30% 留給 val + test
    random_state=42,
    shuffle=True,
)

valid_samples, test_samples = train_test_split(
    temp_samples,
    test_size=0.5,   # 30% 的一半 → 15%
    random_state=42,
)

train_dataset = lp.GCCPhatPathDataset(
    samples=train_samples,
    sample_rate=16000,
    segment_sec=0.5,
    pre_buffer_sec=0.02,
    n_fft=512,
    hop_length=160,
    win_length=512,
    max_tau=20,
)

test_dataset = lp.GCCPhatPathDataset(
    samples=test_samples,
    sample_rate=16000,
    segment_sec=0.5,
    pre_buffer_sec=0.02,
    n_fft=512,
    hop_length=160,
    win_length=512,
    max_tau=20,
)

valid_dataset = lp.GCCPhatPathDataset(
    samples=valid_samples,
    sample_rate=16000,
    segment_sec=0.5,
    pre_buffer_sec=0.02,
    n_fft=512,
    hop_length=160,
    win_length=512,
    max_tau=20,
)


train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

gcc_model = GCC_CRNN(
    n_pairs=3,
    gcc_bins=41,
    num_classes=4,
    gru_hidden=128,
    dropout=0.3,
).to(devices)

optimizers = torch.optim.Adam(gcc_model.parameters(), lr=1e-3)

best_val = float("inf")

num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc = lp.train_one_epoch(model=gcc_model, device=devices, optimizer=optimizers, loader=train_data_loader)
    valid_loss, valid_acc = lp.valid_one_epoch(model=gcc_model, device=devices, loader=valid_data_loader)

    if valid_loss < best_val:
        best_val = valid_loss
        torch.save(gcc_model.state_dict(), "best_model.pth")