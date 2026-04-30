import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(pool)
        )
    def forward(self, x): return self.layer(x)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        w = self.softmax(self.fc(x))
        return x * w

class CNNLSTM(nn.Module):
    def __init__(self, in_channels, conv_channels=(16,32), lstm_hidden=(64,32), dropout=0.2):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, conv_channels[0])
        self.conv2 = ConvBlock(conv_channels[0], conv_channels[1])
        self.lstm1 = nn.LSTM(conv_channels[1], lstm_hidden[0], batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden[0], lstm_hidden[1], batch_first=True)
        self.attn = Attention(lstm_hidden[1])
        self.fc = nn.Linear(lstm_hidden[1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)           # [B, C, T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)           # [B, T, C]
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.attn(x[:, -1, :])      # last hidden
        return self.fc(x)