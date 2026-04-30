import yaml, torch, numpy as np
from torch.utils.data import DataLoader
from datasets.abilene import load_dataset
from models.cnn_lstm_attn import CNNLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

cfg = yaml.safe_load(open("config.yaml"))

data_args = cfg["data"].copy()
data_args.pop("batch_size", None)  # remove batch_size before calling
data = load_dataset(**data_args)

train_loader = DataLoader(data["train"], batch_size=cfg["data"]["batch_size"], shuffle=True)
val_loader = DataLoader(data["val"], batch_size=cfg["data"]["batch_size"])

device = torch.device(cfg["device"])
model = CNNLSTM(in_channels=data["train"].X.shape[2]).to(device)
optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
loss_fn = torch.nn.MSELoss()

for epoch in range(cfg["train"]["epochs"]):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "artifacts/model.pt")