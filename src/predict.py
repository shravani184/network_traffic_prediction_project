import torch, yaml, numpy as np, time
from datasets.abilene import load_dataset
from models.cnn_lstm_attn import CNNLSTM
import matplotlib.pyplot as plt
from utils.metrics import mse, mae  # ✅ Import your custom metrics here

# 1️⃣ Load configuration
cfg = yaml.safe_load(open("config.yaml"))
data_args = cfg["data"].copy()
data_args.pop("batch_size", None)
data = load_dataset(**data_args)

# 2️⃣ Prepare test loader
test_loader = torch.utils.data.DataLoader(data["test"], batch_size=128)

# 3️⃣ Load trained model
device = torch.device(cfg["device"])
model = CNNLSTM(in_channels=data["train"].X.shape[2]).to(device)
model.load_state_dict(torch.load("artifacts/model.pt"))
model.eval()

# 4️⃣ Predict and measure inference time
y_true, y_pred = [], []

start_time = time.time()  # ⏱️ Start measuring inference time
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)
end_time = time.time()  # ⏱️ End measuring inference time

inference_time = end_time - start_time  # Total time in seconds

# 5️⃣ Prepare results
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# 6️⃣ Calculate Metrics using your custom functions
mse_val = mse(y_true, y_pred)
mae_val = mae(y_true, y_pred)

# ✅ Print metrics
print("\n📊 Model Evaluation Metrics:")
print("---------------------------------")
print(f"MSE: {mse_val:.4f}")
print(f"MAE: {mae_val:.4f}")
print(f"Inference Time: {inference_time:.4f} seconds")
print("---------------------------------\n")

# 7️⃣ (Optional) Plot True vs Predicted values
plt.plot(y_true, label="True")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("True vs Predicted Network Traffic")
plt.xlabel("Time Steps")
plt.ylabel("Traffic Volume (Mbps)")
plt.show()
