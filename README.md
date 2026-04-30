# Network Traffic Prediction using Deep Learning

This project implements a deep learning-based approach for predicting network traffic using a hybrid CNN-LSTM model with an attention mechanism. It is designed to learn temporal and spatial patterns from network datasets such as the Abilene dataset.

---

## Project Structure

```
traffic_prediction_project/
│
├── artifacts/
│   └── model.pt
│
├── data/
│
├── outputs/
│   └── cnn_lstm_metrics.json
│
├── src/
│   ├── datasets/
│   │   └── abilene.py
│   │
│   ├── models/
│   │   └── cnn_lstm_attn.py
│   │
│   ├── utils/
│   │   └── metrics.py
│   │
│   ├── train.py
│   └── predict.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Overview

The system predicts future network traffic based on historical data. It combines convolutional layers for feature extraction, LSTM layers for sequence modeling, and an attention mechanism to improve prediction accuracy.

---

## Features

* Deep learning-based traffic prediction
* CNN + LSTM + Attention architecture
* Configurable training via YAML file
* Custom evaluation metrics (MSE, MAE)
* Model saving and loading
* Visualization of predicted vs actual values
* Inference time measurement

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* YAML
* Scikit-learn

---

## Setup Instructions

### 1. Clone the repository

```
git clone <repository-url>
cd traffic_prediction_project
```

### 2. Create and activate virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Project

### Train the model

Run from the project root directory:

```
python src/train.py
```

This will train the model and save it to:

```
artifacts/model.pt
```

---

### Run prediction and evaluation

```
python src/predict.py
```

This will:

* Load the trained model
* Evaluate it on test data
* Print performance metrics (MSE, MAE)
* Display prediction plots

---

## Configuration

All configurable parameters are stored in:

```
config.yaml
```

This includes:

* Dataset parameters
* Batch size
* Learning rate
* Number of epochs
* Device (CPU/GPU)

---

## Sample Output

```
Model Evaluation Metrics:
-------------------------
MSE: 0.0023
MAE: 0.0345
Inference Time: 0.5123 seconds
-------------------------
```

---

## Model Architecture

* CNN layers extract spatial features from input data
* LSTM layers capture temporal dependencies
* Attention mechanism highlights important time steps

---

## Notes

* Always run scripts from the project root directory
* Ensure `config.yaml` is present in the root folder
* The trained model will be saved automatically after training

---

## Future Work

* Real-time traffic prediction system
* Web-based dashboard for visualization
* Hyperparameter tuning and optimization
* Deployment as an API or application

---

## Author

Shravani Itkar

---

## License

This project is intended for academic use.
