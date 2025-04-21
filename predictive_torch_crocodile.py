import torch
import torch.nn as nn
import numpy as np
from predictive_crocodile import load_data
import os

MODEL_PATH = "torch_bite_model.pt"
WINDOW_SIZE = 5
EPOCHS = 300
LR = 0.005
HIDDEN_SIZE = 64

# --- Prepare Dataset ---
def prepare_dataset(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# --- Model ---
class BiteNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, output_size=13):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# --- Train Model ---
def train_model(X_tensor, y_tensor, output_size):
    model = BiteNet(output_size=output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"🧪 Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    return model

# --- Load or Train ---
def get_model(X, y):
    unique_classes = sorted(set(y))
    class_to_idx = {v: i for i, v in enumerate(unique_classes)}
    idx_to_class = {i: v for v, i in class_to_idx.items()}

    y_idx = np.array([class_to_idx[v] for v in y])
    X_tensor = torch.tensor(X, dtype=torch.float32).view(len(X), WINDOW_SIZE, 1)
    y_tensor = torch.tensor(y_idx, dtype=torch.long)

    model = BiteNet(output_size=len(unique_classes))

    if os.path.exists(MODEL_PATH):
        print("📦 Loaded trained model.")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Training new model...")
        model = train_model(X_tensor, y_tensor, output_size=len(unique_classes))

    return model, idx_to_class, class_to_idx

# --- Predict ---
def predict(model, last_inputs, idx_to_class):
    model.eval()
    input_tensor = torch.tensor(last_inputs, dtype=torch.float32).view(1, WINDOW_SIZE, 1)
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).detach().numpy().flatten()
    top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
    
    print(f"\n🎯 Prediction based on {last_inputs}:")
    for idx, p in top3:
        print(f"  - {idx_to_class[idx]}: {p:.2%}")
    print(f"🏁 Top guess: {idx_to_class[top3[0][0]]}")

# --- Main Routine ---
if __name__ == "__main__":
    data = load_data()
    if len(data) < WINDOW_SIZE + 1:
        print("❗ Not enough data.")
        exit()

    X, y = prepare_dataset(data, window_size=WINDOW_SIZE)
    model, idx_to_class, class_to_idx = get_model(X, y)

    last_five = data[-WINDOW_SIZE:]
    predict(model, last_five, idx_to_class)
