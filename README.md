# 🐊 Predictive Crocodile — Torch Edition

Welcome to **Predictive Crocodile**, a playful machine learning project where we predict the next "bite" based on the last few moves — powered by PyTorch and Streamlit!

![Crocodile Logo](550x455.jpg)

---

## 🧠 What it does

Given a sequence of the last 5 numbers, Predictive Crocodile:
- Trains an LSTM-based neural network on past input sequences (`inputs.txt`)
- Predicts the most likely next number
- Displays the **top 3 predictions** with confidence

No database, no continuous training — just **clean fun** and **local prediction**!

---

## 🛠️ How to use

### 1. Clone this repo
```bash
git clone https://github.com/MarcelloMorettoni/predictive-crocodile.git
cd predictive-crocodile
```

### 2. Install requirements:
```bash
pip install -r requirements.txt
```

### 3. PLAY!
```bash
streamlit run torch_predictor_app.py
```

### Training:

Training is based on a simple and intuitive idea:

- `inputs.txt` contains a long sequence of numbers, each representing a **tooth position**.
- The model reads this file and creates **sliding windows** of 5 numbers to learn patterns.
- Each 5-number window is used to predict the **next** number in the sequence.
- The LSTM model is trained locally and saved as `torch_bite_model.pt`.

This approach simulates the way we might remember recent rounds to guess the next move!
