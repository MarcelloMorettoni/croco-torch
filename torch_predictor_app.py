import streamlit as st
import torch
import numpy as np
from predictive_torch_crocodile import BiteNet, load_data, prepare_dataset, get_model, WINDOW_SIZE, MODEL_PATH

# --- Load model and mappings ---
@st.cache_resource
def load_trained_model():
    data = load_data()
    X, y = prepare_dataset(data, window_size=WINDOW_SIZE)
    model, idx_to_class, class_to_idx = get_model(X, y)
    return model, idx_to_class, class_to_idx

model, idx_to_class, class_to_idx = load_trained_model()

# --- UI ---
st.title("🐊 Predictive Crocodile - Torch Edition")
st.subheader("Enter the last 5 numbers to predict the next one!")

inputs = []
cols = st.columns(WINDOW_SIZE)
for i in range(WINDOW_SIZE):
    with cols[i]:
        val = st.number_input(f"Number {i+1}", min_value=0, max_value=20, step=1, key=f"input_{i}")
        inputs.append(val)

# --- Prediction ---
if st.button("🔮 Predict"):
    input_tensor = torch.tensor(inputs, dtype=torch.float32).view(1, WINDOW_SIZE, 1)
    with torch.no_grad():
        model.eval()
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy().flatten()
        top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]

    st.success(f"🎯 Predicted next number: **{idx_to_class[top3[0][0]]}**")
    st.write("📊 Top 3 predictions with confidence:")
    for idx, p in top3:
        st.markdown(f"- {idx_to_class[idx]}: **{p:.2%}**")
