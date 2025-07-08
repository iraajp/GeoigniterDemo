import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


st.set_page_config(layout="wide")
st.title("🔥 GeoIgniter – Forest Fire Risk Prediction Demo")

st.markdown("This demo predicts forest fire risk using a trained model on terrain (slope) and vegetation (LULC) features.")

# --- Load Model ---
model = load_model("models/fire_model.h5")

# --- Load Data ---
X = np.load("data/X.npy")  # shape: (1024, 1024, 2)
H, W, C = X.shape
X_flat = X.reshape(-1, C)

@st.cache_resource
def predict_fire(X_flat):
    y_pred = model.predict(X_flat, verbose=0)
    fire_prob = y_pred.reshape(H, W)
    fire_map = (fire_prob > 0.5).astype(np.uint8)
    return fire_map, fire_prob

# --- Run Prediction ---
with st.spinner("Running model prediction..."):
    fire_mask, fire_scores = predict_fire(X_flat)

# --- Visualize Results ---
st.subheader(" Fire Risk Map")
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(fire_scores, cmap="hot", interpolation="nearest")
ax.set_title("Predicted Fire Probability (0 to 1)")
ax.axis("off")
st.pyplot(fig)

