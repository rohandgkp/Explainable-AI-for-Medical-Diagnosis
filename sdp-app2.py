import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# Set custom page configuration
st.set_page_config(page_title="TumorNet-CNN", page_icon="🧠", layout="centered")

# Apply CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
        }
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #0a58ca;
            text-align: center;
        }
        .sub-title {
            font-size: 18px;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        .prediction-box {
            background-color: #e6f7ff;
            border-left: 5px solid #1890ff;
            padding: 16px;
            border-radius: 10px;
            font-size: 22px;
            font-weight: 600;
            color: #005580;
            text-align: center;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Titles
st.markdown('<div class="main-title">🧠 TumorNet-CNN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Detect brain tumor types using a custom-trained CNN model.</div>', unsafe_allow_html=True)

# Load the saved model
model = tf.keras.models.load_model("fine_tuned_model3.keras")

# Define class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    img_size = (224, 224)
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Display prediction
    st.markdown(f'<div class="prediction-box">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)
    st.write(f"🧪 **Confidence:** {confidence:.2f}%")

    # Show progress bar
    st.progress(int(confidence))

    # Show probabilities bar chart
    prob_df = pd.DataFrame({
        "Tumor Type": class_names,
        "Probability": prediction
    }).sort_values(by="Probability", ascending=False)

    st.write("📊 **Prediction Probabilities**")
    st.bar_chart(prob_df.set_index("Tumor Type"))
    
import matplotlib.pyplot as plt
import io
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing import image as keras_image

# Define size
IMG_SIZE = (224, 224)

# Function 1: Occlusion Sensitivity
def occlusion_sensitivity_streamlit(image, model, patch_size=20):
    img_array = np.array(image.resize(IMG_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    occlusion_map = np.zeros_like(img_array[0, :, :, 0])

    for i in range(0, IMG_SIZE[0], patch_size):
        for j in range(0, IMG_SIZE[1], patch_size):
            occluded_img = img_array.copy()
            occluded_img[0, i:i+patch_size, j:j+patch_size, :] = 0
            prediction = model.predict(occluded_img, verbose=0)
            class_idx = np.argmax(prediction[0])
            occlusion_map[i:i+patch_size, j:j+patch_size] = prediction[0][class_idx]

    fig, ax = plt.subplots()
    ax.imshow(image.resize(IMG_SIZE))
    ax.imshow(occlusion_map, cmap='jet', alpha=0.4)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    st.image(buf, caption="🔍 Occlusion Sensitivity", use_column_width=True)
    plt.close()


# Function 2: LIME Explanation
def explain_lime_streamlit(image, model):
    img_array = np.array(image.resize(IMG_SIZE)) / 255.0
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype('double'),
        model.predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=True
    )
    #lime_img = mark_boundaries(temp / 255.0, mask)

    fig, ax = plt.subplots()
    #ax.imshow(lime_img)
    ax.imshow(temp)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    st.image(buf, caption="🟢 LIME Explanation", use_column_width=True)
    plt.close()


# Function 3: Smooth Gradients
def smoothgrad_streamlit(model, image, num_samples=50, noise_level=0.1):
    img_array = np.array(image.resize(IMG_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    smooth_grad = np.zeros_like(img_array)
    for _ in range(num_samples):
        noise = np.random.normal(scale=noise_level, size=img_array.shape)
        noisy_img = img_array + noise

        with tf.GradientTape() as tape:
            noisy_img = tf.convert_to_tensor(noisy_img, dtype=tf.float32)
            tape.watch(noisy_img)
            predictions = model(noisy_img, training=False)
            class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, noisy_img)
        smooth_grad += grads.numpy()

    smooth_grad /= num_samples
    smooth_grad = np.squeeze(smooth_grad)
    smooth_grad = (smooth_grad - smooth_grad.min()) / (smooth_grad.max() - smooth_grad.min())

    fig, ax = plt.subplots()
    ax.imshow(smooth_grad, cmap="seismic") # YlGnBu, Blues_r, seismic
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    st.image(buf, caption="🌈 Smooth Gradients", use_column_width=True)
    plt.close()


# Function 4: Saliency Map
def compute_saliency_streamlit(model, image):
    img_array = np.array(image.resize(IMG_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    gradients = tape.gradient(loss, img_tensor)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]

    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))

    fig, ax = plt.subplots()
    ax.imshow(saliency, cmap="seismic") # YlGnBu, Blues_r, seismic
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    st.image(buf, caption="🔬 Saliency Map", use_column_width=True)
    plt.close()


# =========================
# XAI Technique Buttons
# =========================
st.markdown("## 🔎 Model Explainability (XAI)")

xai_option = st.radio("Choose an Explainability Method:", 
                      ["Occlusion Sensitivity", "LIME", "Smooth Gradients", "Saliency Map"])

if st.button("Generate Visualization"):
    with st.spinner(f"Generating {xai_option}..."):
        if xai_option == "Occlusion Sensitivity":
            occlusion_sensitivity_streamlit(image, model)
        elif xai_option == "LIME":
            explain_lime_streamlit(image, model)
        elif xai_option == "Smooth Gradients":
            smoothgrad_streamlit(model, image)
        elif xai_option == "Saliency Map":
            compute_saliency_streamlit(model, image)


# Footer
st.markdown("---")
st.markdown("🧬 Powered by Streamlit and TensorFlow | Created by 'Team-20241131'", unsafe_allow_html=True)
