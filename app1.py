import streamlit as st
from streamlit_image_comparison import image_comparison
import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# Page configuration
st.set_page_config(page_title="Super-Resolution Image Comparison", layout="centered")

if 'show_metrics' not in st.session_state:
    st.session_state.show_metrics = False

def calculate_metrics(upscaled_img, hr_img):
    """Calculates Accuracy, F1 Score, MCC, GM1, and GM2."""
    upscaled_gray = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2GRAY)
    hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
    
    upscaled_flat = upscaled_gray.flatten()
    hr_flat = hr_gray.flatten()
    
    diff = np.abs(upscaled_flat - hr_flat)
    threshold = 10
    predictions = (diff < threshold).astype(int)
    actuals = np.ones_like(predictions)
    
    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    mcc = matthews_corrcoef(actuals, predictions)
    tpr = np.sum(predictions) / len(predictions)
    tnr = 1 - tpr
    gm1 = np.sqrt(tpr * tnr)
    precision = np.sum(predictions) / np.sum(actuals)
    gm2 = np.sqrt(precision * tnr)
    
    return accuracy, f1, mcc, gm1, gm2

# Function to perform super-resolution using RRDBNet
def super_resolve_image(input_image_path, model, device):
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    return output

# Sidebar - Load the model
st.sidebar.title("Model Configuration")
model_files = [f for f in os.listdir("G:/Super_Resolution/models") if f.endswith(".pth")]
model_path = st.sidebar.selectbox("Select Model", model_files, index=0)
device_option = st.sidebar.radio("Device", ("cuda", "cpu"))
device = torch.device(device_option)

model_full_path = os.path.join("G:/Super_Resolution/models", model_path)
if os.path.exists(model_full_path):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_full_path), strict=True)
    model.eval()
    model = model.to(device)
    st.sidebar.success("Model Loaded Successfully")
else:
    st.sidebar.error("Model Path Invalid. Please check and reload.")

uploaded_file = st.file_uploader("Upload a Low-Resolution Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    input_path = f"LR/{uploaded_file.name}"
    os.makedirs("LR", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.markdown("### Super-Resolution in Progress...")
    output_image = super_resolve_image(input_path, model, device)
    output_path = f"results/{os.path.splitext(uploaded_file.name)[0]}_SR.png"
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(output_path, output_image)
    st.success(f"Super-Resolution Completed. Saved to: {output_path}")
    
    st.markdown("### Image Comparison")
    image_comparison(img1=input_path, img2=output_path, label1="Low-Resolution", label2="Super-Resolved")
    
    with open(output_path, "rb") as f:
        st.download_button(label="Download Super-Resolved Image", data=f, file_name=os.path.basename(output_path), mime="image/png")
    
    if st.button("More", key="more_button", help="Click to upload Ground Truth Image for Evaluation", use_container_width=True):
        st.session_state.show_metrics = True

    if st.session_state.show_metrics:
        hr_uploaded = st.file_uploader("Upload Ground Truth High-Resolution Image", type=["png", "jpg", "jpeg"], key="hr_upload")
        if hr_uploaded is not None:
            hr_path = f"HR/{hr_uploaded.name}"
            os.makedirs("HR", exist_ok=True)
            with open(hr_path, "wb") as f:
                f.write(hr_uploaded.getbuffer())
            
            hr_image = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            output_image = cv2.imread(output_path, cv2.IMREAD_COLOR)
            accuracy, f1, mcc, gm1, gm2 = calculate_metrics(output_image, hr_image)
            
            st.markdown("### Evaluation Metrics")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**MCC:** {mcc:.4f}")
            st.write(f"**GM1:** {gm1:.4f}")
            st.write(f"**GM2:** {gm2:.4f}")
else:
    st.warning("Please upload an image to start!")