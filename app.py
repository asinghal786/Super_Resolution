import streamlit as st
from streamlit_image_comparison import image_comparison
import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Page configuration
st.set_page_config(page_title="Super-Resolution Image Comparison", layout="centered")

# Function to perform super-resolution using RRDBNet
def super_resolve_image(input_image_path, model, device):
    # Read the input image
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Run the model for super-resolution
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Convert the output to a format suitable for saving with OpenCV
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert back to HWC format
    output = (output * 255.0).round().astype(np.uint8)  # Convert to 8-bit image
    
    return output

# Sidebar - Load the model
st.sidebar.title("Model Configuration")
model_path = st.sidebar.text_input("Model Path", value="models/RRDB_ESRGAN_x4.pth")
device_option = st.sidebar.radio("Device", ("cuda", "cpu"))
device = torch.device(device_option)

# Load RRDBNet model
if os.path.exists(model_path):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    st.sidebar.success("Model Loaded Successfully")
else:
    st.sidebar.error("Model Path Invalid. Please check and reload.")

# Main app - Upload low-resolution image
uploaded_file = st.file_uploader("Upload a Low-Resolution Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image
    input_path = f"LR/{uploaded_file.name}"
    os.makedirs("LR", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Perform super-resolution
    st.markdown("### Super-Resolution in Progress...")
    output_image = super_resolve_image(input_path, model, device)
    
    # Save the output image
    output_path = f"results/{os.path.splitext(uploaded_file.name)[0]}_SR.png"
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(output_path, output_image)

    st.success(f"Super-Resolution Completed. Saved to: {output_path}")

    # Display image comparison
    st.markdown("### Image Comparison")
    image_comparison(
        img1=input_path,
        img2=output_path,
        label1="Low-Resolution",
        label2="Super-Resolved"
    )

    # Provide download link for the output image
    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Super-Resolved Image",
            data=f,
            file_name=os.path.basename(output_path),
            mime="image/png"
        )
else:
    st.warning("Please upload an image to start!")
