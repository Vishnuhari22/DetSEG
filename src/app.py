import streamlit as st
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Polyp Detect & Seg", page_icon="🩺", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_PATH = os.path.join(BASE_DIR, 'models', 'detection', 'weights', 'best.pt')
UNET_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'weights', 'unet_best.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    # Load YOLO
    if not os.path.exists(YOLO_PATH):
        st.error(f"❌ YOLO Model missing: {YOLO_PATH}")
        return None, None
    yolo_model = YOLO(YOLO_PATH)

    # Load U-Net
    if not os.path.exists(UNET_PATH):
        st.error(f"❌ U-Net Model missing: {UNET_PATH}")
        return None, None
        
    unet_model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
    )
    try:
        unet_model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
        unet_model.to(DEVICE)
        unet_model.eval()
    except Exception as e:
        st.error(f"Error loading U-Net: {e}")
        return yolo_model, None
    
    return yolo_model, unet_model

with st.spinner("⏳ Loading AI Models..."):
    yolo, unet = load_models()

# ==========================================
# 3. PROCESSING PIPELINE
# ==========================================
def process_image(image, conf_threshold, iou_threshold):
    # 1. Convert PIL (RGB) -> OpenCV (BGR) for YOLO
    # This is critical. YOLO expects BGR if using OpenCV logic internally.
    img_np = np.array(image)
    original_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    h, w = original_bgr.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Copy for drawing (Keep in BGR for cv2 drawing)
    draw_img_bgr = original_bgr.copy()

    # 2. YOLO Detection
    # We pass the BGR image
    results = yolo(original_bgr, conf=conf_threshold, iou=iou_threshold, verbose=True)
    
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            detections.append((x1, y1, x2, y2, conf))

    if not detections:
        # Return original RGB image if nothing found
        return img_np, "No Polyps Detected (Try lowering confidence!)"

    # 3. U-Net Segmentation
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    for (x1, y1, x2, y2, conf) in detections:
        # Padding
        pad = 15
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        # Crop from the BGR image
        crop_bgr = original_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0: continue
        
        # Convert Crop to RGB for U-Net
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Predict
        augmented = transform(image=crop_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = unet(input_tensor)
            pred = torch.sigmoid(logits) > 0.5
            pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
            
        # Resize mask back
        crop_h, crop_w = crop_bgr.shape[:2]
        pred_resized = cv2.resize(pred_np, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        
        # Add to final mask
        final_mask[y1:y2, x1:x2] = cv2.bitwise_or(final_mask[y1:y2, x1:x2], pred_resized)
        
        # Draw Box (Red in BGR is (0, 0, 255))
        cv2.rectangle(draw_img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(draw_img_bgr, f"Polyp: {conf:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 4. Create Final Visualization
    # Convert drawn BGR image back to RGB for Streamlit display
    result_rgb = cv2.cvtColor(draw_img_bgr, cv2.COLOR_BGR2RGB)
    
    # Green Overlay
    colored_mask = np.zeros_like(result_rgb)
    colored_mask[:, :, 1] = 255 # Green
    
    mask_indices = final_mask == 1
    result_rgb[mask_indices] = cv2.addWeighted(
        result_rgb[mask_indices], 0.7, colored_mask[mask_indices], 0.3, 0
    )
    
    return result_rgb, f"Found {len(detections)} Polyps"

# ==========================================
# 4. UI
# ==========================================
st.title("🩺 AI-Powered Polyp Detection & Segmentation")

st.sidebar.header("⚙️ Settings")
# Set default confidence lower (0.25) to catch more polyps
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.45, 0.05)

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Run Analysis"):
        with st.spinner("Analyzing..."):
            result_img, status = process_image(image, conf_thresh, iou_thresh)
            
            with col2:
                st.subheader("AI Result")
                st.image(result_img, use_container_width=True)
            
            if "No" in status:
                st.warning(status)
            else:
                st.success(status)