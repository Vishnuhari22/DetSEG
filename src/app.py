import streamlit as st
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Polyp Detect & Seg", page_icon="🩺", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_PATH = os.path.join(BASE_DIR, 'models', 'detection', 'weights', 'best.pt')
UNET_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'weights', 'unet_best.pth')
YOLO_SMALL_POLYP_PATH = os.path.join(BASE_DIR, 'models', 'detection', 'weights', 'best_small_polyp.pt')
UNET_UPGRADED_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'weights', 'unet_upgraded.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    # Load standard YOLO
    if not os.path.exists(YOLO_PATH):
        st.error(f"❌ YOLO Model missing: {YOLO_PATH}")
        return None, None, None, None, None
    yolo_model = YOLO(YOLO_PATH)

    # Load standard U-Net (ResNet-34)
    if not os.path.exists(UNET_PATH):
        st.error(f"❌ U-Net Model missing: {UNET_PATH}")
        return None, None, None, None, None
    unet_model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
    )
    try:
        unet_model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
        unet_model.to(DEVICE).eval()
    except Exception as e:
        st.error(f"Error loading U-Net: {e}")
        return yolo_model, None, None, None, None

    # Plan 2: Small-polyp YOLO for SAHI deep scan
    sahi_model = None
    if os.path.exists(YOLO_SMALL_POLYP_PATH):
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=YOLO_SMALL_POLYP_PATH,
            confidence_threshold=0.20, device=DEVICE
        )
    else:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=YOLO_PATH,
            confidence_threshold=0.25, device=DEVICE
        )

    # Plan 3: Upgraded U-Net (EfficientNet-B4)
    unet_upgraded = None
    if os.path.exists(UNET_UPGRADED_PATH):
        unet_upgraded = smp.Unet(
            encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1
        )
        try:
            unet_upgraded.load_state_dict(torch.load(UNET_UPGRADED_PATH, map_location=DEVICE))
            unet_upgraded.to(DEVICE).eval()
        except Exception as e:
            st.warning(f"Plan 3 model failed to load: {e}")
            unet_upgraded = None

    return yolo_model, unet_model, sahi_model, unet_upgraded

with st.spinner("⏳ Loading AI Models..."):
    yolo, unet, sahi_det_model, unet_upg = load_models()

# ==========================================
# 3. PROCESSING PIPELINE
# ==========================================
def process_image(image, conf_threshold, iou_threshold, deep_scan=False):
    # 1. Convert PIL (RGB) -> OpenCV (BGR) for YOLO
    img_np = np.array(image)
    original_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    h, w = original_bgr.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Select segmentation model
    active_seg = unet_upg if (deep_scan and unet_upg) else unet
    
    # Copy for drawing (Keep in BGR for cv2 drawing)
    draw_img_bgr = original_bgr.copy()

    # 2. YOLO Detection (standard or SAHI deep scan)
    if deep_scan and sahi_det_model:
        # SAHI sliced inference with Plan 2 model
        effective_conf = max(0.15, conf_threshold - 0.05)
        sahi_det_model.confidence_threshold = effective_conf
        img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        min_dim = min(h, w)
        if min_dim < 640:
            slice_size, overlap = 256, 0.35
        elif min_dim < 1280:
            slice_size, overlap = 384, 0.3
        else:
            slice_size, overlap = 512, 0.3
        result = get_sliced_prediction(
            image=img_rgb, detection_model=sahi_det_model,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True, postprocess_type="NMS",
            postprocess_match_threshold=iou_threshold * 0.9, verbose=0
        )
        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            detections.append((int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy), float(pred.score.value)))
    else:
        results = yolo(original_bgr, conf=conf_threshold, iou=iou_threshold, verbose=True)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                detections.append((x1, y1, x2, y2, conf))

    if not detections:
        # Return original RGB image if nothing found
        mode_label = "🔬 Deep Scan" if deep_scan else "Standard"
        return img_np, f"No Polyps Detected ({mode_label} — Try lowering confidence!)"

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
            logits = active_seg(input_tensor)
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
    
    mode_label = "🔬 Deep Scan" if deep_scan else "Standard"
    return result_rgb, f"Found {len(detections)} Polyps ({mode_label})"

# ==========================================
# 4. UI
# ==========================================
st.title("🩺 AI-Powered Polyp Detection & Segmentation")

st.sidebar.header("⚙️ Settings")
# Set default confidence lower (0.25) to catch more polyps
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.45, 0.05)
deep_scan_enabled = st.sidebar.checkbox(
    "🔬 Deep Scan",
    value=False,
    help="Enhanced detection using Plan 2 small-polyp YOLO (SAHI) + Plan 3 U-Net (EfficientNet-B4). Slower but catches small/flat polyps."
)

if deep_scan_enabled:
    st.sidebar.info("🔬 Deep Scan: Plan 2 YOLO + Plan 3 U-Net active")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Run Analysis"):
        with st.spinner("Analyzing..."):
            result_img, status = process_image(image, conf_thresh, iou_thresh, deep_scan=deep_scan_enabled)
            
            with col2:
                st.subheader("AI Result")
                st.image(result_img, use_container_width=True)
            
            if "No" in status:
                st.warning(status)
            else:
                st.success(status)