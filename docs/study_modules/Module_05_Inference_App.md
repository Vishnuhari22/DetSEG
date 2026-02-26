# 📘 Module 5: The Inference Pipeline — Streamlit App (`app.py`)
## How the Trained Models Are Used Together in a Real Application

---

## 5.1 What Is Inference?

**Inference** = Using a trained model to make predictions on **new, unseen data**.

During training, the model learns. During inference, the model applies what it learned. The `app.py` file is all about inference — it loads the trained YOLO and U-Net models, and runs them on user-uploaded colonoscopy images.

---

## 5.2 What Is Streamlit?

**Streamlit** is a Python library that turns Python scripts into interactive web applications with minimal code. Instead of building HTML/CSS/JavaScript, you just write Python and Streamlit automatically creates a web UI.

```
Normal Web App:              Streamlit App:
HTML file                    Python script
CSS file        →  →  →      (just one .py file!)
JavaScript file
Backend server
```

---

## 5.3 `app.py` — Complete Line-by-Line Explanation

### Section 1: Imports

```python
import streamlit as st              # Web UI framework
import os                           # File/path operations
import cv2                          # Image processing (OpenCV)
import numpy as np                  # Numerical arrays
import torch                        # PyTorch for ML
import segmentation_models_pytorch as smp  # Pre-built U-Net with ResNet encoder
import albumentations as A          # Image augmentation/transform
from albumentations.pytorch import ToTensorV2  # numpy → PyTorch tensor
from ultralytics import YOLO        # YOLOv8 detection
from PIL import Image               # Python Imaging Library (reads uploaded files)
```

**Note: `segmentation_models_pytorch (smp)`**: In the app, we use `smp.Unet` (a library version of U-Net with a pre-trained ResNet34 encoder) instead of our custom `unet_model.py`. This is because:
- It provides a stronger encoder (ResNet34 pre-trained on ImageNet)
- It's plug-and-play for loading saved weights
- Our training was done using `smp.Unet`

---

### Section 2: Configuration

```python
st.set_page_config(page_title="Polyp Detect & Seg", page_icon="🩺", layout="wide")
```
**Page setup**: Must be the very first Streamlit command. Sets the browser tab title, icon, and layout (`"wide"` = full browser width instead of a narrow centered column).

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```
**Dynamic path resolution**: `__file__` is the path to `app.py` itself.
- `os.path.abspath(__file__)` = `/C:/Users/user/.../DetSEG/src/app.py`
- `os.path.dirname(...)` once = `/C:/Users/user/.../DetSEG/src/`
- `os.path.dirname(...)` twice = `/C:/Users/user/.../DetSEG/`

This finds the project root regardless of where you run the script from.

```python
YOLO_PATH = os.path.join(BASE_DIR, 'models', 'detection', 'weights', 'best.pt')
UNET_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'weights', 'unet_best.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
**Constants**:
- `YOLO_PATH`: Full path to the best YOLO weights from training
- `UNET_PATH`: Full path to best U-Net weights
- `DEVICE`: Automatically detects if a GPU is available (`"cuda"`) or falls back to CPU (`"cpu"`). All tensor operations must run on the same device.

---

### Section 3: Model Loading

```python
@st.cache_resource
def load_models():
```
**`@st.cache_resource`**: This is a Python **decorator** — it modifies the function's behavior. Streamlit re-runs the entire script every time the user interacts with the app. Without caching, models would reload on every button click (taking several seconds each time). `@st.cache_resource` caches the models in memory after the first load, so subsequent runs use the cached version instantly.

```python
    if not os.path.exists(YOLO_PATH):
        st.error(f"❌ YOLO Model missing: {YOLO_PATH}")
        return None, None
    yolo_model = YOLO(YOLO_PATH)
```
**Safety first**: Check that the weight file exists before trying to load it. If missing, show an error in the UI and return `None` — graceful failure, no crash.

`YOLO(YOLO_PATH)` loads the weights and the model architecture simultaneously. The Ultralytics library handles all the complexity.

```python
    unet_model = smp.Unet(
        encoder_name="resnet34",  # Use ResNet34 as the encoder backbone
        encoder_weights=None,     # Don't download ImageNet weights (we have our own)
        in_channels=3,            # RGB input (3 channels)
        classes=1                 # 1 output class (binary segmentation)
    )
```
**Creating U-Net with ResNet34 encoder**:
- ResNet34 is used as the **encoder** (feature extraction backbone) instead of our custom encoder
- `encoder_weights=None`: We won't initialize with ImageNet weights since we'll load our trained weights next
- `in_channels=3`: Input is RGB images
- `classes=1`: Binary output (polyp vs. background)

```python
    unet_model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet_model.to(DEVICE)
    unet_model.eval()
```

**Load weights**:
- `torch.load(UNET_PATH, map_location=DEVICE)`: Loads the saved weight dictionary. `map_location=DEVICE` ensures GPU-trained weights can load on CPU if no GPU is available.
- `load_state_dict(...)`: Copies the saved weights into the model structure
- `.to(DEVICE)`: Moves model parameters to GPU or CPU
- `.eval()`: Switches model to **evaluation mode** (very important!). In eval mode:
  - BatchNorm uses stored running statistics (not per-batch)
  - Dropout (if any) is disabled
  This ensures consistent, deterministic predictions.

---

### Section 4: The Processing Pipeline

```python
def process_image(image, conf_threshold, iou_threshold):
```
This function runs the complete two-stage AI pipeline on one image.

```python
    img_np = np.array(image)               # PIL Image → NumPy array (RGB)
    original_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # RGB → BGR for OpenCV/YOLO
```

**Image format conversion**:
- `PIL.Image` stores pixels as RGB numpy-compatible arrays
- OpenCV expects BGR (blue first)
- We convert explicitly to avoid color problems

```python
    h, w = original_bgr.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    draw_img_bgr = original_bgr.copy()
```

**Initialize output structures**:
- `final_mask`: An empty (all zeros) mask the same size as the input image. U-Net predictions for each detected polyp will be **added** to this mask.
- `draw_img_bgr`: A copy of the original image where we'll draw bounding boxes. We copy so we don't modify the original.
- `.copy()`: Creates a completely independent copy (not just a reference)

---

#### Stage 1: YOLO Detection

```python
    results = yolo(original_bgr, conf=conf_threshold, iou=iou_threshold, verbose=True)
```
**Run YOLO inference**: Passes the BGR image to YOLO with the user-set thresholds:
- `conf=conf_threshold`: Only return detections with confidence ≥ threshold (e.g., 0.25)
- `iou=iou_threshold`: NMS IoU threshold (e.g., 0.45). Lower → more aggressive NMS (removes more boxes)
- `verbose=True`: Print detection info to console/terminal

```python
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            detections.append((x1, y1, x2, y2, conf))
```
**Extract detection results**:
- `results` is a list of `Results` objects (one per image if batching)
- `r.boxes` contains all detected bounding boxes
- `box.xyxy[0]`: Coordinates in (x1,y1,x2,y2) format (top-left and bottom-right corners)
- `.cpu()`: Move tensor from GPU to CPU (needed before converting to numpy)
- `.numpy()`: Convert PyTorch tensor to NumPy array
- `.astype(int)`: Convert float coordinates to integers (pixel coordinates must be whole numbers)
- `box.conf[0].item()`: Get the confidence score as a plain Python float (`.item()` extracts scalar from tensor)

---

#### Stage 2: U-Net Segmentation

```python
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
```
**Preprocessing pipeline for U-Net**: Same transforms applied during training MUST be applied during inference. If you normalized differently during training, the model's internal statistics won't match and predictions will be garbage.

```python
    for (x1, y1, x2, y2, conf) in detections:
        pad = 15
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
```
**Expand detected region with padding**: Adds 15 pixels of context around each YOLO bounding box before passing to U-Net. This gives U-Net some context around the polyp edges, which often improves segmentation quality.

```python
        crop_bgr = original_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
```
**Crop and convert**: Slice out just the polyp region from the full image, then convert to RGB for U-Net (which was trained on RGB).

```python
        augmented = transform(image=crop_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
```
**Prepare input tensor**:
- `transform(image=crop_rgb)` applies Resize + Normalize + ToTensorV2 → returns a dict
- `augmented['image']` extracts the tensor: shape `(3, 256, 256)`
- `.unsqueeze(0)`: Adds a batch dimension → shape `(1, 3, 256, 256)`. U-Net expects batches even for single images.
- `.to(DEVICE)`: Move to GPU/CPU to match the model

```python
        with torch.no_grad():
            logits = unet(input_tensor)
            pred = torch.sigmoid(logits) > 0.5
            pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
```
**Run U-Net inference**:
- `torch.no_grad()`: Disables gradient computation (we're not training, so we don't need gradients). Saves significant memory and computation.
- `logits = unet(input_tensor)`: Forward pass. Output shape: `(1, 1, 256, 256)`
- `torch.sigmoid(logits)`: Convert raw logits to probabilities (0–1)
- `> 0.5`: Binarize: True (1) if probability > 0.5, else False (0)
- `pred[0, 0]`: Extract the single mask from the batch and single channel: shape `(256, 256)`
- `.cpu().numpy().astype(np.uint8)`: Move to CPU, convert to NumPy, cast to 8-bit unsigned int (0 or 1)

```python
        crop_h, crop_w = crop_bgr.shape[:2]
        pred_resized = cv2.resize(pred_np, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        final_mask[y1:y2, x1:x2] = cv2.bitwise_or(final_mask[y1:y2, x1:x2], pred_resized)
```
**Resize and place mask back**:
- The U-Net output is 256×256, but the original cropped region might be different sizes
- Resize back to the original crop dimensions using nearest-neighbor (no blending)
- Place into the correct position in `final_mask` using `bitwise_or` (merges multiple polyp masks if detected)

---

#### Drawing Visualizations

```python
        cv2.rectangle(draw_img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(draw_img_bgr, f"Polyp: {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
```
**Draw bounding box and label**:
- `cv2.rectangle(...)`: Draws a rectangle from (x1,y1) to (x2,y2) in red `(0, 0, 255)` (BGR!) with thickness 3px
- `cv2.putText(...)`: Writes text like "Polyp: 0.87" above the box
  - `(x1, y1-10)`: Position = 10 pixels above the box top-left
  - `cv2.FONT_HERSHEY_SIMPLEX`: Standard OpenCV font
  - `0.6`: Font scale (relative size)
  - `(0, 0, 255)`: Red color (BGR)
  - `2`: Thickness of the text strokes

---

#### Creating the Final Overlay

```python
    result_rgb = cv2.cvtColor(draw_img_bgr, cv2.COLOR_BGR2RGB)
    
    colored_mask = np.zeros_like(result_rgb)
    colored_mask[:, :, 1] = 255  # Green channel = 255, others = 0
    
    mask_indices = final_mask == 1
    result_rgb[mask_indices] = cv2.addWeighted(
        result_rgb[mask_indices], 0.7, colored_mask[mask_indices], 0.3, 0
    )
```

**Alpha blending with green mask**:
- Convert back to RGB for display in Streamlit
- `np.zeros_like(result_rgb)`: Solid black array same size as result image
- `colored_mask[:, :, 1] = 255`: Set green channel to max → pure green `(R=0, G=255, B=0)`
- `final_mask == 1`: Boolean mask (True only where we detected polyp)
- `cv2.addWeighted(src1, alpha, src2, beta, gamma)`: Blends two images:
  - `result_rgb[mask_indices]` × 0.7 + `colored_mask[mask_indices]` × 0.3 + 0
  - Result: 70% original image + 30% green = a semi-transparent green overlay

---

### Section 5: The Streamlit UI

```python
st.title("🩺 AI-Powered Polyp Detection & Segmentation")
```
Renders a large H1 heading in the app.

```python
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.45, 0.05)
```
**Interactive sliders in the sidebar**:
- Arguments: `label, min, max, default_value, step`
- User can drag sliders to adjust thresholds without restarting the app
- The current slider value is returned and stored in the variable

```python
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])
```
**File upload widget**: Creates a file upload button/drop area. `type=["jpg", "png", "jpeg"]` restricts to image files only. If a file is uploaded, `uploaded_file` contains the file data; otherwise it's `None`.

```python
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
```
**Two-column layout**: `st.columns(2)` returns two column objects. Using `with col1:` makes content render in the left column. This is how Streamlit creates side-by-side displays.

```python
    if st.button("🚀 Run Analysis"):
        with st.spinner("Analyzing..."):
            result_img, status = process_image(image, conf_thresh, iou_thresh)
```
**Button + spinner**: `st.button()` returns `True` when clicked. `st.spinner("...")` shows a loading animation while the indented code runs (blocks UI until `process_image` returns).

```python
        if "No" in status:
            st.warning(status)  # Yellow warning box
        else:
            st.success(status)  # Green success box
```
**Status feedback**: Streamlit provides styled notification boxes:
- `st.warning()`: Yellow box for cautionary messages
- `st.success()`: Green box for success messages
- `st.error()`: Red box for errors

---

## 5.4 The Complete Inference Flow Summary

```
User uploads image (JPEG/PNG)
         │
         ▼
PIL.Image.open()
         │            Convert PIL→NumPy→BGR
         ▼
    original_bgr
         │
         ▼
  YOLO(original_bgr)    ← YOLO detects polyps at full resolution (480×640 etc.)
         │
         ▼
  Bounding box list [(x1,y1,x2,y2,conf), ...]
         │
    For each box:
         │
         ▼
  Crop + Pad region from original image
         │
         ▼
  Convert BGR→RGB
         │
         ▼
  Resize to 256×256 + Normalize + ToTensor
         │
         ▼
  unet(input_tensor) → logits
         │
         ▼
  sigmoid(logits) > 0.5 → binary mask (256×256)
         │
         ▼
  Resize mask back to crop size
         │
         ▼
  Paste into final_mask at (y1:y2, x1:x2)
         │
    After all boxes:
         ▼
  Draw boxes on image
  Apply green overlay where final_mask == 1
         │
         ▼
  Display in Streamlit UI
```

---

## 5.5 Key Design Decisions in the App

1. **Why use `best.pt` for YOLO (not `last.pt`)?**
   `best.pt` is the checkpoint with the highest validation mAP — it's the optimal model, not just the most recent.

2. **Why use `smp.Unet` with ResNet34 encoder in the app?**
   The training also used `smp.Unet`. The `state_dict` (saved weights) can only be loaded into an identical architecture. Mismatching architectures cause errors.

3. **Why `map_location=DEVICE` when loading?**
   Weights saved on GPU are stored with GPU memory pointers. `map_location=DEVICE` remaps these to whatever hardware is available at inference time.

4. **Why is padding added around YOLO boxes before U-Net?**
   U-Net performs better when it can see some tissue context around the polyp boundary. The 15px padding provides this context.

5. **Why `torch.no_grad()` during inference?**
   During training, PyTorch builds a computational graph to compute gradients. During inference, we don't need gradients. `no_grad()` skips building this graph, reducing memory usage by ~50% and speeding up inference.
