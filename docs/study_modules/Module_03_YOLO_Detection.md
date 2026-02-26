# 📘 Module 3: Polyp Detection with YOLOv8
## Understanding Object Detection and How YOLO Works

---

## 3.1 What Is Object Detection?

**Object detection** is the task of:
1. Finding **where** objects are in an image (localization → bounding boxes)
2. Identifying **what** each object is (classification → class label)

Unlike simple image classification (which only says "there's a cat in this image"), detection also tells you **where** the cat is using a bounding box.

In our project: "There is a **polyp** at position (x=312, y=250, width=90, height=75)"

---

## 3.2 What Is YOLO?

**YOLO** = **Y**ou **O**nly **L**ook **O**nce

Before YOLO, detection was a two-step process:
1. Generate region proposals (possible locations)
2. Classify each region

YOLO revolutionized this by doing **both steps in a single forward pass through the neural network** — making it dramatically faster (real-time capable).

### How YOLO Works (Conceptually)

```
Input Image (640×640)
       │
       ▼
┌─────────────────────────────────────────┐
│  BACKBONE (Feature Extraction)          │
│  CSPDarknet: Learns edges → shapes      │
│  → textures → abstract features        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  NECK (Feature Pyramid Network)         │
│  Connects features from different       │
│  scales so YOLO detects both            │
│  small and large polyps                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  HEAD (Detection Layer)                 │
│  For each grid cell, predicts:          │
│  - Is there an object here?             │
│  - What class is it?                    │
│  - What are the exact box coordinates?  │
└────────────────┬────────────────────────┘
                 │
                 ▼
     Predictions → Non-Maximum Suppression
                 │
                 ▼
     Final Bounding Boxes + Class Labels
```

### YOLOv8 Variants

| Model | Speed | Accuracy | Use Case |
|---|---|---|---|
| YOLOv8n (nano) | Fastest ⚡⚡⚡ | Lowest | Edge devices |
| **YOLOv8s (small)** | Fast ⚡⚡ | Good | **Our choice** |
| YOLOv8m (medium) | Medium | Better | Balanced |
| YOLOv8l (large) | Slow | High | When accuracy critical |
| YOLOv8x (XL) | Slowest | Highest | Research/cloud |

We use **YOLOv8s** — a balance of speed and accuracy suitable for medical imaging on limited GPU resources.

---

## 3.3 Key Concepts in YOLO Training

### Confidence Score
Every predicted box has a **confidence score** (0.0 to 1.0):
- `0.9` = "I'm 90% sure there's a polyp here"
- `0.25` = "I'm only 25% sure — might be a polyp"
- We set a **threshold** (e.g., 0.25): boxes below this are discarded

### IoU — Intersection over Union
How do we measure if a predicted box is "good"?

```
Predicted Box:     Ground Truth Box:   Overlap:
+--------+         +--------+          +---+
|        |         |   +---+|          |   |  ← Intersection (shared area)
|  +---+ |         |   |   ||          +---+
|  |   | |    +    |   +---+|   =   
|  +---+ |         |        |      IoU = Intersection Area / Union Area
+--------+         +--------+
                              If boxes perfectly overlap: IoU = 1.0
                              If boxes don't overlap at all: IoU = 0.0
```

**Good detection**: IoU > 0.5 is typically considered a correct detection.

### mAP — Mean Average Precision
The standard metric for object detection:
- **Precision**: Of all boxes we predicted, how many were actually polyps?
- **Recall**: Of all actual polyps, how many did we find?
- **AP (Average Precision)**: Area under the Precision-Recall curve (one number summarizing both)
- **mAP**: Average AP across all classes (we only have 1 class: polyp)

### Non-Maximum Suppression (NMS)
YOLO often predicts multiple overlapping boxes for the same object:
```
Without NMS:           With NMS:
+--------+             +--------+
| +-----+|             |        |
| |+----+||    →        |        |  ← Only the best box survives
| |+----+||             |        |
| +-----+|             +--------+
+--------+
```
NMS keeps only the box with the **highest confidence** and removes all others that overlap with it significantly. The **IoU threshold** for NMS controls how much overlap is needed to remove a box.

---

## 3.4 The YOLO Dataset Configuration: `polyp_dataset.yaml`

```yaml
# Path to the root directory of the processed dataset
path: C:/Users/user/OneDrive/Desktop/DetSEG/data/processed

# Train/val/test image paths (relative to 'path' above)
train: images/train
val: images/val
test: images/test

# Class names
names:
  0: polyp
```

### Line-by-Line Explanation:

**`path: C:/Users/.../data/processed`**
The absolute path to the root of our dataset. YOLO uses this as the base when looking for images.

**`train: images/train`**
Combined with `path`, this tells YOLO: "Training images are at `data/processed/images/train/`". YOLO automatically looks for labels in the corresponding `labels/train/` directory.

**`val: images/val`**
Validation images directory. Used after each training epoch to compute mAP and check for overfitting.

**`names: 0: polyp`**
Our dataset has only 1 class (index 0) called "polyp". If we had multiple classes, it would be:
```yaml
names:
  0: polyp
  1: tumor
  2: bleeding
```

---

## 3.5 Training Script: `train_yolo.py` — Line-by-Line Explanation

```python
import os
from ultralytics import YOLO
```
**Imports**: `ultralytics` is the official YOLOv8 library by Ultralytics. One import gives us access to the entire YOLO training, evaluation, and inference pipeline.

---

```python
def train_detection_model():
```
A single function that encapsulates the entire training process. Good practice: keeps main logic organized.

```python
    last_weights_path = 'models/detection/yolov8s_polyp_gpu_final/weights/last.pt'
    model = YOLO(last_weights_path)
```
**Resume from saved weights**: Instead of starting from scratch (which would require loading ImageNet pre-trained weights), we load `last.pt` — the checkpoint saved at the end of the last training run.

- `last.pt`: The model weights saved at the **last completed epoch**. Good for resuming.
- `best.pt`: The weights from the **best epoch** (highest validation mAP). Good for inference.

**Why resume training?** Training can take hours. If it was interrupted, or if we want to train for more epochs, we just load `last.pt` and continue — no wasted computation.

```python
    data_config_path = 'configs/polyp_dataset.yaml'
```
Points to our dataset configuration file. This single YAML file tells YOLO everything about our data.

```python
    epochs = 100        # Train for 100 complete passes through the training data
    batch_size = 16     # Process 16 images simultaneously in one forward pass
    img_size = 640      # Resize all images to 640×640 during training
```
**Hyperparameters** — these control how training happens:
- **Epochs**: One epoch = one complete pass through ALL training data. 100 epochs means YOLO sees each image 100 times.
- **Batch size**: Larger batch = faster GPU utilization but needs more GPU memory. 16 is a good balance.
- **Image size**: YOLO internally resizes all images to 640×640. Larger = better detection of small objects but slower training.

```python
    project_name = 'models/detection'
    experiment_name = 'yolov8s_polyp_gpu_100_epochs'
```
**Experiment tracking**: YOLO saves all results (weights, metrics, plots) to `models/detection/yolov8s_polyp_gpu_100_epochs/`. Using descriptive names lets you compare multiple experiments easily.

```python
    model.train(
        data=data_config_path,      # Where is our data?
        epochs=epochs,              # How many epochs?
        batch=batch_size,           # Batch size
        imgsz=img_size,             # Image size
        project=project_name,       # Where to save results
        name=experiment_name,       # Subfolder name
        exist_ok=True               # Don't error if folder already exists
    )
```
**The Training Call**: `model.train()` handles:
- Loading images and labels from disk in batches
- Forward pass (prediction)
- Computing loss (how wrong we were)
- Backward pass (computing gradients)
- Updating model parameters
- Running validation at end of each epoch
- Saving `last.pt` and `best.pt` checkpoints
- Generating training curve plots

---

## 3.6 What YOLO Learns During Training

YOLO is trained using multiple loss components simultaneously:

| Loss Component | What It Teaches YOLO |
|---|---|
| **Box Loss** | Get the bounding box coordinates right |
| **Classification Loss** | Correctly label the object class (polyp vs. background) |
| **Objectness Loss** (DFL) | Confidence scoring — be more confident when right, less when wrong |

The total loss decreases over epochs as YOLO improves. A typical training curve:

```
Loss
 ▲
 │ ╲
 │  ╲
 │   ╲___
 │       ╲____
 │            ╲________
 │                     ─────────────
 └──────────────────────────────────── Epochs
   0   10   20   30   40   50   100
```

---

## 3.7 Key Files Produced After YOLO Training

After training completes, inside `models/detection/yolov8s_polyp_gpu_100_epochs/`:

```
├── weights/
│   ├── best.pt      ← Best model (highest val mAP). Use this for inference!
│   └── last.pt      ← Last epoch model. Use this to resume training.
│
├── results.csv      ← Per-epoch metrics (box_loss, cls_loss, mAP50, mAP50-95)
├── results.png      ← Visual plot of training curves
├── confusion_matrix.png  ← How often the model got confused
├── PR_curve.png     ← Precision-Recall curve
└── val_batch*.jpg   ← Visual validation predictions vs ground truth
```

---

## 3.8 Our Results Summary

After training on the Kvasir-SEG dataset:

| Metric | Value |
|---|---|
| mAP@50 | ~0.88 (88%) |
| mAP@50-95 | ~0.60 |
| Precision | ~0.86 |
| Recall | ~0.83 |

**Interpretation**: At a 50% IoU threshold, our YOLO model correctly detected polyps with 88% average precision.
