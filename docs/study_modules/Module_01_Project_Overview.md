# 📘 Module 1: Project Overview & Conceptual Foundation
## DetSEG – AI-Powered Polyp Detection and Segmentation

---

## 1.1 What is DetSEG?

**DetSEG** is an intelligent medical image analysis framework designed for the automated **detection** and **segmentation** of polyps in colonoscopy images. The name "DetSEG" combines **"Detection"** (finding where a polyp is) and **"Segmentation"** (outlining the exact shape of a polyp pixel by pixel).

This project solves a real clinical problem: **colorectal cancer is one of the most preventable cancers**, and its prevention depends almost entirely on finding polyps early during colonoscopy. However, studies show that endoscopists miss approximately **22–28% of polyps** during routine colonoscopy. An AI assistant can reduce this miss rate by highlighting suspicious regions in real-time.

---

## 1.2 The Core Problem: Two Complementary Tasks

DetSEG addresses the problem with a **two-stage pipeline**:

```
Colonoscopy Image
     │
     ▼
┌─────────────────────────────┐
│  STAGE 1: DETECTION (YOLO)  │  → "Is there a polyp? Where is it?"
│  Outputs: Bounding Boxes    │    Draws a rectangle around the polyp
└────────────┬────────────────┘
             │
             ▼ (cropped region)
┌─────────────────────────────┐
│  STAGE 2: SEGMENTATION      │  → "What is the exact shape of the polyp?"
│  (U-Net)                    │    Generates a pixel-level mask
│  Outputs: Binary Mask       │
└─────────────────────────────┘
             │
             ▼
   Visualization Overlay
   (bounding box + green mask)
```

### Why Two Stages Instead of One?
- **Detection is fast and coarse**: YOLO can tell you there is something suspicious and roughly where it is in milliseconds.
- **Segmentation is precise but slower**: U-Net can outline the exact boundary of a polyp, giving clinicians precise size and shape information.
- By **combining both**, we get the **speed of YOLO** with the **precision of U-Net**.

---

## 1.3 Key Technologies Used

| Technology | Role in Project | Why Chosen |
|---|---|---|
| **YOLOv8** | Polyp Detection | State-of-the-art real-time object detector |
| **U-Net** | Polyp Segmentation | Designed specifically for biomedical imaging |
| **PyTorch** | Deep Learning Framework | Flexible, powerful, industry standard |
| **OpenCV (cv2)** | Image Processing | Fast image read/write/transform operations |
| **Albumentations** | Data Augmentation | Medical-image-friendly augmentation library |
| **Streamlit** | Web App UI | Rapid prototyping of AI demo apps |
| **Gradio** | Alternative UI | Quick model demos and sharing |

---

## 1.4 Project Directory Structure Explained

```
DetSEG/
│
├── configs/                        # Configuration files
│   └── polyp_dataset.yaml          # Tells YOLO where to find data
│
├── data/                           # All dataset files (not in Git)
│   ├── raw/
│   │   ├── images/                 # Original colonoscopy images
│   │   └── masks/                  # Ground-truth binary mask images
│   ├── interim/
│   │   ├── labels/                 # YOLO-format .txt label files
│   │   └── bounding_boxes.csv      # Raw pixel bbox coordinates
│   └── processed/
│       ├── images/train|val|test/  # Split images for YOLO
│       ├── labels/train|val|test/  # Matched label files
│       ├── masks/train|val|test/   # Matched mask files
│       ├── unet_images/train|val|  # Cropped patches for U-Net
│       └── unet_masks/train|val/   # Corresponding mask patches
│
├── models/                         # Saved model weights
│   ├── detection/                  # YOLO trained weights
│   └── segmentation/               # U-Net trained weights
│
├── src/                            # All source code
│   ├── data_preparation/
│   │   ├── bbox_generator.py       # Mask → YOLO labels + CSV
│   │   ├── data_splitter.py        # Train/Val/Test split
│   │   └── unet_data_generator.py  # Crop patches for U-Net
│   ├── detection/
│   │   └── train_yolo.py           # YOLO training script
│   ├── segmentation/
│   │   ├── unet_model.py           # U-Net architecture
│   │   └── dataset.py              # PyTorch Dataset class
│   └── app.py                      # Streamlit web application
│
├── requirements.txt                # All Python package dependencies
└── research_paper.md               # Full research documentation
```

---

## 1.5 Dataset Used: Kvasir-SEG

The project uses the **Kvasir-SEG** dataset — a widely used open benchmark dataset for polyp segmentation research.

- **Source**: Oslo University Hospital, Norway
- **Size**: 1,000 images with corresponding ground-truth segmentation masks
- **Format**: Each image has a corresponding binary mask where:
  - **White pixels (255)** = Polyp region
  - **Black pixels (0)** = Background / Normal tissue
- **Image sizes**: Vary from 332×487 to 1920×1072 pixels

### What Is a Binary Mask?
A **binary mask** is a black-and-white image the same size as the original colonoscopy image. It acts like a stencil:

```
Original Image:     Binary Mask:        What It Means:
[colonoscopy with   [000000000000        Black = "Not a polyp"
 a polyp visible]   000111110000        White = "This is a polyp!"
                    001111111100
                    000111110000
                    000000000000]
```

---

## 1.6 The Overall Pipeline (Step-by-Step)

```
Step 1: Raw Data
  ├── Colonoscopy images (JPEG/PNG)
  └── Ground truth segmentation masks

Step 2: Data Preparation
  ├── bbox_generator.py:   Mask → Bounding Box coordinates + YOLO labels
  ├── data_splitter.py:    Split 80% Train, 10% Val, 10% Test
  └── unet_data_generator.py: Crop polyp region + resize to 256×256

Step 3: Model Training
  ├── train_yolo.py:  Train YOLOv8s on detection task
  └── train_unet.py:  Train U-Net on segmentation task (patches)

Step 4: Inference / Application
  └── app.py: Upload image → YOLO detects → U-Net segments → Display

Step 5: Evaluation
  └── Metrics: mAP (YOLO), Dice Score / IoU (U-Net)
```

---

## 1.7 Key Concepts to Know Before Starting

### What is Deep Learning?
Deep learning is a branch of machine learning that uses **neural networks** with many layers ("deep" = many layers). These networks learn patterns from large amounts of data.

### What is Object Detection?
Object detection = **Classification + Localization**:
1. *What* is in the image? (Classification: "polyp")
2. *Where* is it? (Localization: "at position x=100, y=150, width=80, height=60")

### What is Image Segmentation?
Segmentation goes **beyond** detection — instead of a box, it assigns a class label to **every single pixel**. This gives us the exact outline of the polyp.

### What is a Bounding Box?
A bounding box (bbox) is a rectangle drawn around an object:
```
+------------------+
|                  |
|   [  Polyp  ]   |   ← This rectangle IS the bounding box
|                  |
+------------------+
(x1, y1) = top-left corner
(x2, y2) = bottom-right corner
OR represented as: (x_center, y_center, width, height) ← YOLO format
```

### What is Normalization?
When neural networks process numbers, they work best when values are in a specific small range. **Normalization** rescales pixel values:
- Original: 0 to 255 (raw pixel values)
- Normalized: ~-2.1 to ~2.6 (using ImageNet statistics)

Formula: `normalized = (pixel_value / 255.0 - mean) / std`
