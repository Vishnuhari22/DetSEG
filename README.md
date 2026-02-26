# DetSEG — Polyp Detection & Segmentation Framework

A deep learning framework for automated **polyp detection** (YOLOv8) and **segmentation** (U-Net) in colonoscopy images and video, with a web-based inference interface.

---

## 🚀 Quick Start (For Team Members)

Follow these steps **in order** to get the project running on your machine.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Vishnuhari22/DetSEG.git
cd DetSEG
```

### Step 2 — Set Up Python Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Download Dataset

The dataset is **not included** in the repo (too large). Download it from the link below and extract it into the `data/` folder.

| File | Link | Extract To |
|------|------|------------|
| Raw images & masks | [📥 Download from Google Drive](<PASTE_DRIVE_LINK_HERE>) | `data/raw/` |
| Processed data (YOLO + U-Net ready) | [📥 Download from Google Drive](<PASTE_DRIVE_LINK_HERE>) | `data/processed/` |

After downloading, your `data/` folder should look like:

```
data/
├── raw/
│   ├── images/
│   └── masks/
├── interim/
│   ├── bounding_boxes.csv
│   └── labels/
└── processed/
    ├── images/
    ├── labels/
    ├── masks/
    ├── unet_images/
    └── unet_masks/
```

### Step 4 — Download Pre-Trained Model Weights

Download the trained model weights and place them in the correct directories.

| Model | Link | Place In |
|-------|------|----------|
| YOLOv8 Detection (`best.pt`) | [📥 Download from Google Drive](<PASTE_DRIVE_LINK_HERE>) | `models/detection/weights/` |
| U-Net Segmentation (`best.pt`) | [📥 Download from Google Drive](<PASTE_DRIVE_LINK_HERE>) | `models/segmentation/weights/` |

### Step 5 — Run the Application

```bash
# Option A: Web application (recommended)
python website/server.py
# Then open http://localhost:8000 in your browser

# Option B: Gradio demo
python src/gradio_demo.py
```

---

## 📁 Project Structure

```
DetSEG/
│
├── configs/                    # Training & dataset configuration files
│   └── polyp_dataset.yaml
│
├── data/                       # Dataset directory (⚠️ download separately)
│   ├── raw/                    #   Original images & masks
│   ├── interim/                #   Intermediate outputs (bounding boxes, labels)
│   └── processed/              #   Model-ready data (YOLO & U-Net formats)
│
├── src/                        # Source code
│   ├── data_preparation/       #   Data preprocessing & augmentation scripts
│   │   ├── bbox_generator.py
│   │   ├── data_splitter.py
│   │   └── unet_data_generator.py
│   ├── detection/              #   YOLOv8 training pipeline
│   │   └── train_yolo.py
│   ├── segmentation/           #   U-Net model & dataset utilities
│   │   ├── unet_model.py
│   │   └── dataset.py
│   ├── app.py                  #   Streamlit inference app
│   ├── gradio_demo.py          #   Gradio inference demo
│   └── video_processor.py      #   Video analysis pipeline
│
├── models/                     # Trained model weights (⚠️ download separately)
│   ├── detection/
│   │   ├── weights/            #   ← place YOLOv8 best.pt here
│   │   └── results/
│   └── segmentation/
│       └── weights/            #   ← place U-Net best.pt here
│
├── results/                    # Inference & evaluation outputs
│   ├── detection/
│   └── segmentation/
│
├── tests/                      # Unit & integration tests
│   ├── test_local_model.py
│   └── test_unet.py
│
├── website/                    # Web application (frontend + backend)
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── server.py
│
├── docs/                       # Research papers, reports & study materials
│   ├── research_paper.tex
│   ├── research_springer.tex
│   ├── research_paper.md
│   ├── methodology_draft.md
│   ├── study_modules/
│   └── ...
│
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## 🛠️ Additional Usage

### Data Preparation (from scratch)

If you need to regenerate the processed data from raw images:

```bash
python src/data_preparation/bbox_generator.py
python src/data_preparation/data_splitter.py
python src/data_preparation/unet_data_generator.py
```

### Training Models

```bash
# Detection (YOLOv8)
python src/detection/train_yolo.py

# Segmentation (U-Net) — see src/segmentation/unet_model.py
```

### Running Tests

```bash
python -m pytest tests/
```

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Make sure your virtual environment is activated and `pip install -r requirements.txt` completed successfully |
| Models not loading | Verify that `best.pt` files are placed in the correct `models/*/weights/` directories |
| CUDA errors | Install the correct PyTorch version for your GPU from [pytorch.org](https://pytorch.org/get-started/locally/) |
| Missing dataset | Download from the Google Drive links above and extract to `data/` |

---

## 📄 License

This project is part of an academic research effort. See `docs/` for published papers and methodology.
