# 📘 Module 2: Data Preparation
## Understanding How We Prepare Medical Image Data for AI Training

---

## 2.1 Why Is Data Preparation So Important?

AI models don't understand raw folders of images directly. We need to:
1. **Label** the data in formulas that AI tools can use (YOLO format labels)
2. **Split** the data into training, validation, and testing sets
3. **Crop and resize** images to a fixed size for the segmentation model

These three tasks are handled by three separate Python scripts:

| Script | Job |
|---|---|
| `bbox_generator.py` | Convert masks → YOLO bounding box labels |
| `data_splitter.py` | Organize and split data into train/val/test |
| `unet_data_generator.py` | Crop polyp regions and resize for U-Net |

---

## 2.2 Script 1: `bbox_generator.py` — Generating Labels from Masks

### The Problem It Solves
We have **ground-truth segmentation masks** (the white blobs showing polyp location), but YOLO needs **bounding box labels** in a specific text format. This script bridges the gap:

```
INPUT:  A mask image (binary image with white blob = polyp)
OUTPUT: A .txt file (YOLO format) and a CSV with raw coordinates
```

### The YOLO Label Format
Each line in a YOLO `.txt` file means one detected object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values (except class_id) are **normalized** between 0 and 1 (relative to image size):
- If the image is 400px wide and the polyp center is at x=200px, then x_center = 200/400 = **0.5**
- Example: `0 0.502345 0.481234 0.312500 0.287500`
  - `0` = class 0 (polyp)
  - `0.502345` = center x is 50.2% from left edge
  - `0.481234` = center y is 48.1% from top
  - etc.

---

### 📄 `bbox_generator.py` — Line-by-Line Explanation

```python
import os      # For file and directory operations
import cv2     # OpenCV – computer vision library for reading/processing images
import csv     # For writing data in CSV (spreadsheet) format
import argparse  # For accepting command-line arguments (like --mask-dir)
from tqdm import tqdm  # Creates a progress bar in the terminal
```
**Imports**: These are the tools we bring into our script.
- `os`: handles file paths like `os.path.join("data", "masks")` → `"data/masks"`
- `cv2`: reads images, finds edges/contours, draws boxes
- `csv`: writes comma-separated values (spreadsheet-like) files
- `argparse`: lets users run the script with options: `python bbox_generator.py --mask-dir my_masks/`
- `tqdm`: shows `[=====>   ] 45%` progress bars while processing hundreds of images

---

```python
def generate_bbox_from_mask(mask_path, class_id=0):
```
**Function definition**: This function takes:
- `mask_path` (str): The full file path to one mask image
- `class_id=0`: The default class is 0 (polyp). The `=0` means if you don't provide it, it defaults to 0.

```python
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
```
**Read mask as grayscale**: Loads the mask image as a 2D array of numbers. Each pixel is 0 (black=background) or 255 (white=polyp). The flag `cv2.IMREAD_GRAYSCALE` tells OpenCV to ignore color and just read brightness.

```python
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}. Skipping.")
        return [], []
```
**Safety check**: If the file doesn't exist or is corrupted, `cv2.imread` returns `None`. We check for this and skip gracefully (return empty lists `[]`).

```python
    H, W = mask.shape
```
**Get image dimensions**: `mask.shape` returns `(height, width)` for a grayscale image. We store these as `H` and `W` because we'll need them to calculate normalized coordinates.

```python
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
**Find contours**: This is the heart of the function. A **contour** is the boundary/outline of a white region.
- `cv2.RETR_EXTERNAL`: Only find the outermost contours (we don't want "holes" inside polyps)
- `cv2.CHAIN_APPROX_SIMPLE`: Store only the key corner points of the boundary (not every single pixel), which saves memory
- The `_` captures the hierarchy info we don't need, so we discard it.

```python
    for contour in contours:
        if cv2.contourArea(contour) < 20:  # Minimum pixel area threshold
            continue
```
**Filter tiny noise**: The mask might have tiny specks (noise) that aren't real polyps. `cv2.contourArea()` calculates the area in pixels². We skip any contour smaller than 20 pixels² because that's likely noise, not a polyp.

```python
        x, y, w, h = cv2.boundingRect(contour)
```
**Get bounding box**: `cv2.boundingRect` calculates the smallest axis-aligned rectangle around the contour and returns:
- `x`, `y`: top-left corner coordinates (in pixels)
- `w`, `h`: width and height (in pixels)

```python
        raw_bboxes.append({'x': x, 'y': y, 'w': w, 'h': h})
```
**Store raw coordinates**: We save the pixel coordinates to a dictionary and add it to the list. These will be saved to CSV for use by the U-Net data generator.

```python
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        norm_width = w / W
        norm_height = h / H
```
**Convert to normalized YOLO format**:
- `x + w/2` = pixel position of the center (left corner + half width)
- Dividing by `W` (image width) normalizes it to 0–1 range
- Same logic for y_center, width, and height

```python
        x_center = max(0.0, min(1.0, x_center))
```
**Clamp values**: This ensures values never go below 0.0 or above 1.0. `max(0.0, ...)` ensures at least 0, `min(1.0, ...)` ensures at most 1. This prevents any floating point edge cases.

```python
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
```
**Format YOLO label string**: Creates a string like `"0 0.502000 0.481000 0.312500 0.287500"`. The `:.6f` means 6 decimal places.

---

```python
def process_all_masks(raw_mask_dir, label_output_dir, csv_output_path):
```
**Main processing function**: Processes ALL masks in a directory, not just one.

```python
    os.makedirs(label_output_dir, exist_ok=True)
```
**Create output directory**: `os.makedirs` creates the folder. `exist_ok=True` means don't raise an error if the folder already exists.

```python
    mask_files = [f for f in os.listdir(raw_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
```
**List comprehension**: This one line does the work of a for loop. It reads all files in the directory, but keeps ONLY image files (filtering by extension). Result: `['mask001.png', 'mask002.png', ...]`

```python
    with open(csv_output_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'x', 'y', 'w', 'h']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
```
**Open CSV for writing**: `with open(...)` ensures file is properly closed after use. `DictWriter` allows writing dictionaries (key-value pairs) as rows. `writeheader()` writes the column names as the first row.

```python
    for mask_filename in tqdm(mask_files, desc="Generating Bounding Boxes"):
```
**Loop with progress bar**: `tqdm` wraps the list and shows a progress bar. `desc=` sets the label shown on the progress bar.

---

## 2.3 Script 2: `data_splitter.py` — Organizing Data into Train/Val/Test

### Why Do We Split Data?

Neural networks learn by seeing examples (**training set**). But we need to know whether the model actually *understood* the concept or just *memorized* the data. We use:

| Split | Proportion | Purpose |
|---|---|---|
| **Training** | 80% | The model learns patterns from this data |
| **Validation** | 10% | Monitor training; tune hyperparameters |
| **Test** | 10% | Final unbiased evaluation of model quality |

> **Analogy**: Training set = Your textbook. Validation set = Practice exams while studying. Test set = The final exam (only opened after studying is done).

### Key Concept: `random.seed(42)`
When you shuffle a list randomly, the result changes every time you run the code. A **seed** makes the randomness reproducible — the same split every time. `42` is conventionally used (from "The Hitchhiker's Guide to the Galaxy").

---

### 📄 `data_splitter.py` — Line-by-Line Explanation

```python
import random   # Python's built-in random number generator
import shutil   # High-level file operations (e.g., copy files)
```
- `random.shuffle()` randomly reorders a list
- `shutil.copy(src, dst)` copies a file from source path to destination path

```python
def find_image_extension(image_dir, base_filename):
    for ext in ['.jpg', '.png', '.jpeg']:
        if os.path.exists(os.path.join(image_dir, base_filename + ext)):
            return ext
    return None
```
**Helper function**: Since images might be `.jpg` or `.png`, this tries each extension until it finds the file. Returns the extension string (e.g., `'.jpg'`) or `None` if not found.
- `os.path.join(image_dir, base_filename + ext)` builds a full path like `data/raw/images/cju0qkwl35piu0993l0dewei2.jpg`
- `os.path.exists(...)` returns `True` if the file exists

```python
    random.seed(42)
    random.shuffle(all_filenames)
```
**Reproducible shuffle**: Sets the random seed then shuffles the filename list in-place (modifies the list directly). After this, the list is in a random-but-consistent order.

```python
    train_end = int(total_files * args.split_ratios[0])   # e.g., 1000 * 0.8 = 800
    val_end = train_end + int(total_files * args.split_ratios[1])  # 800 + 100 = 900
    
    train_files = all_filenames[:train_end]    # First 800 items
    val_files = all_filenames[train_end:val_end]  # Items 800–899
    test_files = all_filenames[val_end:]       # Items 900–999
```
**Slice-based splitting**: Python list slices `[start:end]` extract portions of a list. This is elegant and fast.

```python
    os.makedirs(os.path.join(args.output_dir, 'images', split_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'labels', split_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks', split_name), exist_ok=True)
```
**Create directory structure**: For each of train, val, test — creates 3 folders each. Result:
```
data/processed/
├── images/train/   ├── images/val/   ├── images/test/
├── labels/train/   ├── labels/val/   ├── labels/test/
└── masks/train/    └── masks/val/    └── masks/test/
```

```python
    shutil.copy(src_img_path, dest_img_path)
    shutil.copy(src_mask_path, dest_mask_path)
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dest_label_path)
```
**Copy files**: `shutil.copy` copies the file **bytes exactly** to the new location. We check if the label exists first — some images might not have a polyp (empty/negative case), so label files might be absent.

---

## 2.4 Script 3: `unet_data_generator.py` — Preparing Patches for U-Net

### Why Do We Need Separate Patches for U-Net?

U-Net is trained to **segment** (outline) a polyp, not to find it. So:
- Instead of giving U-Net the entire colonoscopy image (which is mostly normal tissue), we give it a **cropped region centered on the polyp**
- We add padding around the bounding box so U-Net sees some context
- We resize everything to **256×256 pixels** (fixed size required by U-Net)

```
Full Image (e.g., 1280×960):          U-Net Patch (256×256):
+---------------------------+          +----------+
|                           |          |          |
|   +---------+             |   →      | [polyp]  |  ← Cropped, padded, resized
|   | [POLYP] |             |          |          |
|   +---------+             |          +----------+
+---------------------------+
```

### Key Concept: Why Two Interpolation Methods?
When resizing an image, OpenCV needs to "invent" or "remove" pixels:
- **`cv2.INTER_LINEAR`** (Bilinear) for images: Smooth blending of nearby pixels. Good for photographs where smooth color transitions are important.
- **`cv2.INTER_NEAREST`** (Nearest Neighbor) for masks: Copies the nearest pixel exactly. Critical for masks because we need ONLY 0 or 255 — no blending that would create intermediate values like 127!

---

### 📄 `unet_data_generator.py` — Line-by-Line Explanation

```python
    bbox_data = {}
    with open(args.csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bbox_data[row['filename']] = {
                'x': int(row['x']),
                'y': int(row['y']),
                'w': int(row['w']),
                'h': int(row['h'])
            }
```
**Load bounding boxes from CSV**: Reads the CSV created by `bbox_generator.py`. `csv.DictReader` reads each row as a dictionary keyed by column names. The result is a lookup dictionary: `{'cju0abc.png': {'x':100, 'y':50, 'w':80, 'h':60}, ...}`

`int(row['x'])` converts the string `'100'` from the CSV to the integer `100`.

```python
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_W, x + w + padding)
    y2 = min(img_H, y + h + padding)
```
**Add padding with boundary clipping**:
- `x - padding` = 20px to the left of the bounding box
- `max(0, ...)` = never go below 0 (can't crop outside the left edge of the image)
- `x + w + padding` = 20px to the right of the bounding box
- `min(img_W, ...)` = never exceed image width (can't crop outside the right edge)

This ensures the cropped region always stays inside the original image.

```python
    cropped_image = original_image[y1:y2, x1:x2]
    cropped_mask = original_mask[y1:y2, x1:x2]
```
**NumPy array slicing**: OpenCV images are stored as NumPy arrays. Slicing `[y1:y2, x1:x2]` extracts a rectangular subarray. The order is `[rows, columns]` which maps to `[y, x]` in image coordinates.

```python
    resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, output_size, interpolation=cv2.INTER_NEAREST)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
```
**Resize and binarize**:
1. Reset image to 256×256 with smooth interpolation
2. Reset mask to 256×256 with exact nearest-neighbor (no blending)
3. `cv2.threshold(..., 127, 255, cv2.THRESH_BINARY)` ensures mask is strictly binary: any value ≤127 becomes 0, any value ≥128 becomes 255. The `_` discards the threshold value returned (we don't need it).

```python
    cv2.imwrite(out_img_path, resized_image)
    cv2.imwrite(out_mask_path, resized_mask)
```
**Save as PNG**: PNG format is **lossless** (no quality loss from compression). This is critical for masks — JPEG compression would blur the pixel values and corrupt the clean 0/255 boundaries.

---

## 2.5 Summary: Data Flow Diagram

```
data/raw/images/       data/raw/masks/
      │                      │
      └──────────────────────┘
                 │
         bbox_generator.py
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
data/interim/labels/    data/interim/
(YOLO .txt files)       bounding_boxes.csv
      │                     │
      └──────────┬──────────┘
                 │
         data_splitter.py
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
  train/       val/         test/
(800 images) (100 images) (100 images)
(YOLO use)      │
                │
        unet_data_generator.py
                │
    ┌───────────┴────────────┐
    ▼                        ▼
unet_images/train|val   unet_masks/train|val
(256×256 PNG patches)   (256×256 binary masks)
     (U-Net use)
```
