# 📘 Module 4: Polyp Segmentation with U-Net
## Understanding the U-Net Architecture and Implementation

---

## 4.1 What Is Image Segmentation?

**Image segmentation** assigns a label to **every pixel** in an image.

In polyp detection:
- **Input**: A 256×256 RGB image of a polyp region (crop from colonoscopy)
- **Output**: A 256×256 binary mask where each pixel is 0 (not polyp) or 1 (polyp)

```
Input Image (256×256 RGB):       Output Mask (256×256 Binary):
+------------------------+        +------------------------+
|  [mostly dark tissue]  |        |  000000000000000000000 |
|  [pinkish tissue]      |   →    |  001111111111110000000 |
|  [lighter patch=polyp] |        |  011111111111111100000 |
|  [more tissue]         |        |  001111111111110000000 |
+------------------------+        +------------------------+
                                 0 = Background, 1 = Polyp
```

---

## 4.2 Why U-Net for Medical Image Segmentation?

**U-Net** was invented specifically for biomedical image segmentation in 2015 at the University of Freiburg. It was designed to work even with **very limited training data** — a common challenge in medical imaging (getting labeled medical images is expensive and requires expert doctors).

### Key Innovation: Skip Connections
U-Net's main contribution is **skip connections** (also called "encoder-decoder skip connections"):
- The encoder compresses the image (loses spatial detail but gains understanding)
- The decoder reconstructs the segmentation map
- Skip connections pass **spatial detail** from encoder layers directly to decoder layers

This allows U-Net to produce **precise, fine-grained segmentation masks** even with small datasets.

---

## 4.3 U-Net Architecture: The "U" Shape

The name "U-Net" comes from its shape — it looks like the letter U:

```
Input (3×256×256)
       │
   ┌───▼───┐
   │  inc  │─────────────────────────────────────────┐ Skip 1
   │64ch   │                                          │
   └───┬───┘                                          │
   ┌───▼───┐                                          │
   │ down1 │──────────────────────────────────┐       │ Skip 2
   │128ch  │                                  │       │
   └───┬───┘                                  │       │
   ┌───▼───┐                                  │       │
   │ down2 │────────────────────────┐         │       │ Skip 3
   │256ch  │                        │         │       │
   └───┬───┘                        │         │       │
   ┌───▼───┐                        │         │       │
   │ down3 │────────────┐           │         │       │ Skip 4
   │512ch  │            │           │         │       │
   └───┬───┘            │           │         │       │
   ┌───▼───┐            │           │         │       │
   │ down4 │ (bottle-   │           │         │       │
   │512ch  │  neck)     │           │         │       │
   └───┬───┘            │           │         │       │
       │    ┌───────────▼───┐       │         │       │
       └───►│    up1/512ch  │       │         │       │
            └───────┬───────┘       │         │       │
                    │   ┌───────────▼───┐     │       │
                    └──►│    up2/256ch  │     │       │
                        └───────┬───────┘     │       │
                                │   ┌─────────▼───┐   │
                                └──►│    up3/128ch │   │
                                    └───────┬──────┘   │
                                            │   ┌──────▼──┐
                                            └──►│  up4/64ch│
                                                └───┬──────┘
                                                    │
                                               ┌────▼────┐
                                               │  outc   │
                                               │ 1 class │
                                               └─────────┘
                                          Output (1×256×256)
```

---

## 4.4 `unet_model.py` — Line-by-Line Explanation

### Imports

```python
import torch                           # PyTorch base library
import torch.nn as nn                  # Neural network building blocks
import torchvision.transforms.functional as TF  # For padding operations
```

- `torch`: The core library, handles tensors (multi-dimensional arrays) and automatic differentiation
- `torch.nn`: Contains all neural network layers (Conv2d, BatchNorm2d, ReLU, etc.)
- `TF`: We use it for `TF.pad()` — a precise way to add pixels around tensors

---

### Building Block 1: `DoubleConv`

```python
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
```

**Class inheritance**: `nn.Module` is PyTorch's base class for all neural network components. By inheriting from it, our class gets all the machinery needed (parameter tracking, `forward()` method, GPU transfer, etc.).

```python
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
```

**Constructor**: Called when we create a `DoubleConv(64, 128)`. 
- `in_channels=64`: The layer receives 64 feature maps as input
- `out_channels=128`: The layer outputs 128 feature maps
- `mid_channels`: Optional intermediate channel count. If not given, defaults to `out_channels`.
- `super().__init__()`: Calls the parent class's constructor (required for PyTorch modules)

```python
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

**`nn.Sequential`**: Creates a container where layers are applied in order, like a pipeline. Equivalent to chaining functions.

**`nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)`**:
The most important layer in deep learning. What it does:
- Uses a 3×3 "filter" (kernel) that slides across the image
- At each position, multiplies 3×3 patch by the 3×3 weights and sums the result
- This detects specific patterns (edges, textures, shapes)
- `kernel_size=3`: 3×3 filter
- `padding=1`: Adds 1 pixel of zeros around the input so output size = input size (same padding)
- `bias=False`: No additive bias term (BatchNorm makes bias redundant)

```
How a 3×3 Convolution works:
   Input patch     ×     Kernel      =    One output value
   [128, 139, 134]     [-1, -1, -1]
   [135, 145, 130]  ×  [-1,  8, -1]  =  147 (edge detected!)
   [120, 125, 118]     [-1, -1, -1]
```

**`nn.BatchNorm2d(mid_channels)`** (Batch Normalization):
- Normalizes the outputs of the convolution across the batch
- Reduces "internal covariate shift" (prevents training instability)
- Also acts as a form of regularization (reduces overfitting)
- Effect: Makes training faster and more stable

**`nn.ReLU(inplace=True)`** (Rectified Linear Unit):
- Activation function: sets all negative values to 0
- `f(x) = max(0, x)`
- Why? Without non-linear activation functions, stacking many linear layers is equivalent to just one linear layer (the whole "deep" part becomes useless)
- `inplace=True`: Modifies the tensor directly in memory (saves RAM, slight speedup)

```
ReLU:
     Output
       ↑
       │    /
       │   /
       │  /
───────┼─/──────────→ Input
       │    negative → 0
```

---

### Building Block 2: `Down` (Encoder Step)

```python
class Down(nn.Module):
    """Downscaling block: MaxPool followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
```

**`nn.MaxPool2d(2)`**: Shrinks the spatial dimensions by half:
- Takes a 2×2 window and outputs only the **maximum value**
- A 256×256 feature map becomes 128×128
- This allows the network to "look at" a larger area of the image with each subsequent layer

```
MaxPool2d(2) example (on one channel):
Before:                After:
[3,  7,  1,  4]       [7,  5]
[8,  2,  5,  0]  →    [9,  6]
[9,  1,  3,  6]
[4,  5,  2,  1]
Takes max of each 2×2 block
```

Why MaxPool? It provides **translation invariance** (a polyp slightly to the left or right should give the same detection) and compresses the representation.

---

### Building Block 3: `Up` (Decoder Step with Skip Connection)

```python
class Up(nn.Module):
    """Upscaling block: Upsample/ConvTranspose followed by DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
```

**Two upsampling methods**:
1. **Bilinear upsampling** (`mode='bilinear'`): Simply interpolates (guesses) intermediate pixel values mathematically. Fast, no learned parameters.
2. **Transposed Convolution** (`ConvTranspose2d`): Learnable upsampling — the network learns how to upsample. More powerful but more parameters.

We use bilinear (simpler, works well in practice).

```python
    def forward(self, x1, x2):
        x1 = self.up(x1)           # Upsample x1 from decoder
        
        diffY = x2.size()[2] - x1.size()[2]  # Height difference
        diffX = x2.size()[3] - x1.size()[3]  # Width difference
        
        x1 = TF.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
```

**Why padding is needed**: After upsampling, the decoder feature map might be 1 pixel smaller than the encoder feature map (due to odd-sized inputs). We pad `x1` to match `x2`'s size **exactly** before concatenating.

`x2.size()[2]` = height dimension of x2 (encoder feature map)
`x1.size()[2]` = height dimension of x1 (decoder feature map)

The padding `[left, right, top, bottom]` adds zeros on each side to fix the size mismatch.

```python
        x = torch.cat([x2, x1], dim=1)  # Concatenate along channel dimension
        return self.conv(x)
```

**`torch.cat`**: This IS the skip connection! It concatenates:
- `x2`: Feature maps from the matching encoder level (contains fine spatial details)
- `x1`: Feature maps from the decoder (contains high-level context)
- `dim=1`: The channel dimension

Result: Channel count doubles (64+64=128). DoubleConv then compresses this back.

**WHY? The key insight**: Encoder knows "what" is in the image (semantic understanding). The fine spatial details (exact pixel locations of boundaries) were lost during MaxPooling. The skip connection **restores** that detail for precise boundary segmentation.

---

### Building Block 4: `OutConv` (Final Layer)

```python
class OutConv(nn.Module):
    """Final output convolution layer (1x1)"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```

**1×1 Convolution**: A convolution with kernel_size=1 doesn't look at neighboring pixels — it simply mixes channels together. It's used here to convert from 64 feature channels to 1 output channel (the segmentation map). No spatial information is changed, only channel count.

---

### The Main UNet Class

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # 3 (RGB)
        self.n_classes = n_classes    # 1 (binary: polyp or not)
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)     # 3→64 channels
        self.down1 = Down(64, 128)                # 64→128 channels, /2 spatial
        self.down2 = Down(128, 256)               # 128→256 channels, /2 spatial
        self.down3 = Down(256, 512)               # 256→512 channels, /2 spatial
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)    # 512→512 channels, /2 spatial
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)  # Upsample + concat → 512ch
        self.up2 = Up(512, 256 // factor, bilinear)   # Upsample + concat → 256ch
        self.up3 = Up(256, 128 // factor, bilinear)   # Upsample + concat → 128ch
        self.up4 = Up(128, 64, bilinear)              # Upsample + concat → 64ch
        
        # Output
        self.outc = OutConv(64, n_classes)            # 64→1 channel
```

**The encoder doubles channels while halving spatial size**:
```
Input:  3 × 256 × 256
inc:   64 × 256 × 256
down1: 128 × 128 × 128
down2: 256 ×  64 ×  64
down3: 512 ×  32 ×  32
down4: 512 ×  16 ×  16   ← Bottleneck (smallest, most abstract)
```

**The decoder halves channels while doubling spatial size**:
```
down4: 512 × 16 × 16 (skip from down3: 512 × 32 × 32)
up1:   512 × 32 × 32   ← combines down3 skip + upsampled down4
up2:   256 × 64 × 64   ← combines down2 skip + upsampled up1
up3:   128 × 128 × 128 ← combines down1 skip + upsampled up2
up4:    64 × 256 × 256 ← combines inc skip + upsampled up3
outc:    1 × 256 × 256 ← Final segmentation map (same size as input!)
```

```python
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # Save for skip connection
        x2 = self.down1(x1)  # Save for skip connection
        x3 = self.down2(x2)  # Save for skip connection
        x4 = self.down3(x3)  # Save for skip connection
        x5 = self.down4(x4)  # Bottleneck
        
        # Decoder + Skip Connections
        x = self.up1(x5, x4)  # x5 is decoder; x4 is skip
        x = self.up2(x, x3)   # x3 is skip from level 3
        x = self.up3(x, x2)   # x2 is skip from level 2
        x = self.up4(x, x1)   # x1 is skip from level 1
        
        # Output
        logits = self.outc(x)  # Raw scores (not yet probabilities)
        return logits
```

**Why "logits" not "probabilities"?** The final layer outputs raw unnormalized scores (can be any real number). We apply `sigmoid` AFTER to get probabilities (0–1). This separation is for numerical stability when computing the loss function.

**Sigmoid**: `sigmoid(x) = 1 / (1 + e^(-x))`
- Input: any real number (-∞ to +∞)
- Output: 0 to 1 (a probability)
- `sigmoid(0) = 0.5` (equally uncertain)
- `sigmoid(4) ≈ 0.98` (confident it's a polyp)
- `sigmoid(-4) ≈ 0.02` (confident it's NOT a polyp)

---

## 4.5 `dataset.py` — The PyTorch Dataset Class

### What Is a PyTorch Dataset?
PyTorch has a standard interface for loading data: the `Dataset` class. Any class that inherits from `torch.utils.data.Dataset` must implement:
1. `__len__()`: Returns total number of samples
2. `__getitem__(index)`: Returns one sample (image, mask) pair at the given index

The `DataLoader` then uses this class to create **batches** automatically.

```python
class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = sorted([f for f in os.listdir(image_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
```

**Sorted list**: `sorted()` ensures files are always in alphabetical order, making the dataset deterministic and reproducible.

**List comprehension with filter**: `[f for f in os.listdir(...) if f.lower().endswith(...)]` reads all files in the directory and keeps only image files. `.lower()` handles `IMAGE.PNG`, `image.png`, `Image.PNG` all the same way.

```python
    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)
```

When PyTorch calls `dataset[5]`, this method runs with `index=5`. It looks up the filename at position 5 in the sorted list and constructs the full paths.

```python
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

**BGR to RGB**: OpenCV reads images in **BGR** order (Blue, Green, Red) — a legacy from early Windows API. But neural networks (and most of the world) expect **RGB** (Red, Green, Blue). We must convert, otherwise colors are wrong and the model gets confused.

```python
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension → (H, W, 1)
```

**Mask preprocessing steps**:
1. **Threshold**: Binarize — any value ≤127 → 0, any value ≥128 → 255 (strictly 0 or 255)
2. **Normalize**: Divide by 255.0 → values become 0.0 or 1.0 (matches sigmoid output range)
3. **expand_dims**: Add a channel dimension. Shape changes from `(256, 256)` to `(256, 256, 1)`. Neural networks expect a channel dimension even for single-channel data.

```python
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
```

**Albumentations**: The transform pipeline is applied **to both image and mask simultaneously**. This is crucial: if we flip the image horizontally, we MUST also flip the mask. Albumentations handles this automatically via the `mask=mask` argument.

```python
        mask = mask.float()
        
        if mask.dim() == 3 and mask.shape[-1] == 1:  # HWC format [256, 256, 1]
            mask = mask.permute(2, 0, 1)             # → CHW format [1, 256, 256]
```

**Dimension reordering**: PyTorch tensors use `(Channels, Height, Width)` order (CHW). NumPy/OpenCV use `(Height, Width, Channels)` order (HWC). We use `.permute(2, 0, 1)` to rearrange:
- Dimension 0 (H=256) → becomes dimension 1
- Dimension 1 (W=256) → becomes dimension 2  
- Dimension 2 (C=1) → becomes dimension 0

---

## 4.6 Data Augmentation with Albumentations

**Augmentation** artificially increases the effective size of your training dataset by applying random transformations to images. For medical images, common augmentations include:

```python
train_transform = A.Compose([
    A.Resize(height=256, width=256),        # Fixed size
    A.HorizontalFlip(p=0.5),               # 50% chance to flip left-right
    A.VerticalFlip(p=0.5),                 # 50% chance to flip up-down
    A.RandomRotate90(p=0.5),               # 50% chance to rotate 90°
    A.RandomBrightnessContrast(p=0.2),     # 20% chance to adjust brightness
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ToTensorV2()                            # numpy → PyTorch tensor
])
```

**Why ImageNet statistics?** Even though we fine-tune on colonoscopy images, the U-Net encoder often starts from ImageNet pre-trained weights. Using ImageNet normalization statistics ensures compatibility.

---

## 4.7 Loss Function and Metrics for Segmentation

### Binary Cross-Entropy (BCE) Loss
Used for binary classification at each pixel:
```
BCE Loss = -[y * log(p) + (1-y) * log(1-p)]
```
- `y` = 1 if pixel is polyp, 0 if background
- `p` = model's predicted probability
- Perfect prediction has BCE = 0
- Completely wrong prediction has BCE → ∞

### Dice Loss
Specifically designed for segmentation — directly optimizes the overlap metric:
```
Dice Coefficient = (2 × |Prediction ∩ Ground Truth|) / (|Prediction| + |Ground Truth|)
Dice Loss = 1 - Dice Coefficient
```
Dice handles **class imbalance** well (polyps are small → mostly background pixels).

### Combined Loss (BCE + Dice)
Our model is often trained with:
```
Total Loss = BCE Loss + Dice Loss
```
This gives the benefits of both: pixel-wise accuracy (BCE) AND shape overlap accuracy (Dice).

---

## 4.8 Evaluation Metrics for Segmentation

| Metric | Formula | Interpretation |
|---|---|---|
| **Dice Score** | 2×TP / (2×TP + FP + FN) | % overlap between prediction and ground truth |
| **IoU** | TP / (TP + FP + FN) | Stricter overlap metric |
| **Precision** | TP / (TP + FP) | Of all pixels flagged as polyp, how many truly are? |
| **Recall** | TP / (TP + FN) | Of all actual polyp pixels, how many did we find? |

**Our Results**:
- Dice Score: ~0.82 (82% overlap with ground truth)
- IoU: ~0.75
