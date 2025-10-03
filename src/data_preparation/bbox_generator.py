import os
import cv2
import argparse
from tqdm import tqdm

def generate_bbox_from_mask(mask_path, class_id=0):
    """
    Generates a YOLO formatted bounding box string from a single mask image.

    Args:
        mask_path (str): The full path to the mask image file.
        class_id (int): The class ID for the object (default is 0 for polyp).

    Returns:
        list: A list of YOLO formatted annotation strings. Returns empty list if no contours found.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}. Skipping.")
        return []

    H, W = mask.shape
    if H == 0 or W == 0:
        print(f"Warning: Invalid mask dimensions for {mask_path}. Skipping.")
        return []

    # Find contours of the polyp(s) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_labels = []
    for contour in contours:
        # Filter out very small contours that might be noise
        if cv2.contourArea(contour) < 20: # Minimum pixel area
            continue

        # Get bounding box from contour
        x, y, w, h = cv2.boundingRect(contour)

        # Convert to YOLO format (normalized)
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        norm_width = w / W
        norm_height = h / H

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    return yolo_labels

def process_all_masks(raw_mask_dir, label_output_dir):
    """
    Processes all masks in a directory to generate YOLO formatted label files.
    """
    print(f"Reading masks from: {raw_mask_dir}")
    print(f"Saving labels to: {label_output_dir}")
    
    os.makedirs(label_output_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(raw_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not mask_files:
        raise FileNotFoundError(f"No mask images found in '{raw_mask_dir}'. Please check the path.")

    print(f"Found {len(mask_files)} masks. Starting bounding box generation...")

    for mask_filename in tqdm(mask_files, desc="Generating Bounding Boxes"):
        mask_path = os.path.join(raw_mask_dir, mask_filename)
        yolo_labels = generate_bbox_from_mask(mask_path)

        if yolo_labels:
            base_filename = os.path.splitext(mask_filename)[0]
            label_filepath = os.path.join(label_output_dir, f"{base_filename}.txt")
            
            with open(label_filepath, 'w') as f:
                f.write("\n".join(yolo_labels))

    print(f"✅ Bounding box generation complete. Labels saved in '{label_output_dir}'.")

if __name__ == '__main__':
    # This allows running the script from the command line
    parser = argparse.ArgumentParser(description="Generate YOLO bounding box labels from segmentation masks.")
    parser.add_argument('--mask-dir', type=str, default='data/raw/masks', help='Directory containing the raw mask images.')
    parser.add_argument('--out-dir', type=str, default='data/interim/labels', help='Directory to save the generated YOLO .txt labels.')
    
    args = parser.parse_args()
    
    # We will create an 'interim' folder to hold the 1000 generated labels.
    # The next script, 'data_splitter.py', will then organize these into train/val/test.
    os.makedirs(os.path.dirname(args.out_dir), exist_ok=True)
    
    process_all_masks(raw_mask_dir=args.mask_dir, label_output_dir=args.out_dir)