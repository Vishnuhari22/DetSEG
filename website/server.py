"""
DetSEG Website — Flask Server
==============================
Serves the static website and provides API endpoints for AI inference.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import sys
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
import io
import base64
import time
import tempfile
import traceback
import uuid

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBSITE_DIR = os.path.join(BASE_DIR, 'website')
YOLO_PATH = os.path.join(BASE_DIR, 'models', 'detection', 'weights', 'best.pt')
UNET_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'weights', 'unet_best.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__, static_folder=WEBSITE_DIR)

# Global model variables
yolo_model = None
unet_model = None
sahi_detection_model = None

# Processed video storage (keeps last N videos in memory for serving)
processed_videos = {}  # {video_id: output_path}
MAX_STORED_VIDEOS = 3

# ==========================================
# MODEL LOADING
# ==========================================
def load_models():
    global yolo_model, unet_model

    print(f"  Loading YOLO from: {YOLO_PATH}")
    if not os.path.exists(YOLO_PATH):
        raise FileNotFoundError(f"YOLO model not found at: {YOLO_PATH}")
    yolo_model = YOLO(YOLO_PATH)

    # Wrap YOLO model for SAHI sliced inference
    global sahi_detection_model
    sahi_detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=YOLO_PATH,
        confidence_threshold=0.25,
        device=DEVICE
    )
    print("SAHI detection model wrapped successfully!")

    print(f"  Loading U-Net from: {UNET_PATH}")
    if not os.path.exists(UNET_PATH):
        raise FileNotFoundError(f"U-Net model not found at: {UNET_PATH}")

    unet_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    unet_model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet_model.to(DEVICE)
    unet_model.eval()

    print("Models loaded successfully!")

# ==========================================
# SAHI DEEP SCAN DETECTION
# ==========================================
def run_sahi_detection(img_bgr, conf_threshold=0.25, iou_threshold=0.45):
    """Run SAHI sliced inference for enhanced small/multiple polyp detection."""
    # Update confidence threshold for this run
    sahi_detection_model.confidence_threshold = conf_threshold

    # Convert BGR to RGB — SAHI processes images in RGB format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # Use adaptive slice size: ~1/2 of the smaller dimension, clamped between 256-640
    slice_size = max(256, min(640, min(h, w) // 2))

    print(f"Deep Scan: image={w}x{h}, slice={slice_size}, conf={conf_threshold}, iou={iou_threshold}")

    result = get_sliced_prediction(
        image=img_rgb,
        detection_model=sahi_detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        perform_standard_pred=True,
        postprocess_type="NMS",
        postprocess_match_threshold=iou_threshold,
        verbose=0
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        detections.append({
            "x1": int(bbox.minx),
            "y1": int(bbox.miny),
            "x2": int(bbox.maxx),
            "y2": int(bbox.maxy),
            "confidence": float(pred.score.value)
        })

    print(f"Deep Scan result: {len(detections)} polyps detected")
    return detections


# ==========================================
# IMAGE PROCESSING
# ==========================================
def process_image_pipeline(image_bytes, conf_threshold=0.25, iou_threshold=0.45, mask_opacity=0.4, deep_scan=False):
    """Process an image through YOLO detection + U-Net segmentation."""
    start_time = time.time()

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None, {"error": "Could not decode image"}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # YOLO Detection (standard or SAHI deep scan)
    if deep_scan:
        detections = run_sahi_detection(img_bgr, conf_threshold, iou_threshold)
    else:
        results = yolo_model(img_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                detections.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "confidence": float(conf)})

    detection_img = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(detection_img, f"Polyp: {conf:.0%}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    detection_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)

    if not detections:
        elapsed = time.time() - start_time
        det_b64 = numpy_to_base64(detection_rgb)
        return det_b64, None, {
            "polyps_found": 0,
            "processing_time": round(elapsed, 2),
            "device": DEVICE,
            "deep_scan": deep_scan,
            "detections": []
        }

    # U-Net Segmentation
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    final_mask = np.zeros((h, w), dtype=np.uint8)
    seg_img = img_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        pad = 15
        x1_pad, y1_pad = max(0, x1 - pad), max(0, y1 - pad)
        x2_pad, y2_pad = min(w, x2 + pad), min(h, y2 + pad)

        crop_bgr = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop_bgr.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        augmented = transform(image=crop_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = unet_model(input_tensor)
            pred = torch.sigmoid(logits) > 0.5
            pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)

        crop_h, crop_w = crop_bgr.shape[:2]
        pred_resized = cv2.resize(pred_np, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        final_mask[y1_pad:y2_pad, x1_pad:x2_pad] = cv2.bitwise_or(
            final_mask[y1_pad:y2_pad, x1_pad:x2_pad], pred_resized
        )

        cv2.rectangle(seg_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(seg_img, f"Polyp: {conf:.0%}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Apply mask overlay
    seg_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    overlay = seg_rgb.copy()
    mask_indices = final_mask == 1
    overlay[mask_indices] = [0, 255, 0]
    seg_rgb = cv2.addWeighted(seg_rgb, 1 - mask_opacity, overlay, mask_opacity, 0)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_rgb, contours, -1, (255, 255, 0), 2)

    elapsed = time.time() - start_time
    mask_pixels = np.sum(final_mask)
    coverage = (mask_pixels / (h * w)) * 100

    det_b64 = numpy_to_base64(detection_rgb)
    seg_b64 = numpy_to_base64(seg_rgb)

    return det_b64, seg_b64, {
        "polyps_found": len(detections),
        "processing_time": round(elapsed, 2),
        "device": DEVICE,
        "deep_scan": deep_scan,
        "mask_coverage": round(coverage, 2),
        "detections": detections
    }


def process_video_pipeline(video_path, conf_threshold=0.25, iou_threshold=0.45, mask_opacity=0.4, skip_frames=0, deep_scan=False, alert_cooldown=3.0):
    """Process a video through the detection + segmentation pipeline."""
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, {"error": "Could not open video"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup writer
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "processed.mp4")
    output_fps = fps / (skip_frames + 1) if skip_frames > 0 else fps
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    # Fallback to mp4v if avc1 is not available
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    all_detections = []
    frame_count = 0
    processed_count = 0
    preview_frame = None

    # Alert tracking
    alert_timestamps = []
    last_alert_time = -alert_cooldown  # Allow first detection to trigger immediately

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            h, w = frame.shape[:2]

            # Detection (standard or SAHI deep scan)
            if deep_scan:
                sahi_dets = run_sahi_detection(frame, conf_threshold, iou_threshold)
                detections = [(d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]) for d in sahi_dets]
            else:
                results = yolo_model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].item()
                        detections.append((x1, y1, x2, y2, conf))

            output_frame = frame.copy()
            final_mask = np.zeros((h, w), dtype=np.uint8)

            if detections:
                for (x1, y1, x2, y2, conf) in detections:
                    pad = 10
                    x1p, y1p = max(0, x1-pad), max(0, y1-pad)
                    x2p, y2p = min(w, x2+pad), min(h, y2+pad)
                    crop = frame[y1p:y2p, x1p:x2p]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    augmented = transform(image=crop_rgb)
                    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logits = unet_model(input_tensor)
                        pred = torch.sigmoid(logits) > 0.5
                        pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)

                    ch, cw = crop.shape[:2]
                    pred_resized = cv2.resize(pred_np, (cw, ch), interpolation=cv2.INTER_NEAREST)
                    final_mask[y1p:y2p, x1p:x2p] = cv2.bitwise_or(
                        final_mask[y1p:y2p, x1p:x2p], pred_resized
                    )
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                mi = final_mask == 1
                if np.any(mi):
                    output_frame[mi] = (
                        output_frame[mi] * (1 - mask_opacity) +
                        np.array([0, 255, 0]) * mask_opacity
                    ).astype(np.uint8)
                    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(output_frame, contours, -1, (0, 255, 255), 2)

            writer.write(output_frame)
            all_detections.append(detections)
            processed_count += 1

            # Voice alert: track detection events with cooldown
            if detections:
                effective_fps = fps / (skip_frames + 1) if skip_frames > 0 else fps
                current_time = processed_count / effective_fps if effective_fps > 0 else 0
                if current_time - last_alert_time >= alert_cooldown:
                    confidences = [d[4] for d in detections]
                    alert_timestamps.append({
                        "time_sec": round(current_time, 2),
                        "polyp_count": len(detections),
                        "max_confidence": round(max(confidences) * 100, 1)
                    })
                    last_alert_time = current_time

            # Capture a preview frame from the middle
            if processed_count == max(1, total_frames // (2 * (skip_frames + 1))):
                preview_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            frame_count += 1

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start_time
    frames_with_polyps = sum(1 for d in all_detections if d)
    all_confidences = [det[4] for frame_dets in all_detections for det in frame_dets]

    preview_b64 = numpy_to_base64(preview_frame) if preview_frame is not None else None

    stats = {
        "total_frames": processed_count,
        "frames_with_polyps": frames_with_polyps,
        "detection_rate": round(frames_with_polyps / processed_count * 100, 1) if processed_count else 0,
        "processing_time": round(elapsed, 1),
        "fps_achieved": round(processed_count / elapsed, 1) if elapsed > 0 else 0,
        "avg_confidence": round(float(np.mean(all_confidences) * 100), 1) if all_confidences else 0,
        "max_confidence": round(float(max(all_confidences) * 100), 1) if all_confidences else 0,
        "device": DEVICE,
        "deep_scan": deep_scan,
        "alert_timestamps": alert_timestamps,
        "alert_count": len(alert_timestamps),
    }

    return output_path, preview_b64, stats


def numpy_to_base64(img_rgb):
    """Convert numpy RGB array to base64 JPEG string."""
    if img_rgb is None:
        return None
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


# ==========================================
# ROUTES — Static Website
# ==========================================
@app.route('/')
def index():
    return send_from_directory(WEBSITE_DIR, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(WEBSITE_DIR, path)


# ==========================================
# API — Image Analysis
# ==========================================
@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        image_bytes = file.read()

        conf = float(request.form.get('confidence', 0.25))
        iou = float(request.form.get('iou', 0.45))
        opacity = float(request.form.get('opacity', 0.4))
        deep_scan = request.form.get('deep_scan', 'false').lower() == 'true'

        det_b64, seg_b64, stats = process_image_pipeline(image_bytes, conf, iou, opacity, deep_scan=deep_scan)

        return jsonify({
            "detection_image": det_b64,
            "segmentation_image": seg_b64,
            "stats": stats
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==========================================
# API — Video Analysis
# ==========================================
@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files['video']
        # Save temp video file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, file.filename or "input.mp4")
        file.save(video_path)

        conf = float(request.form.get('confidence', 0.25))
        iou = float(request.form.get('iou', 0.45))
        opacity = float(request.form.get('opacity', 0.4))
        skip = int(request.form.get('skip_frames', 0))
        deep_scan = request.form.get('deep_scan', 'false').lower() == 'true'
        alert_cooldown = float(request.form.get('alert_cooldown', 3.0))

        output_path, preview_b64, stats = process_video_pipeline(video_path, conf, iou, opacity, skip, deep_scan=deep_scan, alert_cooldown=alert_cooldown)

        # Store processed video for serving
        video_id = str(uuid.uuid4())
        processed_videos[video_id] = output_path

        # Cleanup old videos (keep last N)
        if len(processed_videos) > MAX_STORED_VIDEOS:
            oldest_id = next(iter(processed_videos))
            old_path = processed_videos.pop(oldest_id)
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
            except OSError:
                pass

        return jsonify({
            "video_ready": True,
            "video_id": video_id,
            "preview_image": preview_b64,
            "stats": stats
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==========================================
# API — Serve Processed Video
# ==========================================
@app.route('/api/video/<video_id>', methods=['GET'])
def serve_video(video_id):
    """Serve a processed video file by its ID."""
    if video_id not in processed_videos:
        return jsonify({"error": "Video not found or expired"}), 404

    video_path = processed_videos[video_id]
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file no longer available"}), 404

    return send_file(video_path, mimetype='video/mp4')


# ==========================================
# API — Health Check
# ==========================================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "yolo_loaded": yolo_model is not None,
        "unet_loaded": unet_model is not None
    })


# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("🩺 DetSEG — AI-Powered Polyp Detection & Segmentation")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Website: {WEBSITE_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nLoading models...")
    try:
        load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    print(f"\nStarting server at http://localhost:5000")
    print("   Press Ctrl+C to stop.\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
