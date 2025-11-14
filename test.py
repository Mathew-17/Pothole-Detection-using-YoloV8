from ultralytics import YOLO
from PIL import Image
import os
import csv
from pathlib import Path
import numpy as np

# =========================
# CONFIGURATION
# =========================
model_path = r"C:\Users\dions\Downloads\best (1).pt"
input_folder = r"C:\Users\dions\Downloads\real_img"
labels_folder = r"C:\Users\dions\Downloads\data\valid\labels"
output_folder = r"C:\Users\dions\Downloads\out_folder"

print("=" * 60)
print("YOLO BATCH PREDICTION SCRIPT WITH ACCURACY METRICS")
print("=" * 60)
print(f"Model: {model_path}")
print(f"Input folder: {input_folder}")
print(f"Labels folder: {labels_folder}")
print(f"Output folder: {output_folder}")
print("=" * 60)

# =========================
# LOAD MODEL
# =========================
try:
    model = YOLO(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

if not os.path.exists(input_folder):
    print(f"Input folder does not exist: {input_folder}")
    exit()

if not os.path.exists(labels_folder):
    print(f"Labels folder does not exist: {labels_folder}")
    calculate_metrics = False
else:
    calculate_metrics = True
    print("Ground truth labels found - advanced metrics will be calculated")

# =========================
# FUNCTIONS
# =========================
def load_ground_truth_labels(label_path, img_width, img_height):
    """Load YOLO-format labels and convert to absolute coordinates"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    labels.append({'class_id': class_id, 'bbox': [x1, y1, x2, y2]})
    return labels

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return intersection / (area1 + area2 - intersection) if (area1 + area2 - intersection) > 0 else 0.0

# =========================
# OUTPUT DIRECTORIES
# =========================
for d in [output_folder,
          os.path.join(output_folder, 'annotated_images'),
          os.path.join(output_folder, 'cropped_detections'),
          os.path.join(output_folder, 'detection_data')]:
    os.makedirs(d, exist_ok=True)
print("Output directories created")

# =========================
# COLLECT IMAGES (DEDUPED)
# =========================
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
image_files = [f for f in Path(input_folder).glob("*") if f.suffix.lower() in image_extensions]

# Deduplicate and sort
image_files = sorted(set(image_files))

if not image_files:
    print(f"No image files found in: {input_folder}")
    exit()

print(f"Found {len(image_files)} unique images to process")

# =========================
# RUN PREDICTIONS
# =========================
print("\nStarting batch prediction...")
print("-" * 40)
results = model.predict(input_folder, save=True, project=output_folder, name='yolo_output', conf=0.25)

if len(results) != len(image_files):
    print(f"⚠️ Warning: YOLO returned {len(results)} results but {len(image_files)} images were found.")
    print("   Processing will continue with the available pairs.")

# =========================
# METRICS INIT
# =========================
total_detections = 0
images_with_detections = 0
detection_data = []

true_positives = false_positives = false_negatives = true_negatives = 0
images_with_ground_truth = ground_truth_objects = correctly_detected_objects = 0
iou_threshold = 0.5

# =========================
# PROCESS EACH IMAGE
# =========================
for i, (result, image_file) in enumerate(zip(results, image_files)):
    image_name = image_file.name
    print(f"Processing {i+1}/{len(image_files)}: {image_name}")

    try:
        original_image = Image.open(image_file)
        img_width, img_height = original_image.size
    except Exception as e:
        print(f"  Error loading image: {e}")
        continue

    # Ground truth
    ground_truth_labels = []
    if calculate_metrics:
        gt_file = os.path.join(labels_folder, f"{image_file.stem}.txt")
        ground_truth_labels = load_ground_truth_labels(gt_file, img_width, img_height)
        if ground_truth_labels:
            images_with_ground_truth += 1
            ground_truth_objects += len(ground_truth_labels)

    # Predictions
    num_detections = len(result.boxes) if result.boxes is not None else 0
    total_detections += num_detections
    matched_gt, matched_pred = set(), set()

    if num_detections > 0:
        images_with_detections += 1
        print(f"  Found {num_detections} detection(s)")
        annotated_image = result.plot()
        Image.fromarray(annotated_image).save(os.path.join(output_folder, 'annotated_images', f'detected_{image_name}'))

        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            confidence, class_id = float(box.conf[0]), int(box.cls[0])
            class_name = model.names[class_id]
            pred_bbox, is_tp = [x1, y1, x2, y2], False

            if calculate_metrics and ground_truth_labels:
                for gt_idx, gt in enumerate(ground_truth_labels):
                    if gt["class_id"] == class_id:
                        if calculate_iou(pred_bbox, gt["bbox"]) >= iou_threshold and gt_idx not in matched_gt:
                            matched_gt.add(gt_idx)
                            matched_pred.add(j)
                            is_tp = True
                            break

            detection_data.append({
                'image': image_name,
                'detection_id': j+1,
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'is_true_positive': is_tp
            })

            # Save crop
            try:
                status = "TP" if is_tp else "FP"
                crop_filename = f"{image_file.stem}_det{j+1}_{class_name}_{confidence:.2f}_{status}.jpg"
                original_image.crop((x1, y1, x2, y2)).save(os.path.join(output_folder, 'cropped_detections', crop_filename))
            except Exception as e:
                print(f"    Error cropping detection {j+1}: {e}")
    else:
        print("  No detections found")

    # Metrics update
    if calculate_metrics:
        tp, fp, fn = len(matched_pred), num_detections - len(matched_pred), len(ground_truth_labels) - len(matched_gt)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        if not ground_truth_labels and num_detections == 0:
            true_negatives += 1
        correctly_detected_objects += len(matched_gt)
        if ground_truth_labels or num_detections > 0:
            print(f"    Ground truth: {len(ground_truth_labels)}, TP: {tp}, FP: {fp}, FN: {fn}")

# =========================
# SAVE CSV
# =========================
csv_path = os.path.join(output_folder, 'detection_data', 'detections.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    if detection_data:
        writer = csv.DictWriter(csvfile, fieldnames=['image','detection_id','class','confidence','bbox_x1','bbox_y1','bbox_x2','bbox_y2','is_true_positive'])
        writer.writeheader()
        for d in detection_data:
            writer.writerow({
                'image': d['image'],
                'detection_id': d['detection_id'],
                'class': d['class'],
                'confidence': d['confidence'],
                'bbox_x1': d['bbox'][0],
                'bbox_y1': d['bbox'][1],
                'bbox_x2': d['bbox'][2],
                'bbox_y2': d['bbox'][3],
                'is_true_positive': d['is_true_positive']
            })

# =========================
# SUMMARY REPORT
# =========================
summary_path = os.path.join(output_folder, 'summary_report.txt')
with open(summary_path, 'w') as f:
    f.write("POTHOLE DETECTION SUMMARY\n" + "="*50 + "\n")
    f.write(f"Images processed: {len(image_files)}\n")
    f.write(f"Total detections: {total_detections}\n")
    f.write(f"Images with detections: {images_with_detections}\n")
    f.write(f"Detection rate: {(images_with_detections/len(image_files))*100:.1f}%\n\n")
    if calculate_metrics:
        precision = true_positives / (true_positives+false_positives) if (true_positives+false_positives)>0 else 0
        recall = true_positives / (true_positives+false_negatives) if (true_positives+false_negatives)>0 else 0
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
        f.write("METRICS:\n")
        f.write(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives}\n")
        f.write(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")

# =========================
# FINAL PRINT
# =========================
print("\n" + "="*60)
print("BATCH PREDICTION COMPLETED!")
print("="*60)
print(f"Processed: {len(image_files)} unique images")
print(f"Detections: {total_detections}")
print(f"Results saved to: {output_folder}")
print(f"Summary: {summary_path}")
print(f"CSV: {csv_path}")
print("="*60)
