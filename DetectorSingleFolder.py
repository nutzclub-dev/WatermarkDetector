import os
import shutil
from ultralytics import YOLO

# Try to load a trained model with fallbacks
model = None
model_type = None

# Attempt 1: best.pt
try:
    if os.path.exists('best.pt'):
        model = YOLO('best.pt')
        model_type = 'yolo'
        print("Loaded best.pt")
except Exception as e:
    print(f"Failed to load best.pt: {e}")

# Attempt 2: yolov8n.pt
if model is None:
    try:
        model = YOLO('yolov8n.pt')
        model_type = 'yolo'
        print("Loaded yolov8n.pt")
    except Exception as e:
        print(f"Failed to load yolov8n.pt: {e}")

# Attempt 3: watermark_detection_model_V2_60000_Data_Set.h5 (Keras/VGG16)
if model is None:
    try:
        from tensorflow.keras.models import load_model
        model = load_model('watermark_detection_model_V2_60000_Data_Set.h5')
        model_type = 'keras'
        print("Loaded watermark_detection_model_V2_60000_Data_Set.h5")
    except Exception as e:
        print(f"Failed to load watermark_detection_model_V2_60000_Data_Set.h5: {e}")

if model is None:
    print("Warning: No model could be loaded. Ensure best.pt, yolov8n.pt, or the .h5 model exists.")

def predict_watermark(img_path):
    detections = []

    if model is None:
        return img_path, detections

    if model_type == 'yolo':
        results = model(img_path, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()
                c = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]

                detections.append({
                    'box': b,
                    'confidence': c,
                    'class_name': cls_name
                })
    elif model_type == 'keras':
        import cv2
        from tensorflow.keras.preprocessing import image
        import numpy as np

        frame = cv2.imread(img_path)
        if frame is not None:
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_array = image.img_to_array(rgb)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            pred = model.predict(img_array, verbose=0)
            conf = float(pred[0][0])
            if conf > 0.3: # arbitrary threshold for detection
                detections.append({
                    'box': [0, 0, frame.shape[1], frame.shape[0]], # No box available
                    'confidence': conf,
                    'class_name': 'watermark'
                })

    return img_path, detections

def organize_images(image_paths, watermarked_dir, not_watermarked_dir, threshold=0.3):
    for img_path in image_paths:
        _, detections = predict_watermark(img_path)

        # Check if any detection has confidence above threshold
        is_watermarked = any(det['confidence'] > threshold for det in detections)

        if is_watermarked:
            # Move to watermarked folder
            shutil.move(img_path, os.path.join(watermarked_dir, os.path.basename(img_path)))
            print(f"Moved {img_path} to Watermarked (Detections: {len(detections)})")
        else:
            # Move to not watermarked folder
            shutil.move(img_path, os.path.join(not_watermarked_dir, os.path.basename(img_path)))
            print(f"Moved {img_path} to Not_Watermarked")

if __name__ == "__main__":
    # Use the current directory where the script is running
    image_dir = os.path.dirname(os.path.abspath(__file__))
    watermarked_dir = os.path.join(image_dir, "Watermarked")
    not_watermarked_dir = os.path.join(image_dir, "Not_Watermarked")

    if os.path.exists(image_dir):
        # Create directories if they don't exist
        os.makedirs(watermarked_dir, exist_ok=True)
        os.makedirs(not_watermarked_dir, exist_ok=True)

        # Get all images in the directory
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        # Organize images
        organize_images(image_paths, watermarked_dir, not_watermarked_dir)
    else:
        print(f"Directory {image_dir} does not exist.")
