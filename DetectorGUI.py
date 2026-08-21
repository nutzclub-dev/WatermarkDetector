import os
import sys

# 1. Prevent OpenMP collisions between PyTorch, OpenCV, and PyQt5
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Register PyTorch DLL directory for Python 3.12+ on Windows
torch_lib_path = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)

# 3. Import Torch & YOLO FIRST before any other C-extensions
import torch
from ultralytics import YOLO

# 4. Math and Computer Vision
import numpy as np
import cv2

# 5. TensorFlow / Keras (optional)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
except ImportError:
    pass

# 6. PyQt5 GUI components LAST
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, 
    QPushButton, QProgressBar, QHBoxLayout, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

def get_location_string(box, img_width, img_height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # 3x3 grid calculation
    col = "LEFT" if cx < img_width / 3 else "RIGHT" if cx > 2 * img_width / 3 else "CENTER"
    row = "TOP" if cy < img_height / 3 else "BOTTOM" if cy > 2 * img_height / 3 else "CENTER"
    
    if col == "CENTER" and row == "CENTER":
        return "CENTER"
    elif row == "CENTER":
        return f"CENTER {col}"
    elif col == "CENTER":
        return f"{row} CENTER"
    else:
        return f"{row} {col}"

class DragDropWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drag & Drop image/video here\n― or click to browse ―")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 8px;
                background-color: #f9f9f9;
                font-size: 15px;
                color: #555;
                padding: 20px;
            }
            QLabel:hover {
                background-color: #f0f0f0;
                border-color: #2196F3;
            }
        """)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 180)
        self.file_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = None
        # Handle standard file URLs
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
        # Handle plain text paths sometimes sent by Windows Explorer
        elif event.mimeData().hasText():
            file_path = event.mimeData().text().strip().replace("file:///", "")

        if file_path and os.path.exists(file_path):
            self.set_loaded_file(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image or Video",
                "",
                "Media Files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
            )
            if file_path:
                self.set_loaded_file(file_path)

    def set_loaded_file(self, path):
        self.file_path = path
        self.setText(f"✓ File Loaded:\n{os.path.basename(path)}")
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                background-color: #e8f5e9;
                font-size: 15px;
                font-weight: bold;
                color: #2E7D32;
                padding: 20px;
            }
        """)

class WatermarkDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Ensures the main window doesn't block drop events for the child widget
        self.setAcceptDrops(True)
        
        # Load watermark model (checks for local best.pt -> Hugging Face -> yolov8n.pt fallback)
        if os.path.exists("best.pt"):
            self.yolo_model = YOLO("best.pt")
        else:
            try:
                self.yolo_model = YOLO("hf://qfisch/yolov8n-watermark-detection")
            except Exception:
                self.yolo_model = YOLO("yolov8n.pt")
        
        try:
            self.vgg_model = load_model(r"watermark_detection_model_V2_60000_Data_Set.h5")
        except Exception:
            self.vgg_model = None
            
        self.initUI()

    def initUI(self):
        self.setWindowTitle('WATERMARK DETECTOR')
        self.setGeometry(100, 100, 500, 520)
        
        layout = QVBoxLayout()
        
        title = QLabel("WATERMARK DETECTOR")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Bold))
        layout.addWidget(title)
        
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_layout.addWidget(model_label)
        self.model_selector = QComboBox()
        self.model_selector.addItems(["YOLOv8", "VGG16", "Both"])
        model_layout.addWidget(self.model_selector)
        layout.addLayout(model_layout)
        
        self.drop_zone = DragDropWidget()
        layout.addWidget(self.drop_zone)
        
        self.scan_button = QPushButton("Scan")
        self.scan_button.setFont(QFont("Arial", 14))
        self.scan_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.scan_button.clicked.connect(self.scan_file)
        layout.addWidget(self.scan_button)
        
        # Results Section
        self.result_label = QLabel("Result: PENDING")
        self.result_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)
        
        # Confidence layout
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence")
        conf_label.setFixedWidth(80)
        conf_layout.addWidget(conf_label)
        
        self.conf_bar = QProgressBar()
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(True)
        conf_layout.addWidget(self.conf_bar)
        layout.addLayout(conf_layout)
        
        # Location layout
        loc_layout = QHBoxLayout()
        loc_label = QLabel("Location")
        loc_label.setFixedWidth(80)
        loc_layout.addWidget(loc_label)
        
        self.loc_value = QLabel("-")
        loc_layout.addWidget(self.loc_value)
        layout.addLayout(loc_layout)
        
        # Frames layout
        frames_layout = QHBoxLayout()
        frames_label = QLabel("Frames")
        frames_label.setFixedWidth(80)
        frames_layout.addWidget(frames_label)
        
        self.frames_value = QLabel("- / -")
        frames_layout.addWidget(self.frames_value)
        layout.addLayout(frames_layout)
        
        self.setLayout(layout)
        
    def scan_file(self):
        if not self.drop_zone.file_path:
            self.result_label.setText("NO FILE LOADED")
            return
            
        self.result_label.setText("SCANNING...")
        QApplication.processEvents()
        
        file_path = self.drop_zone.file_path
        model_choice = self.model_selector.currentText()
        
        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        max_conf = 0.0
        best_loc = "N/A"
        total_frames = 1
        watermarked_frames = 0
        
        if not is_video:
            conf, loc, detected = self.process_image(file_path, model_choice)
            max_conf = conf
            best_loc = loc
            watermarked_frames = 1 if detected else 0
        else:
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                total_frames = 1
                
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % 30 == 0:
                    conf, loc, detected = self.process_frame(frame, model_choice)
                    if detected:
                        watermarked_frames += 1
                        if conf > max_conf:
                            max_conf = conf
                            best_loc = loc
                frame_count += 1
            cap.release()
            total_frames = frame_count
            
        # Update UI results
        if max_conf > 0.25 or watermarked_frames > 0:
            self.result_label.setText("WATERMARK DETECTED")
            self.result_label.setStyleSheet("color: red;")
        else:
            self.result_label.setText("NO WATERMARK DETECTED")
            self.result_label.setStyleSheet("color: green;")
            
        self.conf_bar.setValue(int(max_conf * 100))
        self.loc_value.setText(best_loc)
        
        if is_video:
            self.frames_value.setText(f"{watermarked_frames} / {total_frames}")
        else:
            self.frames_value.setText("1 / 1")
            
    def process_image(self, file_path, model_choice):
        frame = cv2.imread(file_path)
        if frame is None:
            return 0.0, "N/A", False
        return self.process_frame(frame, model_choice)
        
    def process_frame(self, frame, model_choice):
        yolo_conf = 0.0
        yolo_loc = "N/A"
        yolo_detected = False
        
        vgg_conf = 0.0
        vgg_detected = False
        
        if model_choice in ["YOLOv8", "Both"]:
            results = self.yolo_model(frame, verbose=False)
            img_height, img_width = frame.shape[:2]
            
            for r in results:
                for box in r.boxes:
                    conf = box.conf[0].item()
                    if conf > yolo_conf:
                        yolo_conf = conf
                        b = box.xyxy[0].tolist()
                        yolo_loc = get_location_string(b, img_width, img_height)
                        yolo_detected = True
                        
        if model_choice in ["VGG16", "Both"] and self.vgg_model is not None:
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_array = image.img_to_array(rgb)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            pred = self.vgg_model.predict(img_array, verbose=0)
            vgg_conf = float(pred[0][0])
            vgg_detected = vgg_conf > 0.5
            
        if model_choice == "YOLOv8":
            return yolo_conf, yolo_loc, yolo_detected
        elif model_choice == "VGG16":
            return vgg_conf, "N/A", vgg_detected
        else:
            avg_conf = (yolo_conf + vgg_conf) / 2
            detected = yolo_detected or vgg_detected
            return avg_conf, yolo_loc, detected

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WatermarkDetectorApp()
    ex.show()
    sys.exit(app.exec_())