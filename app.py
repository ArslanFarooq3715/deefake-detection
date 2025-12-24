"""
Flask API for Realtime Deepfake Detection
Uses inference logic from inference-book.py
"""
import os
import sys
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import timm
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import base64
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MODEL_PATH = 'best_deepfake_model_optimized-resnet101.pth'
INPUT_SIZE = 384
MAX_FRAMES_TO_ANALYZE = 40  # Analyze 40 frames for better accuracy
MAX_FRAMES_TO_SHOW = 10  # Show only 10 frames on frontend for user engagement

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size


# ============================================================================
# CONFIGURATION (from inference-book.py)
# ============================================================================

class InferenceConfig:
    """Configuration for inference"""
    
    # Model paths
    BEST_MODEL_PATH = MODEL_PATH
    
    # Model settings (must match training config)
    INPUT_SIZE = 384
    NUM_CLASSES = 1
    DROPOUT_RATES = [0.4, 0.2, 0.1]
    
    # Inference settings
    FRAMES_TO_SAMPLE = MAX_FRAMES_TO_ANALYZE
    BATCH_SIZE = 8  # Batch size for frame processing
    CONFIDENCE_THRESHOLD = 0.5
    
    # Aggregation methods
    AGGREGATION_METHOD = 'weighted_avg'  # Options: 'mean', 'max', 'weighted_avg', 'voting'
    
    # Device
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    DEVICE = None
    
    def __init__(self):
        self.DEVICE = self.get_device()
        self.BEST_MODEL_PATH = MODEL_PATH
        self.FRAMES_TO_SAMPLE = MAX_FRAMES_TO_ANALYZE


# ============================================================================
# MODEL ARCHITECTURES (from inference-book.py)
# ============================================================================

class DeepfakeDetectorBase(nn.Module):
    """Base class for deepfake detectors"""
    def __init__(self, config: InferenceConfig):
        super(DeepfakeDetectorBase, self).__init__()
        self.config = config
    
    def forward(self, x):
        raise NotImplementedError


class ResNetDetector(DeepfakeDetectorBase):
    """ResNet-based detector"""
    def __init__(self, config: InferenceConfig):
        super(ResNetDetector, self).__init__(config)
        
        self.backbone = timm.create_model(
            'resnet101', 
            pretrained=False,
            num_classes=0, 
            global_pool='avg'
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.INPUT_SIZE, config.INPUT_SIZE)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATES[0]),
            
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATES[1]),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATES[2]),
            
            nn.Linear(128, config.NUM_CLASSES)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze(1)


# ============================================================================
# FACE DETECTION (from inference-book.py)
# ============================================================================

class FaceDetector:
    """Face detection using Haar Cascade"""
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and extract the largest face from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        face = frame[y:y+h, x:x+w]
        return face


# ============================================================================
# INFERENCE ENGINE (from inference-book.py)
# ============================================================================

class DeepfakeInference:
    """Inference engine for deepfake detection"""
    
    def __init__(self, model_path: str, config: InferenceConfig):
        self.config = config
        self.face_detector = FaceDetector()
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"✅ Model loaded successfully!")
        print(f"✅ Device: {self.config.DEVICE}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        
        # Determine model architecture from checkpoint
        model_arch = checkpoint.get('config', {}).get('MODEL_ARCHITECTURE', 'resnet101')
        print(f"Model architecture: {model_arch}")
        
        # Create model (defaulting to ResNet101)
        model = ResNetDetector(self.config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.config.DEVICE)
        
        return model
    
    def _extract_frames(self, video_path: str) -> List[Tuple[np.ndarray, int]]:
        """Extract frames from video with frame indices"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.config.FRAMES_TO_SAMPLE, dtype=int)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_indices:
                # Detect face
                face = self.face_detector.detect_face(frame)
                if face is not None:
                    # Resize to model input size
                    face_resized = cv2.resize(face, (self.config.INPUT_SIZE, self.config.INPUT_SIZE))
                    frames.append((face_resized, frame_idx))
            
            frame_idx += 1
            
            if len(frames) >= self.config.FRAMES_TO_SAMPLE:
                break
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No faces detected in video: {video_path}")
        
        return frames
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for model input"""
        processed_frames = []
        
        for frame in frames:
            # Apply transformations
            augmented = self.transform(image=frame)
            tensor = augmented['image']
            processed_frames.append(tensor)
        
        # Stack into batch
        batch = torch.stack(processed_frames)
        return batch
    
    def _aggregate_predictions(self, probabilities: np.ndarray) -> Tuple[float, str]:
        """Aggregate frame predictions into final video prediction"""
        if self.config.AGGREGATION_METHOD == 'mean':
            final_prob = np.mean(probabilities)
        
        elif self.config.AGGREGATION_METHOD == 'max':
            final_prob = np.max(probabilities)
        
        elif self.config.AGGREGATION_METHOD == 'weighted_avg':
            # Weight by confidence (distance from 0.5)
            confidence = np.abs(probabilities - 0.5)
            if confidence.sum() > 0:
                weights = confidence / confidence.sum()
                final_prob = (probabilities * weights).sum()
            else:
                final_prob = np.mean(probabilities)
        
        elif self.config.AGGREGATION_METHOD == 'voting':
            # Majority voting
            predictions = (probabilities >= self.config.CONFIDENCE_THRESHOLD).astype(int)
            final_prob = predictions.mean()
        
        else:
            final_prob = np.mean(probabilities)
        
        # Determine label
        if final_prob >= self.config.CONFIDENCE_THRESHOLD:
            label = "FAKE"
        else:
            label = "REAL"
        
        return final_prob, label
    
    def predict_video_for_api(self, video_path: str) -> Dict:
        """Predict video and return results formatted for Flask API"""
        try:
            # Extract frames with indices
            frames_data = self._extract_frames(video_path)
            frames = [frame for frame, _ in frames_data]
            frame_indices = [idx for _, idx in frames_data]
            
            # Preprocess
            batch = self._preprocess_frames(frames)
            batch = batch.to(self.config.DEVICE)
            
            # Predict in batches
            all_probabilities = []
            
            with torch.no_grad():
                for i in range(0, len(batch), self.config.BATCH_SIZE):
                    batch_slice = batch[i:i + self.config.BATCH_SIZE]
                    outputs = self.model(batch_slice)
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                    all_probabilities.extend(probabilities)
            
            all_probabilities = np.array(all_probabilities)
            
            # Get individual frame predictions
            frame_results = []
            fake_count = 0
            real_count = 0
            
            for prob, frame_idx, frame in zip(all_probabilities, frame_indices, frames):
                is_fake = prob >= self.config.CONFIDENCE_THRESHOLD
                label = "FAKE" if is_fake else "REAL"
                confidence = prob if is_fake else (1 - prob)
                
                if label == "FAKE":
                    fake_count += 1
                else:
                    real_count += 1
                
                frame_results.append({
                    'frame_index': int(frame_idx),
                    'label': label,
                    'confidence': round(float(confidence) * 100, 2),
                    'probability': float(prob),
                    'frame': frame
                })
            
            # Aggregate predictions for overall result
            final_probability, overall_label = self._aggregate_predictions(all_probabilities)
            overall_confidence = abs(final_probability - 0.5) * 2 * 100
            
            return {
                'success': True,
                'overall_result': overall_label,
                'overall_confidence': round(overall_confidence, 2),
                'fake_frames': fake_count,
                'real_frames': real_count,
                'total_frames_analyzed': len(frame_results),
                'frame_results': frame_results
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Initialize inference engine
config = InferenceConfig()
inference_engine = DeepfakeInference(MODEL_PATH, config)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{frame_base64}"


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and process"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Use inference engine to analyze video
        print(f"Analyzing video: {filename}...")
        result = inference_engine.predict_video_for_api(filepath)
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Unknown error occurred')}), 400
        
        # Select evenly spaced frames for display (10 frames)
        all_frame_results = result['frame_results']
        if len(all_frame_results) > MAX_FRAMES_TO_SHOW:
            display_indices = np.linspace(0, len(all_frame_results) - 1, MAX_FRAMES_TO_SHOW, dtype=int)
            display_results = [all_frame_results[i] for i in display_indices]
        else:
            display_results = all_frame_results
        
        # Convert selected frames to base64 for frontend display
        frames_for_display = []
        for frame_result in display_results:
            frame_base64 = frame_to_base64(frame_result['frame'])
            frames_for_display.append({
                'frame_index': frame_result['frame_index'],
                'label': frame_result['label'],
                'confidence': frame_result['confidence'],
                'image': frame_base64
            })
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'overall_result': result['overall_result'],
            'overall_confidence': result['overall_confidence'],
            'fake_frames': result['fake_frames'],
            'real_frames': result['real_frames'],
            'total_frames_analyzed': result['total_frames_analyzed'],
            'frames_displayed': len(frames_for_display),
            'frames': frames_for_display
        })
    
    except Exception as e:
        # Clean up on error
        try:
            if 'filepath' in locals():
                os.remove(filepath)
        except:
            pass
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Realtime Deepfake Detection API")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {config.DEVICE}")
    print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Frames to Analyze: {MAX_FRAMES_TO_ANALYZE}")
    print(f"Frames to Display: {MAX_FRAMES_TO_SHOW}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
