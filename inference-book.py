
"""
Deepfake Detection Model Inference Script
Supports single video inference and batch inference
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm

# ============================================================================
# CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Configuration for inference"""
    
    # Model paths
    BEST_MODEL_PATH = "best_deepfake_model_optimized-resnet101.pth"
    
    # Model settings (must match training config)
    INPUT_SIZE = 384
    NUM_CLASSES = 1
    DROPOUT_RATES = [0.4, 0.2, 0.1]
    
    # Inference settings
    FRAMES_TO_SAMPLE = 40  # Number of frames to analyze per video
    BATCH_SIZE = 8  # Batch size for frame processing
    CONFIDENCE_THRESHOLD = 0.6
    
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
        print(f"Using device: {self.DEVICE}")


# ============================================================================
# ATTENTION MECHANISMS (Same as training)
# ============================================================================

class TemporalAttention(nn.Module):
    """Attention mechanism for frame aggregation"""
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        aggregated = weighted_features.sum(dim=1) / attention_weights.sum(dim=1)
        return aggregated, attention_weights.squeeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on important regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


# ============================================================================
# MODEL ARCHITECTURES (Same as training)
# ============================================================================

class DeepfakeDetectorBase(nn.Module):
    """Base class for deepfake detectors"""
    def __init__(self, config: InferenceConfig):
        super(DeepfakeDetectorBase, self).__init__()
        self.config = config
    
    def forward(self, x):
        raise NotImplementedError


class EfficientNetDetector(DeepfakeDetectorBase):
    """EfficientNet-based detector"""
    def __init__(self, config: InferenceConfig):
        super(EfficientNetDetector, self).__init__(config)
        
        self.backbone = timm.create_model(
            'efficientnet_b4', 
            pretrained=False,  # We'll load weights
            num_classes=0, 
            global_pool=''
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.INPUT_SIZE, config.INPUT_SIZE)
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                self.feature_dim = features.shape[1] * features.shape[2] * features.shape[3]
            else:
                self.feature_dim = features.shape[1]
        
        self.spatial_attention = SpatialAttention(1792)
        
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
        
        if len(features.shape) == 4:
            features = self.spatial_attention(features)
            features = features.view(features.size(0), -1)
        
        output = self.classifier(features)
        return output.squeeze(1)


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


class VisionTransformerDetector(DeepfakeDetectorBase):
    """Vision Transformer-based detector"""
    def __init__(self, config: InferenceConfig):
        super(VisionTransformerDetector, self).__init__(config)
        
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool='token'
        )
        
        import torchvision.transforms as transforms
        self.resize = transforms.Resize((224, 224), antialias=True)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATES[0]),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATES[1]),
            
            nn.Linear(512, config.NUM_CLASSES)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resize(x)
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze(1)


class EnsembleDetector(nn.Module):
    """Ensemble of multiple models"""
    def __init__(self, config: InferenceConfig):
        super(EnsembleDetector, self).__init__()
        self.models = nn.ModuleList([
            EfficientNetDetector(config),
            ResNetDetector(config),
            VisionTransformerDetector(config)
        ])
        self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        outputs = torch.stack(outputs, dim=0)
        weights = F.softmax(self.weights, dim=0)
        ensemble_output = (outputs * weights.view(-1, 1)).sum(dim=0)
        return ensemble_output


# ============================================================================
# FACE DETECTION
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
# INFERENCE ENGINE
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
        
        # Create model
        if model_arch == 'efficientnet_b4':
            model = EfficientNetDetector(self.config)
        elif model_arch == 'resnet101':
            model = ResNetDetector(self.config)
        elif model_arch == 'vit_base':
            model = VisionTransformerDetector(self.config)
        elif model_arch == 'ensemble':
            model = EnsembleDetector(self.config)
        else:
            # Default to ResNet
            model = ResNetDetector(self.config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.config.DEVICE)
        
        return model
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video"""
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
                    frames.append(face_resized)
            
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
            weights = confidence / confidence.sum()
            final_prob = (probabilities * weights).sum()
        
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
    
    def predict_video(self, video_path: str, verbose: bool = True) -> Dict:
        """Predict whether a video is real or fake"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing video: {os.path.basename(video_path)}")
            print(f"{'='*60}")
        
        try:
            # Extract frames
            if verbose:
                print("Extracting frames and detecting faces...")
            frames = self._extract_frames(video_path)
            
            if verbose:
                print(f"Extracted {len(frames)} frames with faces")
            
            # Preprocess
            if verbose:
                print("Preprocessing frames...")
            batch = self._preprocess_frames(frames)
            batch = batch.to(self.config.DEVICE)
            
            # Predict
            if verbose:
                print("Running inference...")
            
            all_probabilities = []
            
            with torch.no_grad():
                # Process in batches
                for i in range(0, len(batch), self.config.BATCH_SIZE):
                    batch_slice = batch[i:i + self.config.BATCH_SIZE]
                    outputs = self.model(batch_slice)
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                    all_probabilities.extend(probabilities)
            
            all_probabilities = np.array(all_probabilities)
            
            # Aggregate predictions
            final_probability, label = self._aggregate_predictions(all_probabilities)
            confidence = abs(final_probability - 0.5) * 2 * 100  # Convert to 0-100%
            
            result = {
                'video_path': video_path,
                'prediction': label,
                'fake_probability': float(final_probability),
                'real_probability': float(1 - final_probability),
                'confidence': float(confidence),
                'num_frames_analyzed': len(frames),
                'frame_probabilities': all_probabilities.tolist()
            }
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"RESULT: {label}")
                print(f"{'='*60}")
                print(f"Fake Probability: {final_probability*100:.2f}%")
                print(f"Real Probability: {(1-final_probability)*100:.2f}%")
                print(f"Confidence: {confidence:.2f}%")
                print(f"Frames Analyzed: {len(frames)}")
                print(f"{'='*60}\n")
            
            return result
        
        except Exception as e:
            error_result = {
                'video_path': video_path,
                'error': str(e),
                'prediction': 'ERROR'
            }
            if verbose:
                print(f"❌ Error processing video: {e}")
            return error_result
    
    def predict_batch(self, video_paths: List[str], output_file: Optional[str] = None) -> List[Dict]:
        """Predict multiple videos"""
        print(f"\n{'='*60}")
        print(f"Batch Inference: {len(video_paths)} videos")
        print(f"{'='*60}\n")
        
        results = []
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.predict_video(video_path, verbose=False)
            results.append(result)
            
            # Print summary
            if 'error' not in result:
                print(f"{os.path.basename(video_path)}: {result['prediction']} "
                      f"(Confidence: {result['confidence']:.2f}%)")
            else:
                print(f"{os.path.basename(video_path)}: ERROR - {result['error']}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to: {output_file}")
        
        # Print summary statistics
        successful = [r for r in results if 'error' not in r]
        if successful:
            fake_count = sum(1 for r in successful if r['prediction'] == 'FAKE')
            real_count = sum(1 for r in successful if r['prediction'] == 'REAL')
            avg_confidence = np.mean([r['confidence'] for r in successful])
            
            print(f"\n{'='*60}")
            print("Summary Statistics")
            print(f"{'='*60}")
            print(f"Total Videos: {len(video_paths)}")
            print(f"Successfully Processed: {len(successful)}")
            print(f"Errors: {len(video_paths) - len(successful)}")
            print(f"Predicted FAKE: {fake_count}")
            print(f"Predicted REAL: {real_count}")
            print(f"Average Confidence: {avg_confidence:.2f}%")
            print(f"{'='*60}\n")
        
        return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--video', type=str, help='Path to single video file')
    parser.add_argument('--video_dir', type=str, help='Directory containing videos')
    parser.add_argument('--model', type=str, default='best_deepfake_model_optimized.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, help='Output JSON file for batch results')
    parser.add_argument('--frames', type=int, default=1000,
                       help='Number of frames to sample from video')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for frame processing')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for fake detection')
    parser.add_argument('--aggregation', type=str, default='weighted_avg',
                       choices=['mean', 'max', 'weighted_avg', 'voting'],
                       help='Method to aggregate frame predictions')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video and not args.video_dir:
        parser.error("Either --video or --video_dir must be specified")
    
    # Setup config
    config = InferenceConfig()
    config.BEST_MODEL_PATH = args.model
    config.FRAMES_TO_SAMPLE = args.frames
    config.BATCH_SIZE = args.batch_size
    config.CONFIDENCE_THRESHOLD = args.threshold
    config.AGGREGATION_METHOD = args.aggregation
    
    # Initialize inference engine
    inference = DeepfakeInference(args.model, config)
    
    # Run inference
    if args.video:
        # Single video inference
        result = inference.predict_video(args.video, verbose=True)
        
        # Save result if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Result saved to: {args.output}")
    
    elif args.video_dir:
        # Batch inference
        video_paths = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_paths.extend(list(Path(args.video_dir).glob(ext)))
        
        video_paths = [str(p) for p in video_paths if not os.path.basename(p).startswith("._")]
        
        if len(video_paths) == 0:
            print(f"❌ No videos found in directory: {args.video_dir}")
            return
        
        results = inference.predict_batch(video_paths, output_file=args.output)


if __name__ == '__main__':
    main()






    # python inference-book.py --video data\fake\01_11__meeting_serious__9OM3VE0Y.mp4 --model best_deepfake_model_optimized.pt

    # python inference.py --video_dir path/to/videos/ --model best_deepfake_model_optimized.pth --output results.json