"""
Optimized Deepfake Detection System
- Multiple model architectures (EfficientNet, ResNet, Vision Transformer)
- Ensemble support
- Temporal frame aggregation
- Attention mechanisms
- Focal Loss for better training
- Improved data augmentation
"""

import os
import sys
import argparse
import json
import csv
import hashlib
import pickle
import gc
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Optimized configuration for deepfake detection"""
    
    # Dataset
    REAL_VIDEOS_DIR = "data/real"
    FAKE_VIDEOS_DIR = "data/fake"
    NUM_REAL_VIDEOS = 200
    NUM_FAKE_VIDEOS = 200
    FRAMES_PER_VIDEO = 20  # Increased for better temporal modeling
    TRAIN_VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42
    
    # Model Architecture Options
    MODEL_ARCHITECTURE = 'resnet101'  # Options: 'efficientnet_b4', 'resnet101', 'vit_base', 'ensemble'
    USE_ENSEMBLE = False  # Set to True to use ensemble of models
    USE_TEMPORAL_ATTENTION = True  # Use attention for frame aggregation
    USE_MULTI_SCALE = True  # Use multi-scale features
    
    # Model
    INPUT_SIZE = 384  # Increased for better feature extraction
    NUM_CLASSES = 1
    DROPOUT_RATES = [0.4, 0.2, 0.1]  # Reduced dropout to address underfitting
    WEIGHT_DECAY = 5e-5  # Reduced weight decay
    LABEL_SMOOTHING = 0.05  # Reduced label smoothing
    
    # Training
    BATCH_SIZE = 16  # Reduced for larger input size
    NUM_EPOCHS = 100  # More epochs
    LEARNING_RATE = 0.0002  # Slightly higher learning rate
    NUM_WORKERS = 0 if sys.platform == 'win32' else 4
    USE_AMP = True
    GRADIENT_CLIP_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 15
    OVERFITTING_THRESHOLD = 0.15
    
    # Loss function
    USE_FOCAL_LOSS = True  # Use Focal Loss for better handling of hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = "best_deepfake_model_optimized.pth"
    FINAL_MODEL_PATH = "final_deepfake_model_optimized.pth"
    CHECKPOINT_INTERVAL = 5
    
    # Frame caching
    FRAMES_CACHE_DIR = "frames_cache"
    
    # Inference
    INFERENCE_BATCH_SIZE = 8
    CONFIDENCE_THRESHOLD = 0.5
    
    # Device
    @staticmethod
    def get_device():
        """Get the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ CUDA is available! Found {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    DEVICE = None
    
    def __init__(self):
        """Initialize configuration"""
        self.DEVICE = self.get_device()
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.FRAMES_CACHE_DIR, exist_ok=True)
        torch.manual_seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ Using GPU: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.2f} GB")
            print(f"✅ Model Architecture: {self.MODEL_ARCHITECTURE}")
            print(f"✅ Temporal Attention: {self.USE_TEMPORAL_ATTENTION}")
            print(f"✅ Focal Loss: {self.USE_FOCAL_LOSS}")


# ============================================================================
# IMPROVED DATA AUGMENTATION FOR DEEPFAKE DETECTION
# ============================================================================

def get_train_augmentation():
    """Optimized augmentation for deepfake detection"""
    return A.Compose([
        # Geometric - important for detecting manipulation artifacts
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.4),
        
        # Color - helps detect color inconsistencies
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        
        # Noise - simulates compression artifacts
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        
        # Compression artifacts
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),
        A.Downscale(scale_min=0.8, scale_max=0.95, p=0.3),
        
        # Cutout - helps focus on important regions
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.3),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_augmentation():
    """Minimal augmentation for validation"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# ATTENTION MECHANISMS
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
        # features: (batch, num_frames, feature_dim)
        attention_weights = self.attention(features)  # (batch, num_frames, 1)
        weighted_features = features * attention_weights
        aggregated = weighted_features.sum(dim=1) / attention_weights.sum(dim=1)
        return aggregated, attention_weights.squeeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on important regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        attention = torch.sigmoid(self.conv(x))
        return x * attention


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class DeepfakeDetectorBase(nn.Module):
    """Base class for deepfake detectors"""
    def __init__(self, config: Config):
        super(DeepfakeDetectorBase, self).__init__()
        self.config = config
        self.use_temporal = config.USE_TEMPORAL_ATTENTION and config.FRAMES_PER_VIDEO > 1
    
    def forward(self, x):
        raise NotImplementedError


class EfficientNetDetector(DeepfakeDetectorBase):
    """EfficientNet-based detector"""
    def __init__(self, config: Config):
        super(EfficientNetDetector, self).__init__(config)
        
        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            'efficientnet_b4', 
            pretrained=True, 
            num_classes=0, 
            global_pool=''
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.INPUT_SIZE, config.INPUT_SIZE)
            features = self.backbone(dummy)
            if len(features.shape) == 4:  # (B, C, H, W)
                self.feature_dim = features.shape[1] * features.shape[2] * features.shape[3]
            else:
                self.feature_dim = features.shape[1]
        
        # Spatial attention
        if config.USE_MULTI_SCALE:
            self.spatial_attention = SpatialAttention(1792)  # EfficientNet-B4 channels
        
        # Classifier
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
        # x: (batch, channels, height, width)
        features = self.backbone(x)
        
        # Apply spatial attention if enabled
        if self.config.USE_MULTI_SCALE and len(features.shape) == 4:
            features = self.spatial_attention(features)
        
        # Flatten
        if len(features.shape) == 4:
            features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output.squeeze(1)


class ResNetDetector(DeepfakeDetectorBase):
    """ResNet-based detector"""
    def __init__(self, config: Config):
        super(ResNetDetector, self).__init__(config)
        
        self.backbone = timm.create_model(
            'resnet101', 
            pretrained=True, 
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
    def __init__(self, config: Config):
        super(VisionTransformerDetector, self).__init__(config)
        
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0,
            global_pool='token'
        )
        
        # Add resize layer for ViT
        self.resize = transforms.Resize((224, 224), antialias=True)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)  # Use 224 for dummy
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
        # Resize input from 384 to 224 for ViT
        x = self.resize(x)
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze(1)


class EnsembleDetector(nn.Module):
    """Ensemble of multiple models"""
    def __init__(self, config: Config):
        super(EnsembleDetector, self).__init__()
        self.models = nn.ModuleList([
            EfficientNetDetector(config),
            ResNetDetector(config),
            VisionTransformerDetector(config)
        ])
        self.weights = nn.Parameter(torch.ones(3) / 3)  # Learnable weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        outputs = torch.stack(outputs, dim=0)  # (num_models, batch)
        weights = F.softmax(self.weights, dim=0)
        ensemble_output = (outputs * weights.view(-1, 1)).sum(dim=0)
        return ensemble_output


def create_model(config: Config) -> nn.Module:
    """Factory function to create model based on config"""
    if config.MODEL_ARCHITECTURE == 'efficientnet_b4':
        return EfficientNetDetector(config)
    elif config.MODEL_ARCHITECTURE == 'resnet101':
        return ResNetDetector(config)
    elif config.MODEL_ARCHITECTURE == 'vit_base':
        return VisionTransformerDetector(config)
    elif config.MODEL_ARCHITECTURE == 'ensemble':
        return EnsembleDetector(config)
    else:
        raise ValueError(f"Unknown model architecture: {config.MODEL_ARCHITECTURE}")


# ============================================================================
# FRAME CACHING AND FACE DETECTION (Same as original)
# ============================================================================

def get_cache_path(video_path: str, config: Config) -> str:
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    cache_filename = f"{video_hash}_{config.FRAMES_PER_VIDEO}_{config.INPUT_SIZE}.pkl"
    return os.path.join(config.FRAMES_CACHE_DIR, cache_filename)

def load_frames_from_cache(video_path: str, config: Config) -> Optional[List[np.ndarray]]:
    cache_path = get_cache_path(video_path, config)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache for {video_path}: {e}")
            return None
    return None

def save_frames_to_cache(video_path: str, frames: List[np.ndarray], config: Config):
    cache_path = get_cache_path(video_path, config)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(frames, f)
    except Exception as e:
        print(f"Warning: Failed to save cache for {video_path}: {e}")

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        face = frame[y:y+h, x:x+w]
        return face


# ============================================================================
# DATASET (Same structure, but with improved frame selection)
# ============================================================================

class DeepfakeDataset(Dataset):
    def __init__(self, video_paths: List[str], labels: List[int], 
                 config: Config, is_training: bool = True, 
                 face_detector: Optional[FaceDetector] = None):
        self.video_paths = video_paths
        self.labels = labels
        self.config = config
        self.is_training = is_training
        self.augmentation = get_train_augmentation() if is_training else get_val_augmentation()
        
        face_detector_local = face_detector or FaceDetector()
        self.frames_cache = {}
        print(f"Loading/extracting frames for {len(video_paths)} videos...")
        
        valid_video_paths = []
        valid_labels = []
        skipped_count = 0
        
        for idx in tqdm(range(len(video_paths)), desc="Processing frames"):
            try:
                frames = self._extract_frames(video_paths[idx], face_detector_local)
                if frames is not None and len(frames) > 0:
                    valid_video_paths.append(video_paths[idx])
                    valid_labels.append(labels[idx])
                else:
                    skipped_count += 1
            except Exception as e:
                skipped_count += 1
        
        self.video_paths = valid_video_paths
        self.labels = valid_labels
        
        if skipped_count > 0:
            print(f"\n⚠️  Skipped {skipped_count} invalid videos")
            print(f"✅ Using {len(valid_video_paths)} valid videos")
    
    def _extract_frames(self, video_path: str, face_detector: FaceDetector) -> List[np.ndarray]:
        if video_path in self.frames_cache:
            return self.frames_cache[video_path]
        
        cached_frames = load_frames_from_cache(video_path, self.config)
        if cached_frames is not None:
            self.frames_cache[video_path] = cached_frames
            return cached_frames
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened() or os.path.basename(video_path).startswith("._"):
                return None
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None
            
            # Better frame selection - use more frames from different parts
            frame_indices = np.linspace(0, total_frames - 1, self.config.FRAMES_PER_VIDEO, dtype=int)
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx in frame_indices:
                    face = face_detector.detect_face(frame)
                    if face is not None:
                        face_resized = cv2.resize(face, (self.config.INPUT_SIZE, self.config.INPUT_SIZE))
                        frames.append(face_resized)
                
                frame_idx += 1
                if len(frames) >= self.config.FRAMES_PER_VIDEO:
                    break
            
            cap.release()
            
            while len(frames) < self.config.FRAMES_PER_VIDEO:
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3), dtype=np.uint8))
            
            frames = frames[:self.config.FRAMES_PER_VIDEO]
            save_frames_to_cache(video_path, frames, self.config)
            self.frames_cache[video_path] = frames
            return frames
        except Exception:
            return None
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.frames_cache[video_path]
        
        # Select random frame during training, first frame during validation
        if self.is_training:
            frame = frames[np.random.randint(0, len(frames))]
        else:
            frame = frames[0]
        
        augmented = self.augmentation(image=frame)
        image = augmented['image']
        
        if self.is_training and self.config.LABEL_SMOOTHING > 0:
            label = label * (1 - self.config.LABEL_SMOOTHING) + 0.5 * self.config.LABEL_SMOOTHING
        
        return image, torch.tensor(label, dtype=torch.float32)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics"""
    y_true_binary = np.round(y_true).astype(int)
    
    valid_mask = ~(np.isnan(y_proba) | np.isinf(y_proba))
    if not np.any(valid_mask):
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
    
    y_true_binary = y_true_binary[valid_mask]
    y_proba = y_proba[valid_mask]
    y_pred_binary = (y_proba >= 0.5).astype(int)
    
    if len(y_true_binary) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
    
    metrics = {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }
    
    if len(np.unique(y_true_binary)) > 1 and np.all(np.isfinite(y_proba)):
        try:
            metrics['auc'] = roc_auc_score(y_true_binary, y_proba)
        except ValueError:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


# ============================================================================
# TRAINER (Optimized)
# ============================================================================

class Trainer:
    def __init__(self, config: Config, model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 test_loader: Optional[DataLoader] = None):
        self.config = config
        
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is not available! Training requires a GPU.")
        
        self.model = model.to(config.DEVICE)
        next_param = next(self.model.parameters())
        if next_param.device.type != 'cuda':
            raise RuntimeError(f"❌ Model is not on GPU!")
        
        print(f"✅ Model successfully moved to GPU: {next_param.device}")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Optimizer with different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1},  # Lower LR for pretrained
            {'params': classifier_params, 'lr': config.LEARNING_RATE}
        ], weight_decay=config.WEIGHT_DECAY)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss function
        if config.USE_FOCAL_LOSS:
            self.criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.scaler = GradScaler() if config.USE_AMP else None
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_auc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': [],
            'overfitting_gap': []
        }
        
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probas = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.config.DEVICE, non_blocking=True)
            labels = labels.to(self.config.DEVICE, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
                self.optimizer.step()
            
            running_loss += loss.item()
            
            with torch.no_grad():
                probas = torch.sigmoid(outputs).cpu().numpy()
                probas = np.nan_to_num(probas, nan=0.5, posinf=1.0, neginf=0.0)
                probas = np.clip(probas, 0.0, 1.0)
                preds = (probas >= 0.5).astype(int)
                all_preds.extend(preds)
                all_probas.extend(probas)
                all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        self.scheduler.step()
        
        avg_loss = running_loss / len(self.train_loader)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probas))
        
        return {'loss': avg_loss, **metrics}
    
    def validate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Validate model"""
        loader = test_loader if test_loader is not None else self.val_loader
        desc = "Test" if test_loader is not None else "Validation"
        
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probas = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for images, labels in pbar:
                images = images.to(self.config.DEVICE, non_blocking=True)
                labels = labels.to(self.config.DEVICE, non_blocking=True)
                
                if self.config.USE_AMP:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probas = torch.sigmoid(outputs).cpu().numpy()
                probas = np.nan_to_num(probas, nan=0.5, posinf=1.0, neginf=0.0)
                probas = np.clip(probas, 0.0, 1.0)
                preds = (probas >= 0.5).astype(int)
                all_preds.extend(preds)
                all_probas.extend(probas)
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(loader)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probas))
        
        return {'loss': avg_loss, **metrics}
    
    def train(self):
        """Full training loop"""
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is not available!")
        
        print(f"\n{'='*60}")
        print("Starting Optimized Training")
        print(f"{'='*60}")
        print(f"Device: {self.config.DEVICE}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Model: {self.config.MODEL_ARCHITECTURE}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Focal Loss: {self.config.USE_FOCAL_LOSS}")
        print(f"{'='*60}\n")
        
        torch.cuda.empty_cache()
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
            print("-" * 60)
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            overfitting_gap = train_metrics['accuracy'] - val_metrics['accuracy']
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['overfitting_gap'].append(overfitting_gap)
            
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}")
            print(f"Overfitting Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"GPU Memory: {gpu_memory:.2f} GB")
            
            if overfitting_gap > self.config.OVERFITTING_THRESHOLD:
                print(f"⚠️  WARNING: Overfitting detected!")
            
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.early_stopping_counter = 0
                self.save_checkpoint(self.config.BEST_MODEL_PATH, epoch, is_best=True)
                print(f"✅ New best model saved! Val Acc: {val_metrics['accuracy']:.4f}")
            else:
                self.early_stopping_counter += 1
            
            if epoch % self.config.CHECKPOINT_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"
                )
                self.save_checkpoint(checkpoint_path, epoch, is_best=False)
            
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                break
            
            torch.cuda.empty_cache()
            gc.collect()
        
        self.save_checkpoint(self.config.FINAL_MODEL_PATH, epoch, is_best=False)
        print(f"\n✅ Training completed! Best Val Acc: {self.best_val_acc:.4f}")
        
        if self.test_loader is not None:
            print(f"\n{'='*60}")
            print("Evaluating on Test Set")
            print(f"{'='*60}")
            torch.cuda.empty_cache()
            gc.collect()
            test_metrics = self.validate(test_loader=self.test_loader)
            print(f"\nTest - Loss: {test_metrics['loss']:.4f}, "
                  f"Acc: {test_metrics['accuracy']:.4f}, "
                  f"F1: {test_metrics['f1']:.4f}, "
                  f"AUC: {test_metrics['auc']:.4f}")
            print(f"{'='*60}\n")
        
        self.plot_training_curves()
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': {
                'MODEL_ARCHITECTURE': self.config.MODEL_ARCHITECTURE,
                'BATCH_SIZE': self.config.BATCH_SIZE,
                'LEARNING_RATE': self.config.LEARNING_RATE,
                'INPUT_SIZE': self.config.INPUT_SIZE,
            }
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(checkpoint, path)
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, self.history['overfitting_gap'], 'g-', linewidth=2)
        axes[1, 0].axhline(y=self.config.OVERFITTING_THRESHOLD, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Overfitting Gap')
        axes[1, 0].set_title('Overfitting Gap')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, self.history['train_auc'], 'b-', label='Train AUC')
        axes[1, 1].plot(epochs, self.history['val_auc'], 'r-', label='Val AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Training and Validation AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves_optimized.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Training curves saved to: training_curves_optimized.png")


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def prepare_dataset(config: Config):
    """Prepare dataset with train/test/val split"""
    print("Preparing dataset...")
    
    real_videos = sorted([
        str(p) for p in Path(config.REAL_VIDEOS_DIR).glob("*.mp4") 
        if not os.path.basename(p).startswith("._")
    ])[:config.NUM_REAL_VIDEOS]
    fake_videos = sorted([
        str(p) for p in Path(config.FAKE_VIDEOS_DIR).glob("*.mp4")
        if not os.path.basename(p).startswith("._")
    ])[:config.NUM_FAKE_VIDEOS]
    
    # Split videos
    real_train, real_temp, _, _ = train_test_split(
        real_videos, [0] * len(real_videos),
        test_size=(config.TEST_SPLIT + config.TRAIN_VAL_SPLIT),
        random_state=config.RANDOM_SEED
    )
    test_size_from_temp = config.TEST_SPLIT / (config.TEST_SPLIT + config.TRAIN_VAL_SPLIT)
    real_test, real_val, _, _ = train_test_split(
        real_temp, [0] * len(real_temp),
        test_size=test_size_from_temp,
        random_state=config.RANDOM_SEED
    )
    
    fake_train, fake_temp, _, _ = train_test_split(
        fake_videos, [1] * len(fake_videos),
        test_size=(config.TEST_SPLIT + config.TRAIN_VAL_SPLIT),
        random_state=config.RANDOM_SEED
    )
    fake_test, fake_val, _, _ = train_test_split(
        fake_temp, [1] * len(fake_temp),
        test_size=test_size_from_temp,
        random_state=config.RANDOM_SEED
    )
    
    train_videos = real_train + fake_train
    train_labels = [0] * len(real_train) + [1] * len(fake_train)
    test_videos = real_test + fake_test
    test_labels = [0] * len(real_test) + [1] * len(fake_test)
    val_videos = real_val + fake_val
    val_labels = [0] * len(real_val) + [1] * len(fake_val)
    
    print(f"\n{'='*60}")
    print("Dataset Split Summary:")
    print(f"{'='*60}")
    print(f"TRAIN: {len(train_videos)} videos")
    print(f"TEST:  {len(test_videos)} videos")
    print(f"VAL:   {len(val_videos)} videos")
    print(f"{'='*60}\n")
    
    return train_videos, train_labels, test_videos, test_labels, val_videos, val_labels


def train_model():
    """Main training function"""
    config = Config()
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available!")
    
    train_videos, train_labels, test_videos, test_labels, val_videos, val_labels = prepare_dataset(config)
    
    face_detector = FaceDetector()
    train_dataset = DeepfakeDataset(train_videos, train_labels, config, is_training=True, face_detector=face_detector)
    val_dataset = DeepfakeDataset(val_videos, val_labels, config, is_training=False, face_detector=face_detector)
    test_dataset = DeepfakeDataset(test_videos, test_labels, config, is_training=False, face_detector=face_detector)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)
    
    model = create_model(config)
    trainer = Trainer(config, model, train_loader, val_loader, test_loader)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description='Optimized Deepfake Detection System')
    parser.add_argument('--mode', type=str, required=True, choices=['train'],
                       help='Mode: train')
    parser.add_argument('--model', type=str, default='efficientnet_b4',
                       choices=['efficientnet_b4', 'resnet101', 'vit_base', 'ensemble'],
                       help='Model architecture')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Set model architecture
        config = Config()
        config.MODEL_ARCHITECTURE = args.model
        train_model()


if __name__ == '__main__':
    main()

