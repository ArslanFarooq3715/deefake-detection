# Deepfake Detection System - Step-by-Step Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Step-by-Step Process](#step-by-step-process)
4. [Data Structure](#data-structure)
5. [Configuration](#configuration)
6. [Usage Guide](#usage-guide)
7. [Technical Details](#technical-details)

---

## 🎯 Overview

This is a deep learning-based deepfake detection system that uses the Xception architecture to classify videos as either **REAL** or **FAKE**. The system processes video frames, extracts faces, and uses a convolutional neural network to detect deepfake manipulations.

### Key Features
- ✅ Automatic face detection and extraction
- ✅ Disk caching for extracted frames (faster subsequent runs)
- ✅ Separate train/test/val splits for fake and real videos
- ✅ Heavy data augmentation for robust training
- ✅ Mixed precision training for faster execution
- ✅ Early stopping and overfitting detection
- ✅ Comprehensive evaluation metrics
- ✅ Batch inference support

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   Video Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Face Detection │ (Haar Cascade)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Frame Extraction│ (10 frames per video)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Disk Caching    │ (Save/load frames)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Augmentation│ (Training only)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Xception Model  │ (Pre-trained backbone)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification  │ (REAL/FAKE)
└─────────────────┘
```

---

## 📝 Step-by-Step Process

### **STEP 1: Data Preparation**

#### 1.1 Folder Structure Setup
```
data/
├── real/          # Real videos (MP4 format)
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── fake/          # Fake/deepfake videos (MP4 format)
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

#### 1.2 Data Splitting
The system automatically splits your data into three sets:

**For REAL Videos:**
- **Train Set**: 70% of real videos (for training)
- **Test Set**: 10% of real videos (for final evaluation)
- **Validation Set**: 20% of real videos (for validation during training)

**For FAKE Videos:**
- **Train Set**: 70% of fake videos (for training)
- **Test Set**: 10% of fake videos (for final evaluation)
- **Validation Set**: 20% of fake videos (for validation during training)

**Final Combined Sets:**
- **Train**: Contains both real_train + fake_train
- **Test**: Contains both real_test + fake_test
- **Val**: Contains both real_val + fake_val

**Why separate splits?** This ensures balanced representation and prevents data leakage between classes.

---

### **STEP 2: Frame Extraction & Caching**

#### 2.1 Video Processing
For each video in the dataset:
1. **Open video file** using OpenCV
2. **Calculate frame indices** - Select 15 evenly spaced frames from the video
3. **Extract frames** - Read frames at calculated indices

#### 2.2 Face Detection
For each extracted frame:
1. **Convert to grayscale** for face detection
2. **Detect faces** using Haar Cascade classifier
3. **Select largest face** if multiple faces detected
4. **Add padding** (20 pixels) around the face
5. **Crop face region** from the frame
6. **Resize to 299x299** (Xception input size)

#### 2.3 Disk Caching System
**First Run:**
- Extracts frames from videos
- Detects and crops faces
- **Saves to disk** in `frames_cache/` folder as pickle files
- Filename format: `{video_hash}_{frames_per_video}_{input_size}.pkl`

**Subsequent Runs:**
- **Checks disk cache first**
- If cached frames exist → Loads from disk (much faster!)
- If not cached → Extracts and saves to cache

**Benefits:**
- ⚡ **10-100x faster** on subsequent runs
- 💾 Saves processing time
- 🔄 Consistent frame extraction

---

### **STEP 3: Data Augmentation**

#### 3.1 Training Augmentation (Heavy)
Applied during training to improve model robustness:

**Geometric Transformations:**
- Horizontal flip (50% chance)
- Rotation (±20 degrees, 50% chance)
- Optical distortion
- Elastic transformation
- Grid distortion

**Color Transformations:**
- Random brightness/contrast adjustment
- Hue/saturation/value shifts
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Noise & Blur:**
- Gaussian noise
- Gaussian blur
- Motion blur
- Median blur

**Degradation:**
- Image compression (simulating different quality levels)
- Downscaling
- Random shadows

**Regularization:**
- Coarse dropout (cutout)

#### 3.2 Validation/Test Augmentation (Minimal)
- Only normalization (mean/std normalization)
- No random transformations (for consistent evaluation)

---

### **STEP 4: Model Architecture**

#### 4.1 Backbone: Xception Network
- **Pre-trained Xception-65** from timm library
- Trained on ImageNet (transfer learning)
- Input size: 299x299x3
- Output: Feature vector (2048 dimensions)

#### 4.2 Classifier Head
Custom fully connected layers:
```
Input Features (2048) 
    ↓
Linear(2048 → 2048) + BatchNorm + ReLU + Dropout(0.5)
    ↓
Linear(2048 → 512) + BatchNorm + ReLU + Dropout(0.25)
    ↓
Linear(512 → 128) + BatchNorm + ReLU + Dropout(0.15)
    ↓
Linear(128 → 1) + Sigmoid
    ↓
Output: Probability (0 = REAL, 1 = FAKE)
```

**Progressive Dropout:** Higher dropout in early layers, lower in later layers to prevent overfitting.

---

### **STEP 5: Training Process**

#### 5.1 Training Loop (Per Epoch)

**For each batch:**
1. **Load batch** of images and labels
2. **Apply Mixup augmentation** (50% chance):
   - Mixes two images with random ratio
   - Creates synthetic training samples
3. **Forward pass** through model
4. **Calculate loss** (BCE Loss with Mixup)
5. **Backward pass** (compute gradients)
6. **Gradient clipping** (prevents exploding gradients)
7. **Update weights** using AdamW optimizer
8. **Update learning rate** (Cosine Annealing with Warm Restarts)

**Mixed Precision Training:**
- Uses FP16 for faster computation
- Automatically handles precision scaling

#### 5.2 Validation (Per Epoch)
1. **Set model to eval mode**
2. **Disable gradient computation**
3. **Process validation batches**
4. **Calculate metrics**: Accuracy, Precision, Recall, F1, AUC

#### 5.3 Training Monitoring

**Metrics Tracked:**
- Training loss & accuracy
- Validation loss & accuracy
- Precision, Recall, F1 Score
- AUC (Area Under ROC Curve)
- Overfitting gap (train_acc - val_acc)

**Early Stopping:**
- Monitors validation accuracy
- Stops if no improvement for 10 epochs
- Saves best model automatically

**Overfitting Detection:**
- Warns if gap between train/val accuracy > 15%
- Helps identify when model is memorizing instead of learning

#### 5.4 Model Checkpointing
- **Best model**: Saved when validation accuracy improves
- **Periodic checkpoints**: Every 5 epochs
- **Final model**: Saved at end of training

---

### **STEP 6: Test Evaluation**

After training completes:
1. **Load best model** from checkpoint
2. **Evaluate on test set** (unseen data)
3. **Calculate final metrics**:
   - Test accuracy
   - Test precision, recall, F1
   - Test AUC
4. **Print comprehensive results**

**Why separate test set?** Provides unbiased estimate of model performance on new data.

---

### **STEP 7: Inference (Prediction)**

#### 7.1 Single Video Inference
1. **Load trained model** from checkpoint
2. **Extract frames** from video (with caching)
3. **Detect faces** in each frame
4. **Preprocess** frames (normalization)
5. **Run inference** on each frame
6. **Aggregate predictions**:
   - Average score across frames
   - Min/Max scores
   - Final prediction (REAL/FAKE)
   - Confidence score

#### 7.2 Batch Inference
1. **Process multiple videos** in sequence
2. **Save results** to CSV file
3. **Include statistics**: Average confidence, counts of REAL/FAKE

---

## 📁 Data Structure

### Input Structure
```
project/
├── data/
│   ├── real/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── fake/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
```

### Output Structure (After Running)
```
project/
├── data/              # Input videos
├── frames_cache/      # Cached extracted frames
│   ├── {hash1}.pkl
│   ├── {hash2}.pkl
│   └── ...
├── checkpoints/       # Model checkpoints
│   ├── checkpoint_epoch_5.pth
│   ├── checkpoint_epoch_10.pth
│   └── ...
├── best_deepfake_model.pth    # Best model
├── final_deepfake_model.pth   # Final model
├── training_curves_with_overfitting.png  # Training plots
└── predictions.csv     # Batch inference results
```

---

## ⚙️ Configuration

### Key Configuration Parameters

**Dataset:**
- `REAL_VIDEOS_DIR`: Path to real videos folder
- `FAKE_VIDEOS_DIR`: Path to fake videos folder
- `NUM_REAL_VIDEOS`: Number of real videos to use
- `NUM_FAKE_VIDEOS`: Number of fake videos to use
- `FRAMES_PER_VIDEO`: Number of frames to extract (default: 10)
- `TRAIN_VAL_SPLIT`: Validation split ratio (default: 0.2)
- `TEST_SPLIT`: Test split ratio (default: 0.1)

**Model:**
- `INPUT_SIZE`: Image size (299 for Xception)
- `DROPOUT_RATES`: Dropout probabilities [0.5, 0.25, 0.15]
- `WEIGHT_DECAY`: L2 regularization (1e-4)
- `LABEL_SMOOTHING`: Label smoothing factor (0.1)

**Training:**
- `BATCH_SIZE`: Batch size (32)
- `NUM_EPOCHS`: Maximum epochs (50)
- `LEARNING_RATE`: Initial learning rate (0.0001)
- `USE_AMP`: Mixed precision training (True)
- `EARLY_STOPPING_PATIENCE`: Epochs to wait (10)
- `OVERFITTING_THRESHOLD`: Gap threshold (0.15)

**Caching:**
- `FRAMES_CACHE_DIR`: Cache directory ("frames_cache")

---

## 🚀 Usage Guide

### 1. Training the Model

```bash
python Deepfake_detection.py --mode train
```

**What happens:**
1. Loads videos from `data/real/` and `data/fake/`
2. Splits into train/test/val sets
3. Extracts and caches frames
4. Trains Xception model
5. Saves best model to `best_deepfake_model.pth`
6. Generates training curves plot

### 2. Single Video Inference

```bash
python Deepfake_detection.py --mode inference --video path/to/video.mp4
```

**What happens:**
1. Loads trained model
2. Extracts frames from video
3. Runs inference
4. Prints prediction (REAL/FAKE) with confidence

### 3. Batch Inference

```bash
python Deepfake_detection.py --mode batch --videos video1.mp4 video2.mp4 video3.mp4 --output results.csv
```

**What happens:**
1. Processes multiple videos
2. Saves results to CSV
3. Prints summary statistics

### 4. Using Custom Model

```bash
python Deepfake_detection.py --mode inference --video video.mp4 --model path/to/model.pth
```

---

## 🔧 Technical Details

### Face Detection
- **Method**: Haar Cascade Classifier
- **Classifier**: `haarcascade_frontalface_default.xml`
- **Parameters**:
  - Scale factor: 1.1
  - Min neighbors: 5
  - Min size: 100x100

### Frame Selection
- **Method**: Evenly spaced sampling
- **Formula**: `frame_indices = linspace(0, total_frames-1, FRAMES_PER_VIDEO)`
- **Example**: For 100-frame video, extracts frames at indices [0, 11, 22, 33, ..., 99]

### Loss Function
- **Type**: Binary Cross-Entropy Loss (BCE)
- **With Mixup**: Weighted combination of losses for mixed samples
- **Label Smoothing**: Reduces overconfidence (0.1 smoothing factor)

### Optimizer
- **Type**: AdamW (Adam with weight decay)
- **Learning Rate**: 0.0001
- **Weight Decay**: 1e-4
- **Scheduler**: Cosine Annealing with Warm Restarts
  - T_0: 10 epochs
  - T_mult: 2
  - Eta_min: 1e-6

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve (measures separability)

### Device Support
- **CUDA**: NVIDIA GPUs (fastest)
- **MPS**: Apple Silicon (M1/M2/M3) GPUs
- **CPU**: Fallback option (slower)

---

## 📊 Training Output

### Console Output Example
```
Preparing dataset...
Splitting fake and real videos separately into train/test/val sets...

============================================================
Dataset Split Summary:
============================================================
TRAIN: 280 videos (140 fake, 140 real)
TEST:  40 videos (20 fake, 20 real)
VAL:   80 videos (40 fake, 40 real)
============================================================

Loading/extracting frames for 280 videos...
Cache directory: frames_cache
Processing frames: 100%|████████| 280/280 [02:15<00:00]

Starting Training
============================================================
Device: cuda
Batch Size: 32
Epochs: 50
Mixed Precision: True
============================================================

Epoch 1/50
------------------------------------------------------------
Training: 100%|████████| 9/9 [00:45<00:00]
Validation: 100%|████████| 3/3 [00:12<00:00]

Train - Loss: 0.6234, Acc: 0.6250, F1: 0.6123, AUC: 0.6789
Val   - Loss: 0.5891, Acc: 0.6500, F1: 0.6345, AUC: 0.7123
Overfitting Gap: -0.0250 (-2.50%)
✅ New best model saved! Val Acc: 0.6500
```

### Generated Files
1. **Model Checkpoints**: Saved in `checkpoints/` directory
2. **Best Model**: `best_deepfake_model.pth`
3. **Training Curves**: `training_curves_with_overfitting.png`
4. **Cached Frames**: `frames_cache/*.pkl`

---

## 🎓 Understanding the Results

### Prediction Output
```
============================================================
RESULTS:
============================================================
Video: test_video.mp4
Prediction: 🔴 FAKE
Confidence: 87.45%
Average Score: 0.8745
Min Score: 0.7234
Max Score: 0.9567
Frame-level predictions: ['0.7234', '0.8123', '0.8567', ...]
============================================================
```

**Interpretation:**
- **Prediction**: FAKE or REAL
- **Confidence**: How certain the model is (0-100%)
- **Average Score**: Mean probability across all frames
- **Min/Max Score**: Range of predictions
- **Frame-level**: Individual frame predictions

### Training Curves
The system generates plots showing:
1. **Loss curves**: Train vs Validation loss over epochs
2. **Accuracy curves**: Train vs Validation accuracy
3. **Overfitting gap**: Visual indicator of overfitting
4. **F1 Score**: Model's balanced performance

---

## 🔍 Troubleshooting

### Common Issues

**Issue: "No faces detected in video"**
- **Solution**: Video may not contain clear faces, or face detection parameters need adjustment

**Issue: "Out of memory"**
- **Solution**: Reduce `BATCH_SIZE` or `NUM_WORKERS` in config

**Issue: "Cache files corrupted"**
- **Solution**: Delete `frames_cache/` folder and re-run (will regenerate cache)

**Issue: "Model not found"**
- **Solution**: Ensure you've trained the model first, or specify correct `--model` path

---

## 📈 Performance Tips

1. **Use GPU**: Significantly faster training (10-50x speedup)
2. **Enable caching**: First run extracts frames, subsequent runs are much faster
3. **Adjust batch size**: Larger batches = faster training (if memory allows)
4. **Use mixed precision**: Reduces memory usage and speeds up training
5. **Monitor overfitting**: Adjust dropout or add more augmentation if overfitting

---

## 📚 References

- **Xception Architecture**: Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions
- **Face Detection**: Viola-Jones algorithm (Haar Cascades)
- **Mixup Augmentation**: Zhang et al. (2018). Mixup: Beyond Empirical Risk Minimization
- **Transfer Learning**: Using pre-trained ImageNet weights

---

## 📝 License

This project is for educational and research purposes.

---

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

**Last Updated**: 2024
**Version**: 1.0

