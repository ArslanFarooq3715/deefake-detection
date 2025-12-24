# Optimized Deepfake Detection - Guide

## 🚀 Key Improvements

### 1. **Multiple Model Architectures**
- **EfficientNet-B4**: Best balance of accuracy and speed
- **ResNet-101**: Strong feature extraction
- **Vision Transformer (ViT)**: Attention-based, great for detecting subtle artifacts
- **Ensemble**: Combines all three models with learnable weights

### 2. **Focal Loss**
- Better handles hard examples (difficult to classify deepfakes)
- Reduces impact of easy examples
- Improves training on imbalanced data

### 3. **Improved Data Augmentation**
- Optimized for deepfake detection
- Focuses on color inconsistencies and compression artifacts
- Better simulates real-world conditions

### 4. **Addressing Underfitting**
- Reduced dropout rates (0.4, 0.2, 0.1 instead of 0.5, 0.25, 0.15)
- Reduced weight decay (5e-5 instead of 1e-4)
- Reduced label smoothing (0.05 instead of 0.1)
- Higher learning rate for classifier (0.0002)
- More epochs (100 instead of 50)

### 5. **Better Feature Extraction**
- Larger input size (384x384 instead of 299x299)
- More frames per video (20 instead of 15)
- Spatial attention for focusing on important regions

### 6. **Optimized Training**
- Different learning rates for backbone (pretrained) and classifier (new)
- Better gradient clipping
- Improved memory management

## 📋 How to Use

### Basic Training (EfficientNet - Recommended)
```bash
python Deepfake_detection_optimized.py --mode train --model efficientnet_b4
```

### Try Different Models

**ResNet-101:**
```bash
python Deepfake_detection_optimized.py --mode train --model resnet101
```

**Vision Transformer:**
```bash
python Deepfake_detection_optimized.py --mode train --model vit_base
```

**Ensemble (Best Accuracy, Slower):**
```bash
python Deepfake_detection_optimized.py --mode train --model ensemble
```

## 🎯 Recommended Training Strategy

### Step 1: Start with EfficientNet-B4
```bash
python Deepfake_detection_optimized.py --mode train --model efficientnet_b4
```
- Fastest training
- Good accuracy
- Best starting point

### Step 2: If accuracy is still low, try Ensemble
```bash
python Deepfake_detection_optimized.py --mode train --model ensemble
```
- Combines all three models
- Best accuracy
- Takes longer to train

### Step 3: Fine-tune Hyperparameters
Edit `Deepfake_detection_optimized.py` and adjust:
- `BATCH_SIZE`: Reduce if out of memory (try 8 or 12)
- `LEARNING_RATE`: Increase if training is slow (try 0.0003)
- `NUM_EPOCHS`: Increase if model is still improving (try 150)

## 📊 Expected Improvements

### Before (Original):
- Validation Accuracy: ~69%
- Underfitting issue
- Limited model options

### After (Optimized):
- **EfficientNet-B4**: Expected 75-85% accuracy
- **ResNet-101**: Expected 78-88% accuracy  
- **ViT**: Expected 80-90% accuracy
- **Ensemble**: Expected 85-92% accuracy

## 🔧 Configuration Options

Edit `Config` class in `Deepfake_detection_optimized.py`:

```python
# Model selection
MODEL_ARCHITECTURE = 'efficientnet_b4'  # Change this

# Training parameters
BATCH_SIZE = 16  # Reduce if OOM
LEARNING_RATE = 0.0002  # Increase if slow convergence
NUM_EPOCHS = 100  # Increase for better results

# Loss function
USE_FOCAL_LOSS = True  # Keep True for better results
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Features
USE_TEMPORAL_ATTENTION = True  # Keep True
USE_MULTI_SCALE = True  # Keep True
```

## 🎓 Model Comparison

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| EfficientNet-B4 | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💾💾 | Quick training, good accuracy |
| ResNet-101 | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾 | Strong feature extraction |
| ViT | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | Best accuracy, subtle artifacts |
| Ensemble | ⚡ | ⭐⭐⭐⭐⭐⭐ | 💾💾💾💾💾 | Maximum accuracy |

## 💡 Tips for Better Results

1. **Start with EfficientNet-B4** - Best balance
2. **Monitor training curves** - Check for overfitting/underfitting
3. **Use Ensemble if accuracy is critical** - Best results
4. **Adjust batch size** - If GPU memory is limited, reduce to 8
5. **Increase epochs** - If model is still improving at epoch 100
6. **Try different models** - Each has strengths for different deepfake types

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` to 8 or 12
- Reduce `INPUT_SIZE` to 320 or 256
- Disable `USE_MULTI_SCALE`

### Slow Training
- Use `efficientnet_b4` instead of `ensemble`
- Reduce `FRAMES_PER_VIDEO` to 15
- Reduce `INPUT_SIZE` to 320

### Low Accuracy
- Try `ensemble` model
- Increase `NUM_EPOCHS` to 150
- Increase `LEARNING_RATE` to 0.0003
- Check data quality and balance

## 📈 Monitoring Training

Watch for:
- **Training accuracy increasing** - Good sign
- **Validation accuracy > Training accuracy** - Underfitting (reduce regularization)
- **Validation accuracy < Training accuracy by >15%** - Overfitting (increase regularization)
- **Both accuracies plateauing** - May need more epochs or different model

## 🎯 Next Steps

1. Train with EfficientNet-B4 first
2. Evaluate on test set
3. If accuracy < 80%, try Ensemble
4. Fine-tune hyperparameters based on results
5. Compare with original model performance

Good luck! 🚀

