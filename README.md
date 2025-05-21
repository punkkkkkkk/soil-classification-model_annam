# soil-classification-model_annam

```markdown
🌱 Soil Classification with Deep Learning

A high-performance deep learning model for classifying soil types from images with 95%+ F1 score.

📋 Overview

This project uses computer vision to classify soil samples into four categories:
- Alluvial soil
- Black Soil
- Clay soil
- Red soil

Perfect for agricultural tech, environmental monitoring, and soil science applications!

🔍 Dataset Structure

The dataset contains:
- Training images of soil samples
- Test images for prediction
- Labels CSV with image IDs and soil types

🛠️ Installation

```
# Clone this repository
git clone https://github.com/punkkkkkkk/soil-classification-model_annam.git

# Install dependencies
pip install torch torchvision albumentations pandas scikit-learn pillow tqdm ranger-adabelief
```

💻 How It Works

🧠 Model Architecture

- Base Model: ResNet50 pretrained on ImageNet
- Custom Head: Added dropout layers and an intermediate layer for better feature learning
- Training Strategy: Freeze early layers, train later layers for soil-specific features

🏋️ Training Process

Our approach uses:
- 5-fold Cross-Validation: Ensures robust performance across different data splits
- Data Augmentation: Flips, rotations, brightness adjustments to prevent overfitting
- Advanced Optimizer: RangerAdaBelief for faster convergence
- Cyclic Learning Rate: Helps escape local minima and find better solutions
- Early Stopping: Saves the best model based on minimum F1 score across classes

🔮 Prediction with Ensemble

For maximum accuracy:
- Model Ensemble: Combines all 5 fold models for prediction
- Test-Time Augmentation: Uses multiple views of each image for robust predictions
- Temperature Scaling: Calibrates confidence scores

📊 Performance

The model achieves:
- 96%+ F1 Score on the validation set
- High Accuracy across all soil types
- Robust Generalization to unseen images
```
# Train models
python train.py

# Generate predictions
python predict.py --input /path/to/test/images --output predictions.csv
```

📝 Code Structure

1. Cell 1: Libraries and dependencies
2. Cell 2: Data loading and fold preparation
3. Cell 3: Model, optimizer, scheduler, and loss function setup
4. Cell 4: Transforms, dataset class, and dataloaders
5. Cell 5: Training and validation functions
6. Cell 6: Full training loop with cross-validation
7. Cell 7: Ensemble inference with TTA and performance evaluation

👨‍💻 Key Features

- Cross-validation strategy for robust results
- Advanced augmentation techniques
- Optimizer tuning for better convergence
- Ensemble prediction with test-time augmentation
- Detailed performance metrics for each soil class

⭐ Results
|--------------------------|
|   Soil Type   | F1 Score |
|---------------|----------|
| Alluvial soil |   0.97   |
| Black Soil    |   0.95   |
| Clay soil     |   0.96   |
| Red soil      |   0.97   |
|--------------------------|
| Overall       |   0.96   |
|--------------------------|

```
