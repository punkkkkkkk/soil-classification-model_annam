# 🌱 Soil Classification with Deep Learning


<span style="color:blue">NOTE : soil-classification-model.ipynb contains the code for both training and inference</span>

A high-performance deep learning model for classifying soil types from images with 95%+ F1 score.


https://github.com/user-attachments/assets/ed17539c-5570-4c8a-864c-787401751bf2


## 📋 Overview

This project uses computer vision to classify soil samples into four categories:
- Alluvial soil
- Black Soil
- Clay soil
- Red soil

Perfect for agricultural tech, environmental monitoring, and soil science applications!

## 🔍 Dataset Structure

The dataset contains:
- Training images of soil samples
- Test images for prediction
- Labels CSV with image IDs and soil types

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/soil-classification.git

# Install dependencies
pip install torch torchvision albumentations pandas scikit-learn pillow tqdm ranger-adabelief
```

## 💻 How It Works

### 🧠 Model Architecture

- **Base Model**: ResNet50 pretrained on ImageNet
- **Custom Head**: Added dropout layers and an intermediate layer for better feature learning
- **Training Strategy**: Freeze early layers, train later layers for soil-specific features

### 🏋️ Training Process

Our approach uses:
- **5-fold Cross-Validation**: Ensures robust performance across different data splits
- **Data Augmentation**: Flips, rotations, brightness adjustments to prevent overfitting
- **Advanced Optimizer**: RangerAdaBelief for faster convergence
- **Cyclic Learning Rate**: Helps escape local minima and find better solutions
- **Early Stopping**: Saves the best model based on minimum F1 score across classes

### 🔮 Prediction with Ensemble

For maximum accuracy:
- **Model Ensemble**: Combines all 5 fold models for prediction
- **Test-Time Augmentation**: Uses multiple views of each image for robust predictions
- **Temperature Scaling**: Calibrates confidence scores

## 📊 Performance

The model achieves:
- **95%+ F1 Score** on the validation set
- **High Accuracy** across all soil types
- **Robust Generalization** to unseen images

## 🚀 Quick Start

```python
# Train models
python train.py

# Generate predictions
python predict.py --input /path/to/test/images --output predictions.csv
```

## 📝 Code Structure

1. **Cell 1**: Libraries and dependencies
2. **Cell 2**: Data loading and fold preparation
3. **Cell 3**: Model, optimizer, scheduler, and loss function setup
4. **Cell 4**: Transforms, dataset class, and dataloaders
5. **Cell 5**: Training and validation functions
6. **Cell 6**: Full training loop with cross-validation
7. **Cell 7**: Ensemble inference with TTA and performance evaluation

## 👨‍💻 Key Features

- Cross-validation strategy for robust results
- Advanced augmentation techniques
- Optimizer tuning for better convergence
- Ensemble prediction with test-time augmentation
- Detailed performance metrics for each soil class

## ⭐ Results

| Soil Type | F1 Score |
|-----------|----------|
| Alluvial soil | 0.97 |
| Black Soil | 0.95 |
| Clay soil | 0.96 |
| Red soil | 0.97 |
| **Overall** | **0.96** |

---

# 🚀 How to Run This Project

### Prerequisites
- Python 3.7+ installed
- CUDA-compatible GPU recommended (for faster training)
- Access to the soil classification dataset

### Setup
1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/soil-classification.git
   cd soil-classification
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Prepare your environment:**
   - For Jupyter: `jupyter notebook`
   - For Python script: Make sure your working directory contains the code files

### Dataset Structure
Ensure your dataset is organized as follows:

/dataset

/train           # Training images

/test            # Test images

train_labels.csv # CSV with image_id and soil_type columns

### Running the Model

#### Using Jupyter Notebook
1. Open `soil_classification.ipynb` in Jupyter
2. Execute cells sequentially (Shift+Enter)
3. Modify parameters in cell 6 to adjust training (epochs, batch size, etc.)
4. The final cell will generate `submission.csv` with predictions


### Expected Outputs
- Trained models saved to `/models` directory as `.pth` files
- Training metrics printed to console
- F1 scores for each soil class displayed after training
- Final predictions saved to `submission.csv`

### Troubleshooting
- **Memory errors:** Reduce batch size in cell 4 and cell 6
- **CUDA errors:** Try `torch.cuda.empty_cache()` or restart kernel
- **Import errors:** Ensure all dependencies are installed correctly
- **Dataset errors:** Verify your image paths and CSV structure match the expected format
