# Traffic Sign Classification Model

This repository contains a deep learning model for classifying traffic signs. The model is built using PyTorch and is trained to recognize various traffic signs such as speed limits, stop signs, and traffic lights. The project includes scripts for training the model, making predictions, and evaluating its performance.

---

## Table of Contents
- [Model Architecture](#model-architecture)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [How to Use](#how-to-use)
  - [Training](#training)
  - [Prediction](#prediction)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)

---

## Model Architecture

The model is based on a Convolutional Neural Network (CNN) architecture designed for image classification. The architecture includes:
- **Convolutional Layers**: Extract spatial features from images.
- **Pooling Layers**: Reduce spatial dimensions to prevent overfitting.
- **Fully Connected Layers**: Perform classification based on extracted features.
- **Activation Functions**: ReLU for non-linearity and softmax for output probabilities.

The model is implemented in the `_model.py` file as `TrafficSignCNN`.

---

## Inputs

### Training
- **Input**: Directory paths for training and validation datasets.
  - Each directory should contain subdirectories named after the class labels, with images inside.
  - The dataset used for training and testing the model can be downloaded from Kaggle: https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou 
  - Example structure:
    ```
    data/
    ├── train/
    │   ├── Stop/
    │   │   ├── stop1.jpg
    │   │   ├── stop2.jpg
    │   └── Speed_Limit_30/
    │       ├── speed30_1.jpg
    │       ├── speed30_2.jpg
    └── val/
        ├── Stop/
        ├── Speed_Limit_30/
    ```
- **Image Size**: Images are resized to `(32, 32)` during preprocessing.

### Prediction
- **Input**: 
  - A single image file (e.g., `.jpg`, `.png`, `.jpeg`).
  - A directory containing multiple image files.

---

## Outputs

### Training
- **Metrics**: 
  - Training loss and accuracy.
  - Validation loss and accuracy for each epoch.
- **Model Checkpoint**: The best model is saved to the path specified in `_config.py` (e.g., `_checkpoints/final_weights.pth`).

### Prediction
- **Output**: Predicted class label(s) for the input image(s).
  - For a single image: A string representing the predicted class.
  - For a directory: A dictionary where keys are filenames and values are predicted class labels.

---

## How to Use

### Training

1. **Prepare the Dataset**:
   - Organize your dataset into `train` and `val` directories as described in the [Inputs](#inputs) section.

2. **Run the Training Script**:
   - Update the dataset paths in `_train.py`:
     ```python
     train_dir = "./data/train"
     val_dir = "./data/val"
     ```
   - Execute the script:
     ```bash
     python _train.py
     ```

3. **Monitor Training**:
   - The script will display training and validation metrics for each epoch.
   - The best model will be saved automatically.

---

### Prediction

1. **Prepare the Input**:
   - Place your test images in a directory or use a single image file.

2. **Run the Prediction Script**:
   - Update the input path in [_predict.py](http://_vscodecontentref_/1):
     ```python
     input_path = "./data/test"  # Replace with your image or directory path
     ```
   - Execute the script:
     ```bash
     python _predict.py
     ```

3. **View Results**:
   - For a single image, the predicted class will be printed.
   - For a directory, predictions for all images will be displayed.

---

## Dependencies

Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- Pillow

Install them using:
```bash
pip install torch torchvision tqdm pillow
