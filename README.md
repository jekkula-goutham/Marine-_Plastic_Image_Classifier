🌊 Marine Plastic Pollution Detection using Deep Learning
CNN vs ResNet50 Transfer Learning
📌 Project Overview

Marine plastic pollution poses a serious threat to ocean ecosystems and biodiversity. This project develops a deep learning–based image classification system capable of automatically identifying plastic waste in underwater images.

The project compares two deep learning approaches:

✅ Custom Convolutional Neural Network (CNN)

✅ ResNet50 Transfer Learning Model

The goal is to evaluate performance differences between training a model from scratch and leveraging pretrained deep learning architectures.

🎯 Objectives

Detect plastic waste in underwater images

Compare CNN vs Transfer Learning performance

Apply data augmentation to improve generalization

Evaluate models using classification metrics and confusion matrices

Build an end-to-end deep learning pipeline

📂 Dataset

Dataset Source (Kaggle):

👉 https://www.kaggle.com/datasets/surajit651/souvikdataset

Classes

Plastic

No-Plastic

Dataset structure used:

Plastic train/
    ├── Plastic/
    └── No-Plastic/

Plastic test/
    ├── Plastic/
    └── No-Plastic/


⚠️ Dataset is not included in this repository due to GitHub size limitations.

🛠️ Technologies Used

Python

TensorFlow / Keras

ResNet50 (Transfer Learning)

NumPy

Matplotlib

Seaborn

Scikit-learn

PIL (Image Processing)

Jupyter Notebook

🔎 Exploratory Data Analysis (EDA)

The project begins with dataset inspection:

Image count per class

Class distribution visualization

Random sample image visualization

Bar plots were generated to verify dataset balance across training and testing sets.

🧹 Data Preprocessing & Augmentation

Images were resized to:

224 × 224 pixels

Training Augmentation

Rescaling

Rotation

Zooming

Width & height shifting

Horizontal flipping

ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


This improves model robustness and prevents overfitting.

🧠 Model Architectures
1️ Custom CNN Model

Architecture:

Conv2D (32 filters)

MaxPooling

Conv2D (64 filters)

MaxPooling

Flatten layer

Dense (128 neurons)

Dropout (0.5)

Sigmoid output (Binary classification)

Loss Function:

Binary Crossentropy


Optimizer:

Adam

2️ ResNet50 Transfer Learning

A pretrained ResNet50 model was used for feature extraction.

Steps:

Loaded ImageNet weights

Removed top classification layers

Froze base layers

Added custom dense layers

Applied dropout regularization

Architecture extension:

ResNet50 → Flatten → Dense(512) → Dropout → Sigmoid Output


Learning Rate:

1e-4 (Adam Optimizer)

📊 Model Training

Both models were trained for:

10 epochs
Batch size: 32


Validation accuracy comparison was plotted to analyze performance differences.

📈 Model Evaluation

Evaluation metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Using:

classification_report()
confusion_matrix()


Confusion matrices were visualized using Seaborn heatmaps.

🧪 Prediction System

The project includes a prediction pipeline that:

Loads a local image

Applies preprocessing

Performs classification

Displays prediction result

Example output:

Prediction: Plastic / No-Plastic

* Model Deployment Step

Best performing model saved as:

marine_pollution_resnet50.h5


This allows reuse without retraining.

📊 Key Insights

Transfer learning significantly improves performance with limited datasets.

ResNet50 extracts stronger visual features compared to a custom CNN.

Data augmentation improves generalization on underwater imagery.

Environmental datasets benefit greatly from pretrained architectures.

3️ Download Dataset

Download from Kaggle and place folders locally:

https://www.kaggle.com/datasets/surajit651/souvikdataset

4️ Run Notebook

Open:

marine_plastic_classification.ipynb


Run all cells sequentially.

🌍 Real-World Applications

Marine ecosystem monitoring

Ocean cleanup automation

Underwater robotics vision systems

Environmental sustainability analytics

Smart coastal surveillance

🎓 Skills Demonstrated

Deep Learning Model Development

Transfer Learning

Computer Vision

Data Augmentation

Model Evaluation & Diagnostics

End-to-End ML Workflow

Environmental AI Applications

Author

Goutham Jekkula
MSc Data Science — Atlantic Technological University
GitHub: https://github.com/jekkula-goutham
