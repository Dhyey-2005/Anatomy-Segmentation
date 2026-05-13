# Surgical Anatomy Segmentation System

## Overview
This project focuses on surgical anatomy segmentation using deep learning and transformer-based models. The system identifies and segments different anatomical structures from real surgical images.

The project explores multiple architectures including:
- UNET
- Swin-UNet
- Swin-UNETR
- DeepLabV3

The goal is to improve automated understanding of surgical scenes for future AI-assisted surgery and medical applications.

---

## Features
- Multi-organ surgical anatomy segmentation
- Transformer-based deep learning models
- GPU-accelerated mixed precision training
- Quantitative and qualitative evaluation
- Mask-based anatomy reasoning
- Training visualization and performance analysis

---

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- CUDA / GPU Training

---

## Dataset
The model is trained on surgical images with annotated anatomical masks.  
Different categories of masks are used for segmentation tasks.

Example segmented structures:
- Liver
- Pancreas
- Stomach
- Spleen
- Colon
- Small Intestine
- Abdominal Wall

---

## Model Pipeline
1. Dataset preprocessing
2. Image and mask loading
3. Data augmentation
4. Model training
5. Validation and testing
6. Metric evaluation
7. Visualization of predictions

---

## Evaluation Metrics
The models are evaluated using:
- Pixel Accuracy
- Mean IoU
- Dice Score
- Confusion Matrix

---

## Training Features
- Early stopping
- Mixed precision training
- GPU optimization
- Best model checkpoint saving
- Automated evaluation pipeline

---

## Results
The system successfully segments multiple anatomical structures with strong performance on surgical datasets. Evaluation includes both numerical metrics and visual prediction overlays.

---

## Future Improvements
- Real-time surgical segmentation
- Better pancreas segmentation
- Larger medical datasets
- Multi-task learning
- Clinical deployment optimization

---

## Applications
- AI-assisted surgery
- Medical image analysis
- Surgical training systems
- Clinical decision support
- Medical research
---

## Author
Developed as a deep learning and medical image segmentation project focused on surgical anatomy understanding
