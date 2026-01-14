# Handwritten Digit Recognition using CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** to accurately recognize handwritten digits (0–9) using the **MNIST dataset**. The model learns spatial features from grayscale images and achieves high classification accuracy through optimized preprocessing, architecture design, and training strategies. The project demonstrates strong fundamentals in **deep learning, computer vision, and model evaluation**.

---

## Problem Statement
Handwritten digit recognition is a classic computer vision problem with real-world applications such as:
- Optical Character Recognition (OCR)
- Bank cheque processing
- Postal code recognition
- Form digitization systems

The objective is to build a robust model that can classify handwritten digits with **high accuracy and generalization performance**.

---

## Dataset
- **Dataset:** MNIST Handwritten Digits
- **Training Samples:** 60,000
- **Test Samples:** 10,000
- **Image Size:** 28 × 28 pixels (grayscale)
- **Classes:** 10 (digits 0–9)

---

## Model Architecture
The model is built using a **CNN architecture** optimized for image classification:

- Input Layer: 28 × 28 × 1
- Convolution Layers with ReLU activation
- MaxPooling layers for spatial downsampling
- Fully Connected (Dense) layers
- Softmax output layer for multi-class classification
- LeNet-5 was used as a reference CNN architecture and trained from scratch to compare against custom model variants.

**Frameworks Used:** TensorFlow / Keras

---

## Training Details
- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Batch Size:** 128  
- **Epochs:** 10–15 (configurable)  
- **Normalization:** Pixel values scaled to [0, 1]  
- **Encoding:** One-Hot Encoding for labels  

---

## Performance Metrics
| Metric            | Value |
|-------------------|-------|
| Training Accuracy | ~99.5% |
| Test Accuracy     | ~99.0% |
| Training Loss     | < 0.02 |
| Test Loss         | < 0.05 |

The model shows **strong generalization** with minimal overfitting.

---

## Evaluation
- Confusion Matrix analysis to inspect misclassifications
- Visualization of predictions vs actual labels
- Performance validated on unseen test data

---

How to Run
----------

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition-cnn.git

2. Navigate to the project directory
   ```bash
   cd handwritten-digit-recognition-cnn

3. Install dependencies
   ```bash
   pip install -r requirements.txt

4. Open the Jupyter Notebook
   ```bash
   jupyter notebook Handwritten_Digit_Recognition_using_CNN.ipynb


---

## Project Structure
```bash
├── Handwritten_Digit_Recognition_using_CNN.ipynb
├── README.md
├── requirements.txt
```
## Technologies & Tools
- **Programming Language:** Python
- **Libraries:**
  - TensorFlow / Keras
  - NumPy
  - Matplotlib
  - Scikit-learn

---

## Key Skills Demonstrated
- Convolutional Neural Networks (CNN)
- Image preprocessing & normalization
- One-Hot Encoding
- Model training & evaluation
- Confusion Matrix analysis
- Deep Learning with TensorFlow/Keras
