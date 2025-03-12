# Brain-Tumour-Detection

This project uses a Convolutional Neural Network (CNN) to detect brain tumours from preprocessed MRI scan images. It further classifies three types of brain tumours:
- Glioma tumour
- Pituitary tumour
- Meningioma tumour

## Dataset

The dataset is available on Kaggle:  
[Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

## Model Performance

The model demonstrates robust performance on the testing dataset containing preprocessed MRI scans of the brain. **Achieved an accuracy of 94.16%**, highlighting its effectiveness in accurately detecting and classifying brain tumours on preprocessed images.

## Trained Model

The trained CNN model is saved as a `.h5` file. This file contains the complete model architecture, weights, and training configuration. It can be directly loaded for inference, fine-tuning, or further evaluation using libraries like TensorFlow or Keras.

## Project Structure

- **exp.py:** Contains the main code of the project, including the CNN architecture, training epochs, and necessary deep learning libraries.
- **new.py:** Deploys the model using Gradio.
- Other files include the training history data used to plot curves of validation accuracy and training accuracy.
- **model.h5:** The pre-trained CNN model file. Load it directly for prediction or additional training.

## Web Application

The output is demonstrated in a simple web application built using Gradio and Flask. (The Flask file will be uploaded soon.)

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
