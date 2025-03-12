# Brain-Tumour-Detection

This project uses a Convolutional Neural Network (CNN) to detect brain tumours from preprocessed MRI scan images. It further classifies three types of brain tumours:
- Glioma tumour
- Pituitary tumour
- Meningioma tumour

## Dataset

The dataset is available on Kaggle:  
[Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

## Model Performance

The model works well on the testing dataset containing preprocessed MRI scans of the brain. **I got 94..16 % acciracy** on the testing set.

## Project Structure

- **exp.py:** Contains the main code of the project, including the CNN architecture, training epochs, and necessary deep learning libraries.
- **new.py:** Deploys the model using Gradio.
- Other files(bt.png) include the training history data used to plot curves of validation accuracy and training accuracy.

## Web Application

The output is demonstrated in a simple web application built using Gradio and Flask. (The Flask file will be uploaded soon.)

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
