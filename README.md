# Brain-Tumour-Detection
This model uses Convolutional Neural Network(CNN) to detect brain tumour from preprocessed MRI scan images .Further it classifies 3 types of brain tumour i.e, glioma tumour, pituitary tumour and meningioma tumour.
Dataset is is available in kaggle website: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
The model works well on the testing dataset which contains preprocessed MRI scans of the brain.
The output is shown in a simple web app made using gradio.
exp.py refers the main code of the project , it contains CNN architecture,training epochs,and some deep learning libraries.
new.py is deployement of the model using gradio.
the other file contains the data of the training history , used to plot curves of validation accuracy and training accuracy etc.

