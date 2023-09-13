from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import gradio as gr

model_path = '/Users/siddharthkms/PycharmProjects/pythonProject4/braintumour.h5'
model = load_model(model_path)

import cv2

def predict_image(image):
    # Preprocess the image
    img_array = cv2.resize(image, (150, 150))
    img_array = np.array(img_array)
    img_array = img_array.reshape(1, 150, 150, 3)
    a = model.predict(img_array)
    indices = a.argmax(axis=1)
    if (indices == 0):
        a = "Glioma Tumour detected"
    elif (indices == 1):
        a = "Meningioma Tumour detected"
    elif (indices == 2):
        a = "No Tumour detected"
    else:
        a = "Pituitary Tumour detected"
    result = f'Prediction: {indices} {a}'
    return result


iface = gr.Interface(
    fn=predict_image,
    inputs=gr.inputs.Image(label="Upload MRI scan here"),
    outputs=gr.outputs.Textbox(label="Prediction: "),
    title="Brain Tumor Detection",
    description="Upload a preprocessed MRI scan to get the tumor classification.",

)

iface.launch()



