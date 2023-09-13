from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import gradio as gr
# dirname = os.path.dirname(__file__)
model_path = '/Users/siddharthkms/PycharmProjects/pythonProject4/braintumour.h5'
model = load_model(model_path)

# model = load_model(os.path.join(dirname, '/Users/siddharthkms/PycharmProjects/pythonProject4/braintumour.h5'))
import cv2

# img = cv2.imread('/Users/siddharthkms/PycharmProjects/pythonProject4/archive/Testing/no_tumor/image(67).jpg')
# img = cv2.resize(img,(150,150))
# img_array = np.array(img)
# img_array.shape
# img_array = img_array.reshape(1,150,150,3)
# img_array.shape


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


# iface = gr.Interface(fn=predict_image, inputs="image", outputs="text", capture_session=True)
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.inputs.Image(label="Upload MRI scan here"),
    outputs=gr.outputs.Textbox(label="Prediction: "),
    title="Brain Tumor Detection",
    description="Upload a preprocessed MRI scan to get the tumor classification.",

)


iface.launch()



# Declaring Variables

# gt = "The patient has Glioma Tumour. "
# mt = "The patient has Meningioma Tumour. "
# nt = "The patient does not have any Tumour. "
# pt = "The patient has Pituitary Tumour. "
#
# # Make predictions
# a = model.predict(img_array)
# indices = a.argmax(axis=1)
#
# # Show the image with prediction
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
# if(indices==0):
#     plt.title(f'Predicted Class: {indices} Glioma Tumour detected')
# elif(indices == 1):
#     plt.title(f'Predicted Class: {indices} Meningioma Tumour detected')
# elif(indices == 2):
#     plt.title(f'Predicted Class: {indices} No Tumour detected')
# else:
#     plt.title(f'Predicted Class: {indices} Pituitary Tumour detected')
#
# plt.axis('off')  # Hide axis
# plt.show()


# if(indices==0):
#     print(indices, gt)
# elif(indices == 1):
#     print(indices , mt)
# elif(indices == 2):
#     print(indices, nt)
# else:
#     print(indices, pt)


