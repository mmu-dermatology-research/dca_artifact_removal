import tensorflow as tf
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re

import gradcam

def main():
    # !    Use / not \    ! #
    path_to_model = r"/home/sam/Repositories/temp/artifact_project/models/clean/InceptionResNetV2/SGD/64/InceptionResNetV2_batchSize_0_opt_SGD_model.29.h5"
    path_to_image = r"/home/sam/Repositories/temp/artifact_project/images/MicrosoftTeams-image.png"
    conv_layer_name = "conv_7b_ac"
    
    #########################################################################################
    (heatmap, output) = run_gradcam_experiment(path_to_model, path_to_image, conv_layer_name)
    
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(output);axes[0].set_title("Output", fontsize=16)
    axes[1].imshow(heatmap);axes[1].set_title("Heatmap", fontsize=16)

def run_gradcam_experiment(path_to_model : str, path_to_image : str, conv_layer_name : str):
    # Load the model
    model = load_model_custom(path_to_model)

    # Load the image
    image_name = re.split("/", path_to_image)[-1]
    image = load_image(path_to_image, preprocess = True)

    # Get prediction
    prediction = get_prediction(model, image)

    # Generate gradcam heatmap
    cam = create_gradcam_model(model, prediction, conv_layer_name)
    heatmap = get_heatmap(cam, image)

    # Reload image
    image = load_image(path_to_image)
    (heatmap, output) = layer_map(cam, heatmap, image, alpha = 0.5)

    return (heatmap, output)
    
def run_gradcam_experiment2(model, path_to_image : str, conv_layer_name : str):
    # Load the model
    #model = load_model_custom(path_to_model)

    # Load the image
    image_name = re.split("/", path_to_image)[-1]
    image = load_image(path_to_image, preprocess = True)

    # Get prediction
    prediction = get_prediction(model, image)

    # Generate gradcam heatmap
    cam = create_gradcam_model(model, prediction, conv_layer_name)
    heatmap = get_heatmap(cam, image)

    # Reload image
    image = load_image(path_to_image)
    (heatmap, output) = layer_map(cam, heatmap, image, alpha = 0.5)

    return (heatmap, output)

def load_model_custom(path_to_model : str):
    model = keras.models.load_model(path_to_model)
    return model

def load_image(path_to_image : str, preprocess : bool = False):
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if preprocess:
        image = image.astype('float32') / 255
        image = np.expand_dims(image, axis = 0)
    return image

def get_prediction(model, image):
    prediction = model.predict(image)
    return np.argmax(prediction[0])

def create_gradcam_model(model, prediction, layer_name : str):
    cam = gradcam.GradCAM(model, prediction, layer_name)
    return cam

def get_heatmap(cam, image):
    heatmap = cam.compute_heatmap(image)
    return heatmap

def layer_map(cam, heatmap, image, alpha : float = 0.5):
    (heatmap, output) = cam.overlay_heatmap(heatmap, image, alpha)
    return (heatmap, output)
