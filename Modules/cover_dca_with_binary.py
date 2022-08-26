import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import realistic_dca
import pandas as pd

def main():
    """Module testing main method

    """
    image_path = "/home/sam/Repositories/temp/artifact_project/images/balanced_dca_split/data/test/"
    mask_path = "/home/sam/Repositories/temp/artifact_project/images/balanced_dca_split/masks/"

    for dca_size in os.listdir(image_path + "base/"):
        for im_class in os.listdir(image_path + "base/" + dca_size + "/"):
            for image_name in os.listdir(image_path + "base/" + dca_size + "/" + im_class + "/"):
                image = np.asarray(Image.open(image_path + "base/" + dca_size + "/" + im_class + "/" + image_name))
                mask = np.asarray(Image.open(mask_path + image_name[:-4] + "_MASK.png"))
                augmented = realistic_dca.augment_binary_dca(image, mask)
                Image.fromarray(augmented).save(image_path + "base_binary/" + dca_size + "/" + im_class + "/" + image_name[:-3] + "png")

if __name__ == "__main__":
    main()
