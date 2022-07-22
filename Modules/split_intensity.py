"""Module to extract all of the brightness data and store in a .csv file
from all gradcam heatmaps.

@author Bill Cassidy, Sam Pewton
"""
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageStat
import os
import re

import masking_process

def main():
    """Main method..
    """
    # Filepath to the root of the heatmaps
    base_image_path = "/home/sam/Repositories/temp/artifact_project/images/balanced_dca_split/data/extracted_heatmaps/"

    # All image paths for all heatmaps of all sets
    image_paths = [
                   "clean/base/large/mel/",
                   "clean/base/large/oth/",
                   "clean/base/medium/mel/",
                   "clean/base/medium/oth/",
                   "clean/base/small/mel/",
                   "clean/base/small/oth/",
                   "clean/base/oth/mel/",
                   "clean/base/oth/oth/",
                   "clean/ns/large/mel/",
                   "clean/ns/large/oth/",
                   "clean/ns/medium/mel/",
                   "clean/ns/medium/oth/",
                   "clean/ns/small/mel/",
                   "clean/ns/small/oth/",
                   "clean/ns/oth/mel/",
                   "clean/ns/oth/oth/",
                   "clean/telea/large/mel/",
                   "clean/telea/large/oth/",
                   "clean/telea/medium/mel/",
                   "clean/telea/medium/oth/",
                   "clean/telea/small/mel/",
                   "clean/telea/small/oth/",
                   "clean/telea/oth/mel/",
                   "clean/telea/oth/oth/",
                   "binary/base/large/mel/",
                   "binary/base/large/oth/",
                   "binary/base/medium/mel/",
                   "binary/base/medium/oth/",
                   "binary/base/small/mel/",
                   "binary/base/small/oth/",
                   "binary/base/oth/mel/",
                   "binary/base/oth/oth/",
                   "binary/ns/large/mel/",
                   "binary/ns/large/oth/",
                   "binary/ns/medium/mel/",
                   "binary/ns/medium/oth/",
                   "binary/ns/small/mel/",
                   "binary/ns/small/oth/",
                   "binary/ns/oth/mel/",
                   "binary/ns/oth/oth/",
                   "binary/telea/large/mel/",
                   "binary/telea/large/oth/",
                   "binary/telea/medium/mel/",
                   "binary/telea/medium/oth/",
                   "binary/telea/small/mel/",
                   "binary/telea/small/oth/",
                   "binary/telea/oth/mel/",
                   "binary/telea/oth/oth/",
                   "realistic/base/large/mel/",
                   "realistic/base/large/oth/",
                   "realistic/base/medium/mel/",
                   "realistic/base/medium/oth/",
                   "realistic/base/small/mel/",
                   "realistic/base/small/oth/",
                   "realistic/base/oth/mel/",
                   "realistic/base/oth/oth/",
                   "realistic/ns/large/mel/",
                   "realistic/ns/large/oth/",
                   "realistic/ns/medium/mel/",
                   "realistic/ns/medium/oth/",
                   "realistic/ns/small/mel/",
                   "realistic/ns/small/oth/",
                   "realistic/ns/oth/mel/",
                   "realistic/ns/oth/oth/",
                   "realistic/telea/large/mel/",
                   "realistic/telea/large/oth/",
                   "realistic/telea/medium/mel/",
                   "realistic/telea/medium/oth/",
                   "realistic/telea/small/mel/",
                   "realistic/telea/small/oth/",
                   "realistic/telea/oth/mel/",
                   "realistic/telea/oth/oth/"
                  ]

    # Path to the dump of all masks
    mask_path = "/home/sam/Repositories/temp/artifact_project/images/balanced_dca_split/masks/"

    # Template of the new csv to populate
    csv_template = {
                    "Image_Name" : [],
                    "Model" : [],
                    "Test_Set" : [],
                    "DCA_Size" : [],
                    "Class" : [],
                    "Internal_Brightness_RMS" : [],
                    "External_Brightness_RMS" : [],
                    "Internal_Brightness_Mean" : [],
                    "External_Brightness_Mean" : []
                   }

    # Iterate over all images in all paths. Will take a minute to run
    for path in image_paths:
        for image in os.listdir(base_image_path + path):
            heatmap = get_heatmap(base_image_path + path, image)
            outside_mask = get_outside_mask(mask_path, image[:-4]+"_MASK.png")
            inside_mask = get_inside_mask(outside_mask)

            csv_template["Image_Name"].append(image[:-4])
            csv_template["Model"].append(re.split("/", path)[-5])
            csv_template["Test_Set"].append(re.split("/", path)[-4])
            csv_template["DCA_Size"].append(re.split("/", path)[-3])
            csv_template["Class"].append(re.split("/", path)[-2])
            csv_template["Internal_Brightness_RMS"].append(brightness_rms(heatmap, inside_mask))
            csv_template["External_Brightness_RMS"].append(brightness_rms(heatmap, outside_mask))
            csv_template["Internal_Brightness_Mean"].append(brightness_avg(heatmap, inside_mask))
            csv_template["External_Brightness_Mean"].append(brightness_avg(heatmap, outside_mask))

    # Convert dict to DF and save as csv
    df = pd.DataFrame.from_dict(csv_template)
    df.to_csv("../Data/heatmap_csv_files/extracted_brightness_data.csv", index=False)

def get_heatmap(path_to_images, image_name):
    """Load the heatmap

    Parameters
    ----------
    path_to_images : str
        full path to image directory
    image_name : str
        name of the image to load

    Returns
    -------
    PIL.Image
        the heatmap required
    """
    heatmap = Image.open(path_to_images + image_name).convert('L')
    return heatmap

def get_outside_mask(path_to_masks, mask_name):
    """Get the mask for the image associated with the heatmap.

    The outside mask contains a black circle in the center of a white image.

    Parameters
    ----------
    path_to_masks : str
        path to the mask dump
    mask_name : str
        name of the mask to load

    Returns
    -------
    PIL.Image
        mask required
    """
    try:
        outside_mask = Image.open(path_to_masks + mask_name).convert('L')
    except:
        print("No Mask Found")
    return outside_mask

def get_inside_mask(outside_mask):
    """Get the inverted mask of the original.

    The inside mask is a white circle in the center of an image with a black
    background.

    Parameters
    ----------
    outside_mask : PIL.Image
        the original mask to invert

    Returns
    -------
    PIL.Image
        inverted mask
    """
    inside_mask = ImageOps.invert(outside_mask)
    return inside_mask

def brightness_avg(image, mask):
    """Calculate the average pixel brightness within a masked area.

    Parameters
    ----------
    image : PIL.Image
        the heatmap
    mask : PIL.Image
        the mask

    Returns
    -------
    float
        the average pixel brightness
    """
    stat = ImageStat.Stat(image, mask)
    return stat.mean[0]

def brightness_rms(image, mask):
    """Calculate the average pixel RMS within a masked area.

    Parameters
    ----------
    image : PIL.Image
        the heatmap
    mask : PIL.Image
        the mask

    Returns
    -------
    float
        the average pixel RMS
    """
    stat = ImageStat.Stat(image, mask)
    return stat.rms[0]

if __name__ == "__main__":
    main()
