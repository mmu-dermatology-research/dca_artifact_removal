# -*- coding: utf-8 -*-
"""Script to generate a new dataset for the binary classification of melanoma vs non melanoma.

New generated dataset contains 3 seperate training/validation sets. One clean, the other two augmented with DCA.
Also contains various testing sets containing only DCA images. 3 different varieties are base images and two DCA removal
methods applied.
                             
How To Use
----------
    * Modify the filepaths within the main method (lines 123 - 145) 
    * Execute script to run main
    
Requirements/Dependancies
-------------------------
    Datasets:
        * ISIC unbalanced duplicates removed ( https://github.com/mmu-dermatology-research/isic_duplicate_removal_strategy )
        * Fitzpatrick17k
    Packages:
        * numpy
        * pandas
        * cv2
        * pillow
    Custom Scripts ( https://github.com/mmu-dermatology-research/dark_corner_artifact_removal/tree/master/Modules ):     
        * image_modifications.py
        * realistic_dca.py
        * masking_process.py
        * dca_removal.py
    Other:
        * .csv files describing the dataset from the associated
        * EDSR_x4 network must be located within models folder of the project 
            directory in order to execute inpainting steps 
            ( https://github.com/Saafke/EDSR_Tensorflow/tree/master/models )
    
Generated Structure
-------------------
    ...
     └─ isic_balanced_dca_split
         ├─ masks
         └─ data
             ├─ train
             |    ├─ clean
             |    |    ├─ mel
             |    |    └─ oth
             |    ├─ dca_binary
             |    |    ├─ mel
             |    |    └─ oth
             |    └─ dca_realistic
             |         ├─ mel
             |         └─ oth
      	     ├─ val
             |    ├─ clean
             |    |    ├─ mel
             |    |    └─ oth
             |    ├─ dca_binary
             |    |    ├─ mel
             |    |    └─ oth
             |    └─ dca_realistic
             |         ├─ mel
             |         └─ oth
             └─ test
      	         ├─ base
                 |    ├─ small
                 |    |    ├─ mel
                 |    |    └─ oth
                 |    ├─ medium
                 |    |    ├─ mel
                 |    |    └─ oth
                 |    ├─ large
                 |    |    ├─ mel
                 |    |    └─ oth
                 |    └─ oth
                 |         ├─ mel
                 |         └─ oth
                 ├─ ns
                 |    ├─ small
                 |    |    ├─ mel
                 |    |    └─ oth
                 |    ├─ medium
                 |    |    ├─ mel
                 |    |    └─ oth
                 |    ├─ large
                 |    |    ├─ mel
                 |    |    └─ oth
                 |    └─ oth
                 |         ├─ mel
                 |         └─ oth
    	         └─ telea
                      ├─ small
                      |    ├─ mel
                      |    └─ oth
                      ├─ medium
                      |    ├─ mel
                      |    └─ oth
                      ├─ large
                      |    ├─ mel
                      |    └─ oth
                      └─ oth
                           ├─ mel
                           └─ oth

Methods
-------
main
    main program method
generate_dataset
    function to generate the new dataset
__create_folder_structure
    remove old directory if exists and create new directory ready to transfer data
__distribute_original_images
    distribute the original images to all subsets (train, val, test)
res_crop
    crop an image to the required size
__mask_test_images
    extract a mask for all testing images containing dca
__generate_augmented_dca_sets
    augment binary and realistic DCA onto the training and validation sets
__inpaint_testing_sets
    inpaint the testing sets with both NS and Telea DCA removal methods

Created on Wed Jun  8 11:10:02 2022

@author: Samuel Pewton
"""
import pandas as pd
import random as r
import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import realistic_dca as rd
import masking_process as masking
import dca_removal

def main():
    """Generate balanced dataset. Main method.
    
    """
    ###########################################################################################################
    ############################## MODIFIERS FOR USER #########################################################
    ##############################^^^^^^^^^^^^^^^^^^^^#########################################################
    ###########################################################################################################
                                                                                                           ####
    # SPECIFY PATH TO DIRECTORY CONTAINING .CSV FILES LOCATED ON REPOSITORY                                ####
    csv_home_path = r"C:/Users/Sam/Desktop/isic_balanced_dca_split/csv/"                                   ####
                                                                                                           ####
    # SPECIFY PATH TO THE UNBALANCED ISIC DATASET                                                          ####
    unbalanced_isic_dataset = r"C:/Users/Sam/Downloads/isic_cleaned_unbalanced_mel_oth_224x224/"           ####
                                                                                                           ####
    # SPECIFY PATH TO THE FITZPATRICK17K DATASET                                                           ####
    fitzpatrick17k_dataset = r"C:/Users/Sam/Downloads/fitzpatrick17k/"                                     ####
                                                                                                           ####
    # SPECIFY PATH TO SAVE NEW DATASET TO                                                                  ####
    save_path = r"C:/Users/Sam/Desktop/"                                                                   ####
                                                                                                           ####
    # SPECIFY NAME FOR NEW DATASET                                                                         ####
    new_dataset_name = "balanced_dca_split"                                                                ####
                                                                                                           ####
    # SPECIFY REQUIRED IMAGE SIZE - may not work as anything other than (224,224) - untested               ####
    image_size = (224,224)                                                                                 ####
                                                                                                           ####
    ## !! IMAGE INPAINTING TAKES TIME TO EXECUTE, DO YOU WISH TO CONDUCT IT? !! ##                         ####
    ## !!      FALSE =  INPAINTED TEST SETS TO BE MISSING FROM DATASET       !! ##                         ####
    inpaint_test_sets = True                                                                               ####
                                                                                                           ####
    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################
    
    # Random seed - original experiments set to 12
    r.seed(12)
    generate_dataset(csv_home_path, unbalanced_isic_dataset, fitzpatrick17k_dataset, save_path, new_dataset_name, image_size, inpaint_test_sets)

def generate_dataset(csv_path, isic_path, fitz_path, save_path, new_dataset_name, image_size, inpaint):
    """Generate the new balanced dataset
    
    Parameters
    ----------
    csv_path : str
        path to the .csv files to be used
    isic_path : str
        path to the unbalanced isic dataset root directory
    fitz_path : str
        path to the fitzpatrick17k directory
    save_path : str
        the path to the directory in which to save the dataset to
    new_dataset_name : str
        the name of the new dataset to be created
    image_size : tuple
        size of the image to save to the dataset - should have 2 values
    inpaint : bool
        to inpaint or not to inpaint
    
    Returns
    -------
    None.
    
    """
    # 1. create the folder structure..
    __create_folder_structure(save_path)
    
    # 2. distribute original images..
    image_lists = __distribute_original_images(csv_path, isic_path, fitz_path, save_path, new_dataset_name, image_size)
    
    # 3. generate all of the masks and save to masks path
    __mask_test_images(save_path, new_dataset_name)
    
    # 4. augment train/val sets with DCA
    __generate_augmented_dca_sets(image_lists, save_path + new_dataset_name)
    
    # 5. inpaint testing sets
    if inpaint:
        __inpaint_testing_sets(save_path, new_dataset_name)

    print("Dataset successfully generated.")

def __create_folder_structure(save_path, new_dataset_name = "balanced_dca_split"):
    """Create the directory structure for the new dataset.
    
    !! WARNING !!
    If a dataset of the same name exists in the save_path it will be replaced.
    User is prompted to enter Y or y to continue. Any other input will cause the 
    program to exit.
    
    Parameters
    ----------
    save_path : str
        the path to the directory in which to save the dataset to
    new_dataset_name : str
        the name of the new dataset to be created
    
    Returns
    -------
    None.
    
    """
    # First check if expected root folder already exists
    # if it does, remove it and then recreate it
    u_input = input("WARNING: If a dataset of the same name exists in the save_path it will be replaced.\n[Y] if you are sure? ")
    
    if u_input == "Y" or u_input == "y":
        if new_dataset_name in os.listdir(save_path + "/"):
            print("----------------\n\"" + save_path + "/" + new_dataset_name + "\" already exists. Deleting directory..")
            shutil.rmtree(save_path + new_dataset_name)
            print("Directory deleted\n----------------")
        
        print("Generating new directory structure..")
        os.mkdir(save_path + "/" + new_dataset_name)                               # root folder
        os.mkdir(save_path + "/" + new_dataset_name + "/data")                     # L1
        os.mkdir(save_path + "/" + new_dataset_name + "/masks")                    # L1
        os.mkdir(save_path + "/" + new_dataset_name + "/data/test")                # L2
        os.mkdir(save_path + "/" + new_dataset_name + "/data/train")               # L2
        os.mkdir(save_path + "/" + new_dataset_name + "/data/val")                 # L2
        os.mkdir(save_path + "/" + new_dataset_name + "/data/test/base")           # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/test/ns")             # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/test/telea")          # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/train/clean")         # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/train/dca_binary")    # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/train/dca_realistic") # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/val/clean")           # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/val/dca_binary")      # L3
        os.mkdir(save_path + "/" + new_dataset_name + "/data/val/dca_realistic")   # L3
        for directory in [save_path + "/" + new_dataset_name + "/data/test/base",  # L4
                          save_path + "/" + new_dataset_name + "/data/test/ns", 
                          save_path + "/" + new_dataset_name + "/data/test/telea"]:
            os.mkdir(directory + "/small")
            os.mkdir(directory + "/medium")
            os.mkdir(directory + "/large")
            os.mkdir(directory + "/oth")
        for sub_directory in [save_path + "/" + new_dataset_name + "/data/test/base" + "/small", 
                              save_path + "/" + new_dataset_name + "/data/test/base" + "/medium", 
                              save_path + "/" + new_dataset_name + "/data/test/base" + "/large", 
                              save_path + "/" + new_dataset_name + "/data/test/base" + "/oth",
                              save_path + "/" + new_dataset_name + "/data/test/ns" + "/small", 
                              save_path + "/" + new_dataset_name + "/data/test/ns" + "/medium", 
                              save_path + "/" + new_dataset_name + "/data/test/ns" + "/large", 
                              save_path + "/" + new_dataset_name + "/data/test/ns" + "/oth",
                              save_path + "/" + new_dataset_name + "/data/test/telea" + "/small", 
                              save_path + "/" + new_dataset_name + "/data/test/telea" + "/medium", 
                              save_path + "/" + new_dataset_name + "/data/test/telea" + "/large", 
                              save_path + "/" + new_dataset_name + "/data/test/telea" + "/oth",
                              save_path + "/" + new_dataset_name + "/data/train/clean",
                              save_path + "/" + new_dataset_name + "/data/train/dca_binary",
                              save_path + "/" + new_dataset_name + "/data/train/dca_realistic",
                              save_path + "/" + new_dataset_name + "/data/val/clean",
                              save_path + "/" + new_dataset_name + "/data/val/dca_binary",
                              save_path + "/" + new_dataset_name + "/data/val/dca_realistic"]:
            os.mkdir(sub_directory + "/mel")
            os.mkdir(sub_directory + "/oth")
        print("New directory structure successfully generated. \n----------------")
    else:
        print("Dataset generation cancelled.")
        exit()

def __distribute_original_images(csv_path, isic_path, fitz_path, save_path, new_dataset_name, image_size):
    """Distribute the images from the original datasets to the new dataset as labelled in the 
    relevent .csv files.
    
    Parameters
    ----------
    csv_path : str
        path to the .csv files to be used
    isic_path : str
        path to the unbalanced isic dataset root directory
    fitz_path : str
        path to the fitzpatrick17k directory
    save_path : str
        the path to the directory in which to save the dataset to
    new_dataset_name : str
        the name of the new dataset to be created
    image_size : tuple
        size of the image to save to the dataset - should have 2 values
    
    Returns
    -------
    list
        list of all image lists for the train and validation sets
    
    """
    isic_mel_dirs = [isic_path + "/train/mel/", isic_path + "/val/mel/"]
    isic_oth_dirs = [isic_path + "/train/oth/", isic_path + "/val/oth/"]
    
    image_lists = [] # stores lists of all image_lists from train/val
    
    for csv_file in os.listdir(csv_path):
        csv_data = pd.read_csv(csv_path + csv_file)
        image_names, image_origins = list(csv_data["img_name"]), list(csv_data["origin"])

        if csv_file[:3] == "tes":
            image_save_path = save_path + "/" + new_dataset_name + "/data/test/base/" + csv_file[5:-8] + "/" + csv_file[-7:-4] + "/" 
            print("Distributing images from \"" + csv_file + "\" to \"" + image_save_path + "\"\n...")
        elif csv_file[:3] == "tra":
            image_save_path = save_path + "/" + new_dataset_name + "/data/train/clean/" + csv_file[5:-8] + "/" + csv_file[-7:-4] + "/" 
            image_lists.append(image_names)
            print("Distributing images from \"" + csv_file + "\" to \"" + image_save_path + "\"\n...")
        elif csv_file[:3] == "val":
            image_save_path = save_path + "/" + new_dataset_name + "/data/val/clean/" + csv_file[5:-8] + "/" + csv_file[-7:-4] + "/" 
            image_lists.append(image_names)
            print("Distributing images from \"" + csv_file + "\" to \"" + image_save_path + "\"\n...")
            
        for i, image in enumerate(image_names):
            if image_origins[i][-4:] == "mel/":
                try:
                    loaded_image = Image.open(isic_mel_dirs[0] + image)
                except:
                    loaded_image = Image.open(isic_mel_dirs[1] + image)
                
            elif image_origins[i][-4:] == "oth/": 
                try:
                    loaded_image = Image.open(isic_oth_dirs[0] + image)
                except:
                    loaded_image = Image.open(isic_oth_dirs[1] + image)
                
            elif image_origins[i][-3] == "17k":
                loaded_image = Image.open(fitz_path + image)
            loaded_image = res_crop(np.asarray(loaded_image), image_size)
            loaded_image = Image.fromarray(loaded_image)
            loaded_image.save(image_save_path + "/" + image)
    print("Images successfully distributed from .csv files\n----------------")
    return image_lists

def res_crop(image, size = (224,224)):
    """Taken from Bill Cassidy's script for image ratio resizing.
    
    resize_images.py
    
    Parameters
    ----------
    image : np.ndarray
        image to resize
    size : tuple
        size to resize image to
    
    Returns
    -------
    np.ndarray
        resized image
    
    """
    width, height, _ = image.shape
    ratio = max(size[0]/width, size[1]/height)
    target_size_before_crop_keep_ratio = int(height * ratio), int(width * ratio)
    image = cv.resize(image, target_size_before_crop_keep_ratio)
    width, height,_ = image.shape
    left_corner = int(round(width/2)) - int(round(size[0]/2))
    top_corner = int(round(height/2)) - int(round(size[1]/2))
    image = image[left_corner:left_corner+size[0],top_corner:top_corner+size[1]]
    return image

def __mask_test_images(save_path, new_dataset_name):
    """Gather masks from all of the testing sets and save to the appropriate location
    in the dataset
    
    Parameters
    ----------
    save_path : str
        the path to the directory in which to save the dataset to
    new_dataset_name : str
        the name of the new dataset to be created
    
    Returns
    -------
    None.
    
    """
    
    test_directories = [save_path + "/" + new_dataset_name + "/data/test/base/large/mel/",
                 save_path + "/" + new_dataset_name + "/data/test/base/large/oth/",
                 save_path + "/" + new_dataset_name + "/data/test/base/medium/mel/",
                 save_path + "/" + new_dataset_name + "/data/test/base/medium/oth/",
                 save_path + "/" + new_dataset_name + "/data/test/base/oth/mel/",
                 save_path + "/" + new_dataset_name + "/data/test/base/oth/oth/",
                 save_path + "/" + new_dataset_name + "/data/test/base/small/mel/",
                 save_path + "/" + new_dataset_name + "/data/test/base/small/oth/"]
    
    mask_save_path = save_path + "/" + new_dataset_name + "/masks/"
    print("Extracting masks from test sets and saving to \"" + mask_save_path + "\"")
    
    for directory in test_directories:
        for image in os.listdir(directory):
            data = np.asarray(Image.open(directory + image))
            masking.save_mask(image, data, mask_save_path)
    
    print("Masks successfully extracted and saved.\n----------------")

def __generate_augmented_dca_sets(images, root):
    """Generate augmented DCA sets for both binary and realistic DCA.
    
    Parameters
    ----------
    images
        list of all image sets
    root
        path to root directory

    Returns
    -------
    None.
    """    
    mask_path = root + "/masks/"
    s = sum(1 for f in os.listdir(mask_path))
    
    # should remain in this order if .csv file names remain unchanged    
    clean_paths = ["/data/train/clean/mel/", "/data/train/clean/oth/", "/data/val/clean/mel/", "/data/val/clean/oth/"]
    binary_paths = ["/data/train/dca_binary/mel/", "/data/train/dca_binary/oth/", "/data/val/dca_binary/mel/", "/data/val/dca_binary/oth/"] 
    realistic_paths = ["/data/train/dca_realistic/mel/", "/data/train/dca_realistic/oth/", "/data/val/dca_realistic/mel/", "/data/val/dca_realistic/oth/"] 

    print("Augmenting training/validation images, this may take a while..")
    for j in range(len(clean_paths)):
        print("Augmenting: \"" + root + clean_paths[j] + "\"")
        for img in images[j]:
           # Select and load image and mask
           index = r.randrange(s)
           image = Image.open(root + clean_paths[j] + img)
           mask = Image.open(mask_path + os.listdir(mask_path)[index])
           
           ## BINARY
           augmented_image = rd.augment_binary_dca(np.asarray(image), np.asarray(mask))
           Image.fromarray(augmented_image).save(root + binary_paths[j] + img)
           
           ## REALISTIC
           mask = cv.imread(mask_path + os.listdir(mask_path)[index])
           mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
           augmented_image = rd.augment_dca(np.asarray(image), np.asarray(mask), blur_type = "erode")
           Image.fromarray(augmented_image).save(root + realistic_paths[j] + img)
    print("Images successfully augmented.\n----------------")

def __inpaint_testing_sets(save_path, new_dataset_name):
    """Inpaint the testing sets with both NS and Telea methods    
    
    Parameters
    ----------
    save_path : str
        the path to the directory in which to save the dataset to
    new_dataset_name : str
        the name of the new dataset to be created
    
    Returns
    -------
    None.
    
    """
    print("WARNING: Inpainting will take a long time to execute due to super resolution.\n Starting inpainting...")
        
    base_filepaths = [save_path + "/" + new_dataset_name + "/data/test/base/large/mel/",
                      save_path + "/" + new_dataset_name + "/data/test/base/medium/mel/",
                      save_path + "/" + new_dataset_name + "/data/test/base/small/mel/",
                      save_path + "/" + new_dataset_name + "/data/test/base/oth/mel/",
                      save_path + "/" + new_dataset_name + "/data/test/base/large/oth/",
                      save_path + "/" + new_dataset_name + "/data/test/base/medium/oth/",
                      save_path + "/" + new_dataset_name + "/data/test/base/small/oth/",
                      save_path + "/" + new_dataset_name + "/data/test/base/oth/oth/"]

    ns_target_filepaths = [save_path + "/" + new_dataset_name + "/data/test/ns/large/mel/",
                           save_path + "/" + new_dataset_name + "/data/test/ns/medium/mel/",
                           save_path + "/" + new_dataset_name + "/data/test/ns/small/mel/",
                           save_path + "/" + new_dataset_name + "/data/test/ns/oth/mel/",
                           save_path + "/" + new_dataset_name + "/data/test/ns/large/oth/",
                           save_path + "/" + new_dataset_name + "/data/test/ns/medium/oth/",
                           save_path + "/" + new_dataset_name + "/data/test/ns/small/oth/",
                           save_path + "/" + new_dataset_name + "/data/test/ns/oth/oth/"]

    telea_target_filepaths = [save_path + "/" + new_dataset_name + "/data/test/telea/large/mel/",
                              save_path + "/" + new_dataset_name + "/data/test/telea/medium/mel/",
                              save_path + "/" + new_dataset_name + "/data/test/telea/small/mel/",
                              save_path + "/" + new_dataset_name + "/data/test/telea/oth/mel/",
                              save_path + "/" + new_dataset_name + "/data/test/telea/large/oth/",
                              save_path + "/" + new_dataset_name + "/data/test/telea/medium/oth/",
                              save_path + "/" + new_dataset_name + "/data/test/telea/small/oth/",
                              save_path + "/" + new_dataset_name + "/data/test/telea/oth/oth/"]
    
    for i in range(len(base_filepaths)):
        print("Inpainting: \"" + base_filepaths[i] + "\"")
        for img in os.listdir(base_filepaths[i]):
            image = np.asarray(Image.open(os.path.join(base_filepaths[i], img)))
            mask = np.asarray(masking.get_mask(image))
            
            inpainted_ns = dca_removal.remove_DCA(image, mask)
            inpainted_telea = dca_removal.remove_DCA(image, mask, 'inpaint_telea')
            
            Image.fromarray(inpainted_ns).save(os.path.join(ns_target_filepaths[i],img[:-4] + '.png'))
            Image.fromarray(inpainted_telea).save(os.path.join(telea_target_filepaths[i],img[:-4] + '.png'))
        print("\"" + base_filepaths[i] + "\" successfully inpainted.")
    print("Inpainting successful.\n----------------")

if __name__ == "__main__":
    main()