# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:59:32 2022

@author: Sam

Generate new dataset for 224x224

New dataset comprises of a clean training set with no DCA, a clean validation set
with no DCA and 4 testing sets.

Testing sets are for each category of DCA size.
"""
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image


def main():
    """Main method for module
    
    Define the directory to save all images into for the new dataset with split testing sets
    
    Define the dataset paths to the last project datasets - this is needed to extract
    original and inpainted images and save to correct location
    
    Datasets are then distributed into a single dataset in the following structure
    
    ...
    ├─ Data
    |   └─ dca_split_224x224
    |          ├─ train
    |          |    ├─ mel
    ...        |    └─ oth
         	   ├─ val
        	   |    ├─ mel
               |    └─ oth
               └─ test
         	        ├─ large
       	    	    ├─ large_ns
       	    	    ├─ large_telea
       	    	    ├─ medium
       	    	    ├─ medium_ns
       	    	    ├─ medium_telea
       	    	    ├─ oth
       	    	    ├─ oth_ns
        	   	    ├─ oth_telea
        	   	    ├─ small
        	   	    ├─ small_ns
        	   	    └─ small_telea

    The resulting dataset is unbalanced
    """
    # clear and create directory to distribute images into
    
    # SET THE DIR NAME HERE - MUST GO INSIDE DATA FOLDER
    dir_name = "dca_split_224x224"
    
    # DEFINE DATASET FILEPATHS (PREVIOUS PROJECT SETS)
    original_dataset_path = r"D:\\OneDrive\\Desktop\\Lesion_Classification\\Data\\train_balanced_224x224\\"
    ns_dataset_path = r"D:\\OneDrive\\Desktop\\Lesion_Classification\\Data\\train_balanced_224x224_inpainted_ns\\"
    telea_dataset_path = r"D:\\OneDrive\\Desktop\\Lesion_Classification\\Data\\train_balanced_224x224_inpainted_telea\\"
    
    generate_dca_split_set(dir_name, original_dataset_path, ns_dataset_path, telea_dataset_path)

def generate_dca_split_set(dir_name, original_dataset_path, ns_dataset_path, telea_dataset_path):           
    """See main docstring
    
    """
    if dir_name in os.listdir(r"..\\Data\\"):
        shutil.rmtree(r"..\\Data\\" + dir_name, ignore_errors=True)
    create_dir(dir_name)
    
    # load the annotation files from data\\annotations dir - remove the file extension and _MASK from all image names
    # this is for comparison against the image names to make it easier
    dca_t_mel_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_train_mel.csv")
    dca_t_mel_csv['Image_Name'] = [x[:-9] for x in dca_t_mel_csv['Image_Name']]
    dca_t_oth_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_train_oth.csv")
    dca_t_oth_csv['Image_Name'] = [x[:-9] for x in dca_t_oth_csv['Image_Name']]
    dca_v_mel_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_val_mel.csv")
    dca_v_mel_csv['Image_Name'] = [x[:-9] for x in dca_v_mel_csv['Image_Name']]
    dca_v_oth_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_val_oth.csv")
    dca_v_oth_csv['Image_Name'] = [x[:-9] for x in dca_v_oth_csv['Image_Name']]

    

    #print(dca_t_mel_csv)
    
    # train/mel
    for img in os.listdir(original_dataset_path + "train\\mel\\"):
        if img[:-4] in [x for x in dca_t_mel_csv["Image_Name"]]:
            # save the image to the correct test location
            im = Image.open(os.path.join(original_dataset_path + "train\\mel\\", img))
            ns_im = Image.open(os.path.join(ns_dataset_path + "train\\mel\\", img[:-3] + "png"))
            telea_im = Image.open(os.path.join(telea_dataset_path + "train\\mel\\", img[:-3] + "png"))
            
            if list(dca_t_mel_csv["Small_DCA"][dca_t_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\small\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\small_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\small_telea\\mel\\" + img[:-3] + "png")
                
            elif list(dca_t_mel_csv["Medium_DCA"][dca_t_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\medium\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_telea\\mel\\" + img[:-3] + "png")
                
                
            elif list(dca_t_mel_csv["Large_DCA"][dca_t_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\large\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\large_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\large_telea\\mel\\" + img[:-3] + "png")
                
                
            elif list(dca_t_mel_csv["Oth"][dca_t_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\oth\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_telea\\mel\\" + img[:-3] + "png")
                        
        else:
            # save the image to the clean dataset location
            im = Image.open(os.path.join(original_dataset_path + "train\\mel\\", img))
            im.save(r"..\\Data\\dca_split_224x224\\train\\mel\\" + img)
            

    # train/oth
    for img in os.listdir(original_dataset_path + "train\\oth\\"):
        if img[:-4] in [x for x in dca_t_oth_csv["Image_Name"]]:
            # save the image to the correct test location
            im = Image.open(os.path.join(original_dataset_path + "train\\oth\\", img))
            ns_im = Image.open(os.path.join(ns_dataset_path + "train\\oth\\", img[:-3] + "png"))
            telea_im = Image.open(os.path.join(telea_dataset_path + "train\\oth\\", img[:-3] + "png"))
            
            if list(dca_t_oth_csv["Small_DCA"][dca_t_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\small\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\small_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\small_telea\\oth\\" + img[:-3] + "png")
                
            elif list(dca_t_oth_csv["Medium_DCA"][dca_t_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\medium\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_telea\\oth\\" + img[:-3] + "png")
                
                
            elif list(dca_t_oth_csv["Large_DCA"][dca_t_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\large\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\large_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\large_telea\\oth\\" + img[:-3] + "png")
                                
            elif list(dca_t_oth_csv["Oth"][dca_t_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\oth\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_telea\\oth\\" + img[:-3] + "png")
                                
        else:
            # save the image to the clean dataset location
            im = Image.open(os.path.join(original_dataset_path + "train\\oth\\", img))
            im.save(r"..\\Data\\dca_split_224x224\\train\\oth\\" + img)
            

    # val/mel
    for img in os.listdir(original_dataset_path + "val\\mel\\"):
        if img[:-4] in [x for x in dca_v_mel_csv["Image_Name"]]:
            # save the image to the correct test location
            im = Image.open(os.path.join(original_dataset_path + "val\\mel\\", img))
            ns_im = Image.open(os.path.join(ns_dataset_path + "val\\mel\\", img[:-3] + "png"))
            telea_im = Image.open(os.path.join(telea_dataset_path + "val\\mel\\", img[:-3] + "png"))
            
            if list(dca_v_mel_csv["Small_DCA"][dca_v_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\small\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\small_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\small_telea\\mel\\" + img[:-3] + "png")
                
            elif list(dca_v_mel_csv["Medium_DCA"][dca_v_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\medium\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_telea\\mel\\" + img[:-3] + "png")
                
            elif list(dca_v_mel_csv["Large_DCA"][dca_v_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\large\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\large_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\large_telea\\mel\\" + img[:-3] + "png")
                
            elif list(dca_v_mel_csv["Oth"][dca_v_mel_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\oth\\mel\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_ns\\mel\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_telea\\mel\\" + img[:-3] + "png")
                
        else:
            # save the image to the clean dataset location
            im = Image.open(os.path.join(original_dataset_path + "val\\mel\\", img))
            im.save(r"..\\Data\\dca_split_224x224\\val\\mel\\" + img)
            
    # val/oth
    for img in os.listdir(original_dataset_path + "val\\oth\\"):
        if img[:-4] in [x for x in dca_v_oth_csv["Image_Name"]]:
            # save the image to the correct test location
            im = Image.open(os.path.join(original_dataset_path + "val\\oth\\", img))
            ns_im = Image.open(os.path.join(ns_dataset_path + "val\\oth\\", img[:-3] + "png"))
            telea_im = Image.open(os.path.join(telea_dataset_path + "val\\oth\\", img[:-3] + "png"))
            
            if list(dca_v_oth_csv["Small_DCA"][dca_v_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\small\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\small_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\small_telea\\oth\\" + img[:-3] + "png")
                
            elif list(dca_v_oth_csv["Medium_DCA"][dca_v_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\medium\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\medium_telea\\oth\\" + img[:-3] + "png")
                
            elif list(dca_v_oth_csv["Large_DCA"][dca_v_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\large\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\large_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\large_telea\\oth\\" + img[:-3] + "png")
                
            elif list(dca_v_oth_csv["Oth"][dca_v_oth_csv["Image_Name"] == img[:-4]])[0] == 1:
                im.save(r"..\\Data\\dca_split_224x224\\test\\oth\\oth\\" + img)
                ns_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_ns\\oth\\" + img[:-3] + "png")
                telea_im.save(r"..\\Data\\dca_split_224x224\\test\\oth_telea\\oth\\" + img[:-3] + "png")
                
        else:
            # save the image to the clean dataset location
            im = Image.open(os.path.join(original_dataset_path + "val\\oth\\", img))
            im.save(r"..\\Data\\dca_split_224x224\\val\\oth\\" + img)
    
    
        
def create_dir(dir_name):
    """Create the dataset directories
    
    """
    data_dir = r"..\\Data\\"
    os.mkdir(data_dir + dir_name)
    os.mkdir(data_dir + dir_name + "\\train")
    os.mkdir(data_dir + dir_name + "\\val")
    os.mkdir(data_dir + dir_name + "\\test")
    os.mkdir(data_dir + dir_name + "\\train\\mel")
    os.mkdir(data_dir + dir_name + "\\train\\oth")
    os.mkdir(data_dir + dir_name + "\\val\\mel")
    os.mkdir(data_dir + dir_name + "\\val\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\small")
    os.mkdir(data_dir + dir_name + "\\test\\medium")
    os.mkdir(data_dir + dir_name + "\\test\\large")
    os.mkdir(data_dir + dir_name + "\\test\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\small\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\medium\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\large\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\oth\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\small\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\medium\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\large\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\oth\\oth")
    
    os.mkdir(data_dir + dir_name + "\\test\\small_ns")
    os.mkdir(data_dir + dir_name + "\\test\\medium_ns")
    os.mkdir(data_dir + dir_name + "\\test\\large_ns")
    os.mkdir(data_dir + dir_name + "\\test\\oth_ns")
    os.mkdir(data_dir + dir_name + "\\test\\small_ns\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\medium_ns\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\large_ns\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\oth_ns\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\small_ns\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\medium_ns\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\large_ns\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\oth_ns\\oth")
    
    os.mkdir(data_dir + dir_name + "\\test\\small_telea")
    os.mkdir(data_dir + dir_name + "\\test\\medium_telea")
    os.mkdir(data_dir + dir_name + "\\test\\large_telea")
    os.mkdir(data_dir + dir_name + "\\test\\oth_telea")
    os.mkdir(data_dir + dir_name + "\\test\\small_telea\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\medium_telea\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\large_telea\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\oth_telea\\mel")
    os.mkdir(data_dir + dir_name + "\\test\\small_telea\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\medium_telea\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\large_telea\\oth")
    os.mkdir(data_dir + dir_name + "\\test\\oth_telea\\oth")
    
    
if __name__ == '__main__':
    main()