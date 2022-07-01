# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:54:53 2022

@author: Sam
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dca_removal import __calculate_reduction_rate
from isic_data import _image_alter_selection

def main():
    """Module testing main method
    
    """
    # Specify the image to apply DCA to
    img_name = "ISIC2019_0033636_mel"
    
    # Load image from file
    orig_image = cv2.imread(r"..\\Data\\dca_split_unbalanced_224x224\\val\\" + img_name[-3:] + "\\" + img_name + ".jpg")
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    # Load mask from file
    #mask = cv2.imread(r"D:\OneDrive - MMU\Complete_Final_Project\Lesion_Classification\Data\DCA_Masks\train\mel\ISIC2019_0053828_mel_MASK.png")
    mask = cv2.imread(r"D:\OneDrive - MMU\Complete_Final_Project\Lesion_Classification\Data\DCA_Masks\train\mel\ISIC2019_0025265_mel_MASK.png")
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Call the method
    #augmented_image = augment_standardised_dca(orig_image, blur_type = "erode")
    augmented_image = augment_binary_dca(orig_image, mask)
    plt.imshow(augmented_image)

def extract_mask_vars(mask):
    """Extract the centre point and the radius of the dca in the mask
    
    """
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # extract the 'contour' location into coordinates
    contour_region = np.column_stack(np.where(mask < 1))
    
    # enclosing circle back on the 'contour'
    (x,y), radius = cv2.minEnclosingCircle(contour_region)
    center = (int(x), int(y))
    
    return center, radius


def augment_dca(image, mask, kernel_size = (9, 9), sigma = 20, blur_type = "dilate", blur_depth = 10):
    """Augment a dca onto an image.
    
    blurring types:
        * erode = blurs into image area
        * dilate = blurs into dca area
    
    points to note:
        * when augmenting a dca < 1% of the image, dilate will not show DCA
    
    Parameters
    ----------
    image
        image to inpaint - must be in RGB format
    mask
        mask of dca to apply to image - must be in RGB format
    kernel_size
        size of the kernel to blur the image
    sigma
        sigma value for image blurring
    blur_type
        erode for blurring inside of the image area, dilate for blurring inside DCA area
    blur_depth
        radius size to blur
    """
    image = _image_alter_selection(image)
    if blur_type == "erode":
    
        # Mask the image
        masked_image = image.copy()
        masked_image[mask.astype(bool)] = 0
    
        # Gaussian blur the image as is
        blurred_img = cv2.GaussianBlur(src = masked_image, ksize = kernel_size, sigmaX = sigma,sigmaY = sigma, borderType = cv2.BORDER_DEFAULT)

        # extract circle properties and reduce radius
        center, radius = extract_mask_vars(mask)
        radius = int(radius) - blur_depth
        
        # create the new masking circle using the centre point and the new radius
        this_contour = cv2.circle(np.ones(mask.shape),center,radius,(0,0,0),-1)
    
        # combine the blurred image with the original image
        output = np.where(this_contour!=np.array([0, 0, 0]), blurred_img, image)
        #plt.imshow(output)
        
    elif blur_type == "dilate":
        
        # extract circle properties and increase radius
        center, radius = extract_mask_vars(mask) 
        radius = int(radius) + blur_depth
        
        # draw new masking circle
        this_contour = cv2.circle(np.ones(mask.shape),center,radius,(0,0,0),-1)
        
        # mask the image with the new contour
        masked_image = image.copy()
        masked_image[this_contour.astype(bool)] = 0
        
        # blur the image
        blurred_img = cv2.GaussianBlur(src = masked_image, ksize = kernel_size, sigmaX = sigma,sigmaY = sigma, borderType = cv2.BORDER_DEFAULT)
        
        # combine the blurred image with the original image
        output = np.where(mask!=np.array([0, 0, 0]), blurred_img, image)
        #plt.imshow(output)
    return output

def augment_standardised_dca(image, kernel_size = (9, 9), sigma = 20, blur_type = "dilate", blur_depth = 10):
    """
    
    """    
    # DEFINE THE CENTER AND RADIUS - CREATE STANDARD MASK
    center = (112, 112) # - center of 224x224 image, radius to edge
    radius = 112
    mask = cv2.circle(np.ones((224,224,3)),center,radius,(0,0,0),-1)
    mask = mask.astype('float32') 
    
    return augment_dca(image, mask, kernel_size, sigma, blur_type, blur_depth)

def augment_binary_dca(image, mask):
    """Augment a DCA without gaussian blur.
    
    """
    image = _image_alter_selection(image)
    output = image.copy()
    output[mask.astype(np.bool)] = 0
    return output
    
    

    

if __name__ == '__main__':
    main()