#importing general packages to be used
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

"""
This module contains useful functions for image analasis,
and commmonly used packages.
"""



def import_images(path: str) -> list:
    """
    function to load images into a list 
    path: folder leading to a path of one or several images
    ---
    returns list of images inside folder, 
    or just an image if there is only one image in path.
    """
    images_inside_path = os.listdir(path)
    if ('.DS_Store' in images_inside_path):
        images_inside_path.remove('.DS_Store')

    image_list = []
    for image in images_inside_path:
        img_path = os.path.join(path, image)
        img = np.array(Image.open(img_path))
        image_list.append(img)
        image_array = image_list

    if len(image_array) > 1:
        return image_array
    else:
        return image_array[0]



def save_declared_figure(task_number: int, file_name: str) -> 0:
    """
    Saves a figure to a set file path

    task_number: the task number
    file_name: name to be saved as 
    format: handles the axises for spatial or fourier domain
    v_min_max: if the intensity colorbar should be set in a fixed range of 0,255 or not
    
    ---
    saves figure that to the a path result_images/task_[task_number] 
    """

    #getting working directory and making path
    wd = os.getcwd()
    directory_savepath = os.path.join(wd, "result_images", "task_" + str(task_number))

    #checking if path exist if not makes it
    if os.path.exists(directory_savepath) == False:
        os.makedirs(directory_savepath)

    #saving image
    image_savepath = os.path.join(directory_savepath, file_name + ".png")
    plt.savefig(image_savepath, format = "png")
    plt.close()


    return 0



def im2double(image):
    """
    image: image to transform to 16 bit

    Function transforms 8 bit images to 16 bit
    and normalizes intensities to be in range [0,1] instead of [0, 255]
    """
    image_array = np.array(image)
    image_array_double = image_array.astype(np.float64)
    image_array_double /= 255
    
    return(image_array_double)

def im2uint8(image):
    """
    image: to transform to 8 bit

    Function transforms 16 bit image to 8 bit
    rescales the intensities back to the range [0, 255] from [0, 1]
    """
    image_array = np.array(image)
    image_array *= 255
    image_array_uint8 =image_array.astype(np.uint8)
    
    return image_array_uint8

def plot_comp(image, transformed_image):
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    org_plot = ax1.imshow(image, cmap = "gray")
    ax1.set_title("original")
    plt.colorbar(org_plot, ax = ax1)
    trans_plot = ax2.imshow(transformed_image, cmap = "gray")
    ax2.set_title("transformed")
    plt.colorbar(trans_plot, ax = ax2)

if __name__ == "__main__":
    pass
    