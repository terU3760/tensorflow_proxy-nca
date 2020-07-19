import tensorflow as tf
import random

import math

import numpy as np

import PIL.Image as Image
import warnings


def get_params(img, scale, ratio):
    """Get parameters for ``crop`` for a random sized crop.
    
    Args:
        img (PIL Image): Image to be cropped.
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
    
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    area = img.shape[2] * img.shape[1]

    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))
        
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if w <= img.shape[2] and h <= img.shape[1]:
            i = random.randint(0, img.shape[1] - h)
            j = random.randint(0, img.shape[2] - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = img.shape[2] / img.shape[1]
    if (in_ratio < min(ratio)):
        w = img.shape[2]
        h = w / min(ratio)
    elif (in_ratio > max(ratio)):
        h = img.shape[1]
        w = h * max(ratio)
    else:  # whole image
        w = img.shape[2]
        h = img.shape[1]
    i = (img.shape[1] - h) // 2
    j = (img.shape[2] - w) // 2
    return i, j, h, w



def to_float(img , **kwargs):
    return img / 255.0





def center_crop(img, center_crop_ratio , **kwargs):
    h = img.shape[0]
    w = img.shape[1]
    central_fraction = float( w * center_crop_ratio ) / float( h )
    
    
    
    
    
    img_shape = img.shape
    
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = math.floor( img_shape[0] / fraction_offset )
    bbox_w_start = math.floor( img_shape[1] / fraction_offset )

    bbox_h_size = img_shape[0] - bbox_h_start * 2
    bbox_w_size = img_shape[1] - bbox_w_start * 2
    
    new_image = img[ bbox_h_start : bbox_h_size , bbox_w_start : bbox_w_size ]
    
    
    
    
    
    return np.array( Image.fromarray( np.asarray( new_image ) , mode = "RGB" ).resize( ( h , w ) , resample = Image.BILINEAR ) )





def random_resized_crop(img, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, sync_seed=None, **kwargs):
    
    
    if isinstance(size, tuple):
        size = size
    else:
        size = (size, size)
    if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
        warnings.warn("range should be of kind (min, max)")

    
    i, j, h, w = get_params(img, scale, ratio)
    
    img_cropped = img[ : , j:j+h , i:i+w ]
    

    return np.array( Image.fromarray( np.transpose( np.asarray( img_cropped ) , ( 1 , 2 , 0 ) ) , mode = "RGB" ).resize( ( size[ 1 ] , size[ 0 ] ) , resample = interpolation ) )




    
def random_flip_left_right(img, seed=None, **kwargs):
    
    
    prob_to_flip = random.random()
    
    if( prob_to_flip  < 0.5):
        
        new_img = img[ : , ::-1 , : ]

    else:
        
        new_img = img
    
    return new_img
    




def normalize(img, mean, std, **kwargs):

    return (img - mean) / std



def permute_to_right_shape(img , **kwargs):

    return np.transpose( img , ( 2 , 0 , 1 ) )




def only_resize(img , size , **kwargs):

    return np.array( Image.fromarray( np.transpose( np.asarray( img ) , ( 1 , 2 , 0 ) ) , mode = "RGB" ).resize( ( size[ 1 ] , size[ 0 ] ) , resample = Image.BILINEAR ) )

