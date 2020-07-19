


from net.keras_vgg19_based_proxy_nca_test import VGG19_with_ProxyNCA

import utils

import keras

import PIL.Image

import numpy as np

import math



backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils


def smallest_sort( a ):

    b = a.reshape( ( 6 * 6 ) )
    sorted = np.sort( b )

    c = np.ones( ( 6 * 6 ) ).reshape( ( 6 , 6 ) ) * math.inf

    for k in range( 15 ):

        for i in range( 6 ):
            for j in range( i + 1 , 6 ):

                if( abs( a[ i , j ] - sorted[ k ] ) < 1.0e-8 ):
                    c[ i , j ] = k + 1

    return c





import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"





config = utils.load_config( '../config_cars_tensorflow.json' )

model = VGG19_with_ProxyNCA( weights= '../weights/cars_epoch:150_loss:4.59_val_loss:4.59(lr=0.5,data_augmentation)' , classes = len(config['dataset']['cars']['classes']['train']) ,
                            input_shape = (3, 299, 299) , backend = backend , layers = layers , models = models , utils = keras_utils )

get_intermediate_layer_output = backend.function([model.input],
                                                 [model.get_layer("predictions").output])

y_embedding = []


for i in range( 1 , 7 ):

    single_car_image_file = "./" + str(i) + ".jpg"        # <----- this is the input image

    im = PIL.Image.open( single_car_image_file ).resize( ( 299 , 299 ) , PIL.Image.ANTIALIAS )

    X_val_temp = I = np.asarray( im ).transpose( 2 , 0 , 1 ).reshape( 1 , 3 , 299 , 299 )

    y_embedding_temp = get_intermediate_layer_output( [ X_val_temp ] )[ 0 ]

    y_embedding.append( y_embedding_temp )


dist_matrix = np.ones( ( 6 , 6 ) ) * math.inf


for i in range( 6 ):
    for j in range( i + 1 , 6 ):

        dist_matrix[ i , j ] = np.linalg.norm( y_embedding[ i ] - y_embedding[ j ] )



rank_matrix = smallest_sort( dist_matrix )


print( rank_matrix )