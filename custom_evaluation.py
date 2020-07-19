

import evaluation


import logging

from keras.callbacks import Callback

import numpy as np

import keras


import datetime





backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils





class IntervalEvaluation(Callback):
    def __init__(self, model, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        
        self.validation_data = validation_data
        
        self.model = model
        
        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers])


    def on_epoch_end(self, epoch, logs={}):
        

        if epoch % self.interval == 0:

            get_intermediate_layer_output = backend.function( [ self.model.input ] ,
                                  [ self.model.get_layer( "predictions" ).output ] )
            y_given = []

            y_embedding = []
            nb_classes = 0
            
            
            
            
            
            print( "Before getting validation samples:" )
            print( datetime.datetime.now().time() )

            
            
            
            
            for i in range( 128 ):

                X_val_temp , y_val_temp = self.validation_data.next()

                nb_classes = y_val_temp.shape[ 1 ]

                y_given.append( y_val_temp )

                y_embedding_temp = get_intermediate_layer_output( [ X_val_temp ] )[ 0 ]
                y_embedding.append( y_embedding_temp )


            
            
            
            print( "After getting validation samples:" )
            print( datetime.datetime.now().time() )

            
            
            
            
            y_embedding = np.concatenate( y_embedding , axis = 0 )

            y_given = np.concatenate( y_given , axis = 0 )
            y_given_class_order = np.argsort( y_given , axis = -1 )

            y_given_class = np.transpose( y_given_class_order )[ -1 ]


            

            
            nmi = evaluation.calc_normalized_mutual_information(
                y_given_class,
                evaluation.cluster_by_kmeans(
                    y_embedding, nb_classes
                )
            )
            
            
            
            
            logging.info("NMI: {:.3f}".format(nmi * 100))


            # get predictions by assigning nearest 8 neighbors with euclidian
            Y = evaluation.assign_by_euclidian_at_k(y_embedding, y_given_class, 8)





            # calculate recall @ 1, 2, 4, 8
            recall = []
            for k in [1, 2, 4, 8]:
                r_at_k = evaluation.calc_recall_at_k( y_given_class , Y , k )
                recall.append( r_at_k )
                logging.info( "R@{} : {:.3f}".format( k , 100 * r_at_k ) )
            
            return nmi, recall

