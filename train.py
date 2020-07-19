

import logging, imp

import utils

import tensorflow as tf

import custom_evaluation

import custom_checkpoint

from keras.preprocessing.image import ImageDataGenerator

from utils import smooth_labels


import keras


import os
os.putenv("OMP_NUM_THREADS", "8")

import numpy as np


import time
import argparse
import json
import random

from net.keras_vgg19_based_proxy_nca_test import VGG19_with_ProxyNCA




backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils






seed = 0
random.seed(seed)
np.random.seed(seed)




def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='No Fuss Distance Metric Learning using Proxies Training With Tensorflow and Keras')


parser.add_argument('--dataset',
    default='cars',
    help = 'Path to root CARS folder, containing the images folder.'
)

parser.add_argument('--config',
    default='config_cars_tensorflow.json',
    help = 'Path to root CARS folder, containing the images folder.'
)

parser.add_argument('--embedding-size', default = 64, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to InceptionV2.'
)

parser.add_argument('--batch_size', default=16, type=int,
    dest = 'sz_batch' ,
    help='Batch size for training'
)


parser.add_argument('--val_batch_size', default = 64, type = int,
    dest = 'val_sz_batch' ,
    help='Batch size for validation'
)


parser.add_argument('--epochs', default = 150, type = int,
    dest = 'nb_epochs' ,
    help = 'Number of training epochs.'
)

parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)

parser.add_argument('--workers', default = 16, type = int,
    dest = 'nb_workers' ,
    help = 'Number of workers for dataloader.'
)

parser.add_argument('--resume', default=None, type=str,
    help = 'Checkpoint state_dict file to resume training from'
)

parser.add_argument('--samples_per_epoch', default=800, type=int,
    help = 'Training samples used per epoch'
)

parser.add_argument('--val_samples', default=50, type=int,
    dest = 'nb_val_samples' ,
    help = 'Samples used in validation each time'
)

parser.add_argument('--save_path_filename', default='weights/cars_epoch:{epoch:02d}_loss:{loss:.2f}_val_loss:{val_loss:.2f}',
    help = 'Directory for saving checkpoint models'
)

parser.add_argument('--save_interval', default=5, type=int,
    help = 'The interval of iterations to save pth model'
)

parser.add_argument('--start_epoch', default=0, type=int,
    help = 'Start from this epoch to train the model'
)

parser.add_argument('--log-filename', default = 'example',
    help = 'Name of log file.')

parser.add_argument( '--log_loss_output_filename' , default = "loss_output_log.csv",
    help = 'Name of output of loss log filename')

parser.add_argument( '--lr' , default = 0.1 , type = float ,
    help = 'The initial learning rate of the SGD algorithm.' )

parser.add_argument( '--weight_decay' , default = 2.0e-5 , type = float ,
    help = 'The weight decay of the SGD algorithm.' )

parser.add_argument( '--momentum' , default = 0.0 , type = float ,
    help = 'The momentum of the SGD algorithm.' )

parser.add_argument( '--custom_evaluation_interval' , default = 1 , type = int ,
    help = 'The interval of epoches for custom evaluation interval.' )


args = parser.parse_args()


config = utils.load_config(args.config)
root = config['dataset'][args.dataset]['root']


from utils import JSONEncoder, json_dumps

print(json_dumps(obj = config, indent=4, cls = JSONEncoder, sort_keys = True))
with open('log/' + args.log_filename + '.json', 'w') as x:
    json.dump(
        obj = config, fp = x, indent=4, cls = JSONEncoder, sort_keys = True
    )





imp.reload(logging)

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler( args.log_filename ),
        logging.StreamHandler()
    ]
)

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []





training_generator = ImageDataGenerator(
    fill_mode='reflect')

training_generator.config[ 'center_crop_ratio' ] = 0.9
training_generator.config[ 'mean' ] = config[ 'transform_parameters' ][ 'mean' ]
training_generator.config[ 'std' ] = config[ 'transform_parameters' ][ 'std' ]
training_generator.config[ 'size' ] = ( 299 , 299 )





training_generator = training_generator.flow_from_directory( os.path.join( root , "for_training" ) , class_mode = "categorical" , batch_size = args.sz_batch , target_size = ( 299 , 299 ) )


validation_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    fill_mode='reflect')

validation_generator.config[ 'size' ] = ( 299 , 299 )

validation_generator = validation_generator.flow_from_directory( os.path.join( root , "for_evaluation" ) , class_mode = "categorical" , batch_size = args.val_sz_batch , target_size = ( 299 , 299 ) )





def nca_loss( y_true , y_pred ):

    y_true_smoothed = smooth_labels( y_true , backend.shape( y_pred )[ 1 ] )

    loss = backend.sum( - y_true_smoothed * tf.nn.log_softmax( - y_pred ) , axis = -1 )

    return backend.mean( loss )





def train():
    




    checkpoint = custom_checkpoint.EpochCheckpoint( args.save_path_filename , save_interval = args.save_interval , start_at = args.start_epoch )

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model = VGG19_with_ProxyNCA( weights = args.resume , classes = len( config['dataset'][args.dataset]['classes']['train'] ) , input_shape = ( 3 , 299 , 299 ) , backend = backend , layers = layers , models = models , utils = keras_utils )
    else:
        model = VGG19_with_ProxyNCA( weights = None , classes = len( config['dataset'][args.dataset]['classes']['train'] ) , input_shape = ( 3 , 299 , 299 ) , backend = backend , layers = layers , models = models , utils = keras_utils )

    model.summary()

    sgd_optimizer = keras.optimizers.SGD( lr = args.lr , decay = args.weight_decay , momentum = args.momentum , nesterov = True )

    model.compile( optimizer = sgd_optimizer , loss = nca_loss , run_eagerly = True )

    ival = custom_evaluation.IntervalEvaluation( model , validation_data = validation_generator , interval = args.custom_evaluation_interval )

    history_object = model.fit_generator( generator = training_generator ,
                    samples_per_epoch = args.samples_per_epoch ,
                    nb_epoch = args.nb_epochs - args.start_epoch ,
                    validation_data = validation_generator ,
                    nb_val_samples = args.nb_val_samples ,
                    callbacks = [ ival , checkpoint ] ,
                    nb_worker = args.nb_workers ,
                    verbose = 1 )





    loss_history = history_object.history[ "loss" ]
    numpy_loss_history = np.array( loss_history )
    np.savetxt( args.log_loss_output_filename , numpy_loss_history , delimiter = "," )





if __name__ == '__main__':
    train()

