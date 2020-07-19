'''
Created on Oct 30, 2019

@author: ngaimanchow
'''
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer

"""VGG19 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""


import os


from .image_utils import get_submodules_from_kwargs

from .image_utils import _obtain_input_shape





WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')



    
    
def VGG19(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          **kwargs):
    """Instantiates the VGG19 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='vgg19')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model





import keras

layers = keras.layers
backend = keras.backend





class ProxyNCALayer_UseRegularizer_Wrong(layers.Layer):

    def __init__(self, embedding_size, classes, **kwargs):
        self.embedding_size = embedding_size
        self.classes = classes
        self.name = 'proxy_embeddings'
        super(ProxyNCALayer_UseRegularizer_Wrong, self).__init__(**kwargs)


    def add_loss(self):
        pass





    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        """Adds a weight variable to the layer.

        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.

        # Returns
            The created weight variable.
        """
        initializer = keras.initializations.get(initializer)
        if dtype is None:
            dtype = backend.floatx()
        weight = backend.variable(initializer(shape),
                            dtype=dtype,
                            name=name)

        if regularizer is not None:

            for i in range( self.classes ):
                one_proxy_l2_regularizer = keras.regularizers.l2(1.0)
                one_proxy_l2_regularizer.set_param( weight[i] )
                self.regularizers.append( one_proxy_l2_regularizer )

        if trainable:
            self.trainable_weights.append(weight)
        else:
            self.non_trainable_weights.append(weight)
        return weight


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        assert len(input_shape) == 2

        
        self.kernel = self.add_weight(name='proxy_embeddings',
                                      shape=( self.classes , self.embedding_size ),
                                      initializer='uniform',
                                      regularizer=keras.regularizers.l2,
                                      trainable=True)

        super(ProxyNCALayer_UseRegularizer_Wrong, self).build(input_shape)  # Be sure to call this at the end
        
    
    def call(self, x, mask = None):

        x = backend.l2_normalize( x , axis = -1 )

        x_expand = backend.reshape( backend.tile( x , [ 1 , self.classes ] ) , [ backend.shape( x )[ 0 ] , self.classes , self.embedding_size ] )

        weights_expand = backend.reshape( backend.tile( self.kernel , ( backend.shape( x )[ 0 ] , 1 ) ) , [ backend.shape( x )[ 0 ] , self.classes , self.embedding_size ] )

        return backend.sqrt( backend.sum( backend.square( x_expand - weights_expand ) , axis = -1 ) )


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.classes )

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.classes )





##### writing my own ProxyNCA layer
class ProxyNCALayer(layers.Layer):

    def __init__(self, embedding_size, classes, **kwargs):
        self.embedding_size = embedding_size
        self.classes = classes
        self.name = 'proxy_embeddings'
        super(ProxyNCALayer, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        assert len(input_shape) == 2

        
        initializer = keras.initializations.get('one')
        

        self.kernel = backend.variable(initializer( ( self.classes , self.embedding_size ) ),
                            dtype= backend.floatx() ,
                            name= 'proxy_embeddings' )

        self.trainable_weights.append(self.kernel)


        super(ProxyNCALayer, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x, mask = None):


        x = 3 * backend.l2_normalize( x , axis = -1 )
        

        x_expand = backend.reshape( backend.tile( x , [ 1 , self.classes ] ) , [ backend.shape( x )[ 0 ] , self.classes , self.embedding_size ] )

        proxies_normalized = self.kernel

        proxies_normalized = 3 * backend.l2_normalize( proxies_normalized , axis = -1 )
        

        weights_expand = backend.reshape( backend.tile( proxies_normalized , ( backend.shape( x )[ 0 ] , 1 ) ) , [ backend.shape( x )[ 0 ] , self.classes , self.embedding_size ] )


        return backend.sum( backend.square( x_expand - weights_expand ), axis=-1 )


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.classes )


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.classes )







    

##### writing my own vgg19 with custom proxy nca layer
def VGG19_with_ProxyNCA(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          embedding_size=64,
          classes=1000,
          **kwargs):
    """Instantiates the VGG19 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=32,
                                      data_format="channels_first",
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = layers.Conv2D(64, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block1_conv1')(img_input)

    x = layers.Conv2D(64, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block2_conv1')(x)

    x = layers.Conv2D(128, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block3_conv1')(x)

    x = layers.Conv2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block3_conv2')(x)

    x = layers.Conv2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block3_conv3')(x)

    x = layers.Conv2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block4_conv1')(x)

    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block4_conv2')(x)

    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block4_conv3')(x)

    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block5_conv1')(x)

    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block5_conv2')(x)

    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block5_conv3')(x)

    x = layers.Conv2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    
    
    
    
    x = layers.Flatten(name='flatten')(x)

    
    
    
    
    x = layers.Dense(embedding_size, name='predictions')(x)
    

    ##### our own proxy nca layer
    x = ProxyNCALayer(embedding_size, classes, name = 'proxynca')(x)
    
    



    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='vgg19_based_proxy_nca')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model



