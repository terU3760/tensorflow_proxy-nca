

import json

import keras
backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils


import tensorflow as tf





# __repr__ may contain `\n`, json replaces it by `\\n` + indent
json_dumps = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def load_config(config_name = 'config.json'):
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)
    return config





def smooth_labels(T, nb_classes, smoothing_const = 0.1):

    temp_T = tf.cast( T , tf.float32 )
    temp_T = temp_T * (1 - smoothing_const)
    temp_T += smoothing_const / backend.cast( nb_classes - 1 , "float32" )


    return temp_T

