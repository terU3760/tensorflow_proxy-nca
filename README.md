# About
A TensorFlow with Keras reimplementation of the paper [*No Fuss Distance Metric Learning using Proxies*](https://arxiv.org/pdf/1703.07464.pdf) as introduced by Google Research.

Take advantage of the a little recent network VGG19 with somewhat naive optimizer SGD. These two parts could be easily changed to any other compatible components.

Provided two training processes. One is direct and the other applies some data augmentation tricks. Both produce applicable results.

The dataset applied is [Cars 196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). But it could be easily changed to any other datasets under Keras framework.

The anaconda environment to run is provided as the environment.yaml file.

A custom version of Keras is included which is necessary for data augmentation.

#Train

To train directly, type:

```
python train.py
```

To train with data augmentation, type:

```
python train_data_augmentation.py
```

Since it is run under Keras framework, no need to mind the validation loss, a custom evaluation process is used.

#Results

Since using a different network VGG19 instead of [BN-Inception](http://arxiv.org/abs/1502.03167), the result is not listed here. But the base network could be very easily change back into it.