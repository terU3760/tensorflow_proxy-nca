'''
Created on Jan 16, 2020

@author: ngaimanchow
'''



from keras.callbacks import Callback
import os





class EpochCheckpoint(Callback):
    def __init__( self , path_and_filename , save_interval = 5 , start_at = 0 ):
        # call the parent constructor
        super( Callback , self ).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.path_and_filename = path_and_filename
        self.save_interval = save_interval
        self.current_epoch = start_at

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if ( self.current_epoch + 1 ) % self.save_interval == 0:
            '''
            p = os.path.sep.join([self.outputPath,
                "epoch_{}.hdf5".format(self.intEpoch + 1 , **logs)])
            '''
            self.model.save( self.path_and_filename.format( epoch = self.current_epoch + 1 , **logs ) , overwrite = True )

        # increment the internal epoch counter
        self.current_epoch += 1