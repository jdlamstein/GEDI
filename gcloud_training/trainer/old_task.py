#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:59:27 2019

@author: joshlamstein
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GEDI. 

To do:
    Explore tensorboard features - embedding, graphs, fit_generator doesn't do histograms. 
    Save tf model
    
    
"""
import argparse
import numpy as np
import trainer.param as param
import os
from trainer.model import deep
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.lib.io import file_io
from io import BytesIO
from keras.callbacks import Callback
import glob

from trainer.ops import apply_reshape, apply_crop, apply_rescale, copy_file_to_gcs, to_savedmodel


class ContinuousEval(Callback):
    """Continuous eval callback to evaluate the checkpoint once every so many epochs."""
    def __init__(self,
               eval_frequency,
               eval_files,
               learning_rate,
               job_dir,
               steps=1000):
        self.eval_files = eval_files
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.steps = steps
    
    def on_epoch_begin(self, epoch, logs={}):
        """Compile and save model."""
        if epoch > 0 and epoch % self.eval_frequency == 0:
            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith('gs://'):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
                checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
#        u_model = load_model(checkpoints[-1])
#        u_model = compile_model(u_model, self.learning_rate)
#        loss, acc = u_model.evaluate_generator(
#            generator(self.eval_files, hparams.eval_batch_size),
#            steps=self.steps)
#        print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
#            epoch, loss, acc, u_model.metrics_names))
                if self.job_dir.startswith('gs://'):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))

def train_and_evaluate(hparams):
  
    f = BytesIO(file_io.read_file_to_string(hparams.X_train_str, binary_mode=True))
    X_train = np.load(f)
    X_train = X_train[X_train.files[0]]
    f = BytesIO(file_io.read_file_to_string(hparams.Y_train_str, binary_mode=True))
    Y_train = np.load(f)
    Y_train = Y_train[Y_train.files[0]]
    f = BytesIO(file_io.read_file_to_string(hparams.X_val_str, binary_mode=True))
    X_val = np.load(f)
    X_val = X_val[X_val.files[0]]
    f = BytesIO(file_io.read_file_to_string(hparams.Y_val_str, binary_mode=True))
    Y_val = np.load(f)
    Y_val = Y_val[Y_val.files[0]]
    f = BytesIO(file_io.read_file_to_string(hparams.X_test_str, binary_mode=True))
    X_test = np.load(f)
    X_test = X_test[X_test.files[0]]
    f = BytesIO(file_io.read_file_to_string(hparams.Y_test_str, binary_mode=True))
    Y_test = np.load(f)
    Y_test = Y_test[Y_test.files[0]]
    
    p = param.param()
    
    
    X_train = apply_crop(X_train, p.input_shape)
    Y_train = apply_crop(Y_train, p.input_shape)
    X_test = apply_crop(X_test, p.input_shape)
    Y_test = apply_crop(Y_test, p.input_shape)
    X_val = apply_crop(X_val, p.input_shape)
    Y_val = apply_crop(Y_val, p.input_shape)

    X_train = apply_rescale(X_train, p.input_shape)
    Y_train = apply_rescale(Y_train, p.input_shape)
    X_test = apply_rescale(X_test, p.input_shape)
    Y_test = apply_rescale(Y_test, p.input_shape)
    X_val = apply_rescale(X_val, p.input_shape)
    Y_val = apply_rescale(Y_val, p.input_shape)
        
    deepnet = deep()
    model = deepnet.cnn(input_shape = p.input_shape, output_shape =  p.output_shape)
    try:
        os.makedirs(hparams.job_dir)
    except:
        pass

    checkpointer = ModelCheckpoint(os.path.join(hparams.job_dir, hparams.checkpoint_file),
                                   monitor = 'val_loss', verbose = 0, save_best_only = True)

    # Continuous eval callback.
    evaluation = ContinuousEval(hparams.eval_frequency, (X_test, Y_test),
                              hparams.learning_rate, hparams.job_dir)
    
    tb = TensorBoard(log_dir=os.path.join(hparams.job_dir, 'logs'), 
                     histogram_freq = 0,
                     write_graph = True, 
                     embeddings_freq = 0
            )
    
    callbacks = [checkpointer, evaluation, tb]
    
    History = model.fit(X_train, Y_train, 
                           validation_data = (X_val, Y_val),
                           batch_size=hparams.train_batch_size, 
                           epochs=hparams.epochs, 
                           callbacks=callbacks, 
                           steps_per_epoch = hparams.train_steps,
                           validation_steps = hparams.eval_steps)
    
    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if hparams.job_dir.startswith('gs://'):
        model.save(p.MODEL_hdf5)
        copy_file_to_gcs(hparams.job_dir, p.MODEL_hdf5)
    else:
        model.save(os.path.join(hparams.job_dir, p.MODEL_hdf5))

    # Convert the Keras model to TensorFlow SavedModel.
    to_savedmodel(model, os.path.join(hparams.job_dir, 'export'))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    p = param.param()
    parser.add_argument(
            '--X-train-str',
            nargs='+',
            help='Training file local or GCS',
            default='gs://rebelbase/GCS/xtrain.npz')
    parser.add_argument(
            '--Y-train-str',
            nargs='+',
            help='Training file local or GCS',
            default='gs://rebelbase/GCS/ytrain.npz')
    parser.add_argument(
            '--X-val-str',
            nargs='+',
            help='Training file local or GCS',
            default='gs://rebelbase/GCS/xval.npz')
    parser.add_argument(
            '--Y-val-str',
            nargs='+',
            help='Training file local or GCS',
            default='gs://rebelbase/GCS/yval.npz')
    parser.add_argument(
            '--X-test-str',
            nargs='+',
            help='Training file local or GCS',
            default='gs://rebelbase/GCS/xtest.npz')
    parser.add_argument(
            '--Y-test-str',
            nargs='+',
            help='Training file local or GCS',
            default='gs://rebelbase/GCS/ytest.npz')
    parser.add_argument(
            '--job-dir',
            type=str,
            help='GCS or local dir to write checkpoints and export model',
            default='/tmp/nuclei-keras')
    parser.add_argument(
            '--tb-dir',
            type=str,
            help='GCS or local dir to for tensorboard',
            default='/tmp/nuclei-keras/logs')
    parser.add_argument(
            '--train-steps',
            type=int,
            default=p.TRAIN_STEPS,
            help="""\
        Maximum number of training steps to perform
        Training steps are in the units of training-batch-size.
        So if train-steps is 500 and train-batch-size if 100 then
        at most 500 * 100 training instances will be used to train.""")
    parser.add_argument(
            '--eval-steps',
            help='Number of steps to run evalution for at each checkpoint',
            default=p.EVAL_STEPS,
            type=int)
    parser.add_argument(
            '--train-batch-size',
            type=int,
            default=p.TRAIN_BATCH_SIZE,
            help='Batch size for training steps')
    parser.add_argument(
            '--eval-batch-size',
            type=int,
            default=p.EVAL_BATCH_SIZE,
            help='Batch size for evaluation steps')
    parser.add_argument(
            '--learning-rate',
            type=float,
            default=p.LEARNING_RATE,
            help='Learning rate for model')
    parser.add_argument(
            '--eval-frequency',
            default=p.EVAL_FREQUENCY,
            help='Perform one evaluation per n epochs')
    parser.add_argument(
            '--eval-num-epochs',
            type=int,
            default=p.EVAL_NUM_EPOCHS,
            help='Number of epochs during evaluation')
    parser.add_argument(
            '--epochs',
            type=int,
            default=p.EPOCHS,
            help='Maximum number of epochs on which to train')
    parser.add_argument(
            '--checkpoint-epochs',
            type=int,
            default=p.CHECKPOINT_EPOCHS,
            help='Checkpoint per n training epochs')
    parser.add_argument(
            '--checkpoint-file',
            type=str,
            default=p.CHECKPOINT_FILE,
            help='File name of checkpoint')
    
    args, _ = parser.parse_known_args()
    
    hparams = hparam.HParams(**args.__dict__)
    train_and_evaluate(hparams)