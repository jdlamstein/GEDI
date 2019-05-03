#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GEDI. 

To do:
    Freeze weights in model and train on top of VGG and Inception
    
    
"""
import argparse
import numpy as np
import trainer.param as param
import trainer.processing as pro
import os
from trainer.model import deep
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.contrib.training.python.training import hparam
from tensorflow.keras.callbacks import Callback
import glob
from trainer.ops import copy_file_to_gcs, to_savedmodel


class ContinuousEval(Callback):
    """Continuous eval callback to evaluate the checkpoint once every so many epochs."""
    def __init__(self,
               eval_frequency,
               learning_rate,
               job_dir,
               steps=1000):
#        self.eval_files = eval_files
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
  
#    image_batch, label = pro.create_dataset([hparams.X_train_str, hparams.Y_train_str])
    training_set = pro.tfdata_generator(filepaths = [hparams.train_str])
    validation_set = pro.tfdata_generator(filepaths = [hparams.val_str])
#    training_generator = pro._tfdata_generator(filepaths = [hparams.X_train_str, hparams.Y_train_str])
#    validation_generator = pro._tfdata_generator(filepaths = [hparams.X_val_str, hparams.Y_val_str])
#    image_input = Input( tensor=image_batch )
    nn = deep(network_type = p.network_type,input_shape = p.input_shape, output_shape = p.output_shape  )
#    model = deepnet.vgg(input_shape = p.input_shape, output_shape = p.output_shape )

    try:
        os.makedirs(hparams.job_dir)
    except:
        pass

    checkpointer = ModelCheckpoint(os.path.join(hparams.job_dir, hparams.checkpoint_file),
                                   monitor = 'val_loss', verbose = 0)

    # Continuous eval callback.
    evaluation = ContinuousEval(hparams.eval_frequency,
                              hparams.learning_rate, hparams.job_dir)
    
#    tb = TensorBoard(log_dir=os.path.join(hparams.job_dir, 'logs'), 
#                     histogram_freq = 0,
#                     write_graph = False)
    
    callbacks = [evaluation, checkpointer]
    
#    History = model.fit(epochs=hparams.epochs, 
#                           steps_per_epoch = hparams.train_steps)
#    model.fit(
#            steps_per_epoch=p.TRAIN_STEPS,
#            epochs=hparams.epochs,
#            callbacks = callbacks)
    
    nn.model.fit(training_set, validation_data=validation_set,
              steps_per_epoch = p.TRAIN_STEPS,
              validation_steps = p.VALIDATION_STEPS,
              epochs = hparams.epochs,
              callbacks = callbacks)
#    model.fit_generator(training_generator, 
#                        steps_per_epoch= p.TRAIN_STEPS, 
#                        epochs = hparams.epochs, 
#                        validation_data = validation_generator,
#                        validation_steps = p.VALIDATION_STEPS,
#                        workers = 0,
#                        callbacks = callbacks)
    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if hparams.job_dir.startswith('gs://'):
        nn.model.save(p.MODEL_hdf5)
        copy_file_to_gcs(hparams.job_dir, p.MODEL_hdf5)
    else:
        nn.model.save(os.path.join(hparams.job_dir, p.MODEL_hdf5))

    # Convert the Keras model to TensorFlow SavedModel.
    to_savedmodel(nn.model, os.path.join(hparams.job_dir, 'export'))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    p = param.param()
    parser.add_argument(
            '--train-str',
            nargs='+',
            help='Training file local or GCS',
            default=p.TRAIN_STR)
    parser.add_argument(
            '--val-str',
            nargs='+',
            help='Training file local or GCS',
            default=p.VAL_STR)
    parser.add_argument(
            '--test-str',
            nargs='+',
            help='Testin file local or GCS',
            default=p.TEST_STR)
    parser.add_argument(
            '--job-dir',
            type=str,
            help='GCS or local dir to write checkpoints and export model',
            default='/tmp/gedi-keras')
    parser.add_argument(
            '--tb-dir',
            type=str,
            help='GCS or local dir to for tensorboard',
            default='/tmp/gedi-keras/logs')
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
            default=0,
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
            default=0,
            help='Number of epochs during evaluation')
    parser.add_argument(
            '--epochs',
            type=int,
            default=p.EPOCHS,
            help='Maximum number of epochs on which to train')
    parser.add_argument(
            '--checkpoint-epochs',
            type=int,
            default=0,
            help='Checkpoint per n training epochs')
    parser.add_argument(
            '--checkpoint-file',
            type=str,
            default=p.CHECKPOINT_FILE,
            help='File name of checkpoint')
    
    args, _ = parser.parse_known_args()
    
    hparams = hparam.HParams(**args.__dict__)
    train_and_evaluate(hparams)