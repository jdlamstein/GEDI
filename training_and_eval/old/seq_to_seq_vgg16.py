import os
import sys
import time
import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from exp_ops.data_loader import inputs
from exp_ops.tf_fun import make_dir, softmax_cost, fine_tune_prepare_layers, \
    ft_non_optimized, class_accuracy
from gedi_config import GEDIconfig
from models import GEDI_vgg16_trainable_batchnorm_shared as vgg16


def lstm_layers(layer_units, dropout=0.5):
    lstms = []
    for u in layer_units:
        lstms += [tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(y),
            input_keep_prob=tf.constant(dropout),
            output_keep_prob=tf.constant(dropout))]
    return tf.nn.rnn_cell.MultiRNNCell(lstms)


# Train or finetune a vgg16 for the GEDI dataset
def train_vgg16(train_dir=None, validation_dir=None):
    config = GEDIconfig()
    if train_dir is None:  # Use globals
        train_data = os.path.join(config.tfrecord_dir, 'train.tfrecords')
        meta_data = np.load(
            os.path.join(
                config.tfrecord_dir, config.tvt_flags[0] + '_' +
                config.max_file))
    else:
        meta_data = np.load(
            os.path.join(
                train_dir, config.tvt_flags[0] + '_' + config.max_file))

    # Prepare image normalization values
    max_value = np.nanmax(meta_data['max_array']).astype(np.float32)
    if max_value == 0:
        max_value = None
        print 'Derived max value is 0'
    else:
        print 'Normalizing with empirical max.'
    if 'min_array' in meta_data.keys():
        min_value = np.min(meta_data['min_array']).astype(np.float32)
        print 'Normalizing with empirical min.'
    else:
        min_value = None
        print 'Not normalizing with a min.'
    ratio = meta_data['ratio']
    print 'Ratio is: %s' % ratio

    if validation_dir is None:  # Use globals
        validation_data = os.path.join(config.tfrecord_dir, 'val.tfrecords')
    elif validation_dir is False:
        pass  # Do not use validation data during training

    # Make output directories if they do not exist
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.which_dataset + '_' + dt_stamp + '/'
    config.train_checkpoint = os.path.join(
        config.train_checkpoint, dt_dataset)  # timestamp this run
    out_dir = os.path.join(config.results, dt_dataset)
    dir_list = [
        config.train_checkpoint, config.train_summaries,
        config.results, out_dir]
    [make_dir(d) for d in dir_list]
    # im_shape = get_image_size(config)
    im_shape = config.gedi_image_size

    print '-'*60
    print('Training model:' + dt_dataset)
    print '-'*60

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(
            train_data,
            config.train_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            train=config.data_augmentations,
            num_epochs=config.epochs,
            normalize=config.normalize)
        val_images, val_labels = inputs(
            validation_data,
            config.validation_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            num_epochs=config.epochs,
            normalize=config.normalize)
        tf.summary.image('train images', train_images)
        tf.summary.image('validation images', val_images)

    # Prepare model on GPU
    num_timepoints = len(config.channel)
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:

            # Prepare the loss function
            if config.balance_cost:
                cost_fun = lambda yhat, y: softmax_cost(yhat, y, ratio=ratio)
            else:
                cost_fun = lambda yhat, y: softmax_cost(yhat, y)

            def cond(i, state, images, output, loss, num_timepoints):  # NOT CORRECT
                return tf.less(i, num_timepoints)

            def body(
                    i,
                    images,
                    label,
                    cell,
                    state,
                    output,
                    loss,
                    vgg,
                    train_mode,
                    output_shape,
                    batchnorm_layers,
                    cost_fun,
                    score_layer='fc7'):
                vgg.build(
                    images[i],
                    output_shape=config.output_shape,
                    train_mode=train_mode,
                    batchnorm=config.batchnorm_layers)
                    it_output, state = cell(vgg[score_layer], state)
                    output= output.write(i, it_output)
                    it_loss = cost_fun(output, label)
                    loss = loss.write(i, it_loss)
                return (i+1, images, label, cell, state, output, loss, vgg, train_mode, output_shape, batchnorm_layers) 

            # Prepare LSTM loop
            train_mode = tf.get_variable(name='training', initializer=True)
            cell = lstm_layers(layer_units=config.lstm_units)
            output = tf.TensorArray(tf.float32, num_timepoints) # output array for LSTM loop -- one slot per timepoint
            loss = tf.TensorArray(tf.float32, num_timepoints)  # loss for LSTM loop -- one slot per timepoint
            i = tf.constant(0)  # iterator for LSTM loop
            loop_vars = [i, images, label, cell, state, output, loss, vgg, train_mode, output_shape, batchnorm_layers, cost_fun]

            # Run the lstm
            processed_list = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=loop_vars,
                back_prop=True,
                swap_memory=False)
            output_cell = processed_list[3]
            output_state = processed_list[4]
            output_activity = processed_list[5]
            cost = processed_list[6]

            # Optimize
            combined_cost = tf.reduce_sum(cost)
            tf.summary.scalar('cost', combined_cost)
            import ipdb;ipdb.set_trace()  # Need to make sure lstm weights are being trained
            other_opt_vars, ft_opt_vars = fine_tune_prepare_layers(
                tf.trainable_variables(), config.fine_tune_layers)
            if config.optimizer == 'adam':
                train_op = ft_non_optimized(
                    cost, other_opt_vars, ft_opt_vars,
                    tf.train.AdamOptimizer, config.hold_lr, config.new_lr)
            elif config.optimizer == 'sgd':
                train_op = ft_non_optimized(
                    cost, other_opt_vars, ft_opt_vars,
                    tf.train.GradientDescentOptimizer,
                    config.hold_lr, config.new_lr)

            train_accuracy = class_accuracy(
                vgg.prob, train_labels)  # training accuracy
            tf.summary.scalar("training accuracy", train_accuracy)

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                # Validation graph is the same as training except no batchnorm
                val_vgg = vgg16.Vgg16(
                    vgg16_npy_path=config.vgg16_weight_path,
                    fine_tune_layers=config.fine_tune_layers)
                val_vgg.build(val_images, output_shape=config.output_shape)
                # Calculate validation accuracy
                val_accuracy = class_accuracy(val_vgg.prob, val_labels)
                tf.summary.scalar("validation accuracy", val_accuracy)

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    summary_dir = os.path.join(
        config.train_summaries, config.which_dataset + '_' + dt_stamp)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(out_dir + 'meta_info', config)
    step, val_max, losses = 0, 0, []

    try:
        # Launch a tensorboard process
        # response = subprocess.Popen(
                    # [sys.executable, '-c', 'tensorboard --logdir=%s'
                    # % (summary_dir)],
                    # shell=True, stdout=subprocess.PIPE,
                    # stderr=subprocess.STDOUT) #Start tensorboard
        # print response
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value, train_acc = sess.run(
                [train_op, cost, train_accuracy])
            losses.append(loss_value)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                if validation_data is not False:
                    _, val_acc = sess.run([train_op, val_accuracy])
                else:
                    val_acc -= 1  # Store every checkpoint

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | logdir = %s')
                print (format_str % (
                    datetime.now(), step, loss_value,
                    config.train_batch / duration, float(duration),
                    train_acc, val_acc, summary_dir))

                # Save the model checkpoint if it's the best yet
                if 1:  # val_acc >= val_max:
                    saver.save(
                        sess, os.path.join(
                            config.train_checkpoint,
                            'model_' + str(step) + '.ckpt'), global_step=step)
                    # Store the new max validation accuracy
                    val_max = val_acc

            else:
                # Training status
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | Training accuracy = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.train_batch / duration,
                                     float(duration), train_acc))
            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
    finally:
        coord.request_stop()
        np.save(os.path.join(config.tfrecord_dir, 'training_loss'), losses)
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir", type=str, dest="train_dir",
        default=None, help="Directory of training data tfrecords bin file.")
    parser.add_argument(
        "--validation_dir", type=str, dest="validation_dir",
        default=None, help="Directory of validation data tfrecords bin file.")
    args = parser.parse_args()
    train_vgg16(**vars(args))
