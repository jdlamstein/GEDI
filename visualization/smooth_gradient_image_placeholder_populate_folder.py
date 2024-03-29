"""
To do:
    Get rid of dead vs live, just take images, get prediction, get gradient


"""


import os
import time
import re
import tensorflow as tf
import numpy as np
import imageio
from argparse import ArgumentParser
from glob import glob
from exp_ops.tf_fun import make_dir
from exp_ops.preprocessing_GEDI_images import produce_patch
from gedi_config import GEDIconfig
from models import baseline_vgg16 as vgg16
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from tqdm import tqdm
from skimage import exposure


def normalize(ims, thresh=0.05, max_val=None):
    norm_im = np.zeros_like(ims)
    for idx, im in enumerate(ims):
        min_x = np.min(im)
        if max_val is None:
            max_x = np.max(im)
        else:
            max_x = max_val
        norm_im[idx] = (im - min_x) / (max_x - min_x)
        if thresh:
            norm_im[idx] = np.maximum(norm_im[idx], thresh)
            norm_im[idx][norm_im[idx] == thresh] = 0
    return norm_im


def alpha_mosaic(
        ims,
        maps,
        output,
        title='Mosaic',
        rc=None,
        cc=None,
        top_n=None,
        cmap='gray',
        vmax=None,
        pad=True,
        colorbar=False):
    if rc is None:
        rc = np.ceil(np.sqrt(len(maps))).astype(int)
        cc = np.ceil(np.sqrt(len(maps))).astype(int)
    if top_n is not None:
        l2 = np.asarray([np.sum(x ** 2) for x in maps])
        l2_order = np.argsort(l2)[::-1]
        maps = maps[l2_order]
        maps = maps[:top_n]
    f = plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=20)
    gs1 = gridspec.GridSpec(rc, cc)
    gs1.update(wspace=0.2, hspace=0)  # set the spacing between axes.
    for idx, (tpn_ap_grads, im) in enumerate(zip(maps, ims)):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        if pad:
            trim_h, trim_w = 1, 5
            tpn_ap_grads = np.flipud(tpn_ap_grads)
            tpn_ap_grads = tpn_ap_grads[:-trim_h, trim_w:]
            im = im[:-trim_h, :-trim_w]

        alphas = Normalize(0, 0.4, clip=True)(np.abs(tpn_ap_grads))
        alphas = np.clip(alphas, .05, .9)
        vmax = np.percentile(tpn_ap_grads, 99.9)
        vmin = np.percentile(tpn_ap_grads, 0.1)
        tpn_ap_grads = Normalize(vmin, vmax)(tpn_ap_grads)
        tpn_ap_grads = cmap(tpn_ap_grads)
        tpn_ap_grads[..., -1] = alphas
        ax1.imshow(
            exposure.equalize_adapthist(im, clip_limit=0.008), cmap='Greens')
        xmin, ymin = 0, 0
        xmax, ymax = im.shape[:2]
        cf = ax1.imshow(tpn_ap_grads, extent=(xmin, xmax, ymin, ymax))
        if colorbar and idx == len(maps) - 1:
            plt.colorbar(cf)
    plt.savefig(output)
    plt.show()
    plt.close(f)


def flatten_list(l):
    """Flatten a list of lists."""
    return [val for sublist in l for val in sublist]


def save_images(
        y,
        yhat,
        viz,
        files,
        output_folder,
        target,
        label_dict,
        ext='.png'):
    """Save TP/FP/TN/FN images in separate folders."""
    quality = ['true', 'false']
    folders = [[os.path.join(
        output_folder, '%s_%s' % (
            k, quality[0])), os.path.join(
        output_folder, '%s_%s' % (
            k, quality[1]))] for k in label_dict.keys()]
    flat_folders = flatten_list(folders)
    [make_dir(f) for f in flat_folders]
    for iy, iyhat, iviz, ifiles in list(zip(y, yhat, viz, files)):
        print(iy.shape)
        print(iyhat.shape)
        print(np.shape(iviz))
        print(np.shape(ifiles))
        print('iy', iy)
        correct = iy == iyhat
        target_label = iy == target
        print('correct', correct)

        print('target_label', target_label)
        f = plt.figure()
        iviz = np.squeeze(iviz)
        plt.imshow(iviz)
        print(type(files))
        print(type(ifiles))
        print(ifiles.shape)
        print('ifiles', ifiles)
        ifiles = ifiles.tolist()
        it_f = ifiles.split('/')[-1].split('\.')[0]
        if correct and target_label:
            # TP
            it_folder = folders[0][0]
        elif correct and not target_label:
            # TN
            it_folder = folders[0][1]
        elif not correct and target_label:
            # FP
            it_folder = folders[1][0]
        elif not correct and not target_label:
            # FN
            it_folder = folders[1][1]
        plt.title('Predicted label=%s, true label=%s' % (iyhat, iy))
        plt.savefig(
            os.path.join(
                it_folder,
                '%s%s' % (it_f, ext)))
        plt.close(f)

#iviz.squeeze()


def visualization_function(images, viz):

    """Wrapper for summarizing visualizations across channels."""

    if viz == 'sum_abs':
        return np.sum(np.abs(images), axis=-1)
    elif viz == 'sum_p':
        return np.sum(np.pow(images, 2), axis=-1)
    elif viz == 'none':
        return images
    else:
        raise RuntimeError('Visualization method not implemented.')


def add_noise(image_batch, loc=0, scale=0.15 / 255):
    """Add gaussian noise to the input for smoothing visualizations."""
    return np.copy(image_batch) + np.random.normal(
        size=image_batch.shape, loc=loc, scale=scale)


def crop_center(img, crop_size):
    """Center crop images."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def crop_and_save(savename,img, crop_size):
    im = crop_center(img, crop_size)
    imageio.imwrite(savename, im)


def renormalize(img, max_value, min_value):
    """Normalize images to [0, 1]."""
    return (img - min_value) / (max_value - min_value)


def image_batcher(
        start,
        num_batches,
        images,
        labels,
        config,
        training_max,
        training_min,
        num_channels=3,
        per_timepoint=False,
        output_folder = ''):
    """Placeholder image/label batch loader."""
    for b in range(num_batches):
        next_image_batch = images[start:start + config.validation_batch]
        image_stack, output_files = [], []
        label_stack = labels[start:start + config.validation_batch]
        
        
        for f in next_image_batch:
            if per_timepoint:
                for channel in range(num_channels):
                    # 1. Load image patch
                    patch = produce_patch(
                        f,
                        channel,
                        config.panel,
                        divide_panel=config.divide_panel,
                        max_value=config.max_gedi,
                        min_value=config.min_gedi).astype(np.float32)
                    #added
#                    patch = patch *255.0/ 65535
                    # 2. Repeat to 3 channel (RGB) image
                    patch = np.repeat(patch[:, :, None], 3, axis=-1)
                    # 3. Renormalize based on the training set intensities
                    patch = renormalize(patch,max_value=training_max,min_value=training_min)
                    print('max patch', np.max(patch))

                    # 4. Crop the center
                    patch = crop_center(patch, config.model_image_size[:2])
                    # 5. Clip to [0, 1] just in case
                    patch[patch > 1.] = 1.
                    patch[patch < 0.] = 0.
                    imageio.imwrite(os.path.join(output_folder, 'cropped', ), patch)
                    # 6. Add to list
                    image_stack += [patch[None, :, :, :]]
                    output_files += ['f_%s' % channel]
            else:
                # 1. Load image patch
                patch = produce_patch(
                    f,
                    config.channel,
                    config.panel,
                    divide_panel=config.divide_panel,
                    max_value=config.max_gedi,
                    min_value=config.min_gedi).astype(np.float32)
                # 2. Repeat to 3 channel (RGB) image
                patch = np.repeat(patch[:, :, None], 3, axis=-1)
                # 3. Renormalize based on the training set intensities
                patch = renormalize(
                    patch,
                    max_value=training_max,
                    min_value=training_min)
                # 4. Crop the center
                patch = crop_center(patch, config.model_image_size[:2])
                # 5. Clip to [0, 1] just in case
                patch[patch > 1.] = 1.
                patch[patch < 0.] = 0.
                print('max patch', np.max(patch))
                # 6. Add to list
                image_stack += [patch[None, :, :, :]]
                output_files = np.copy(next_image_batch)
        # Add dimensions and concatenate
        start += config.validation_batch
        yield np.concatenate(image_stack, axis=0), label_stack, output_files


def randomization_test(y, yhat, iterations=10000):
    """Randomization test of difference of predicted accuracy from chance."""
    true_score = np.mean(y == yhat)
    perm_scores = np.zeros((iterations))
    lab_len = len(y)
    for it in range(iterations):
        perm_scores[it] = np.mean(
            yhat == np.copy(y)[np.random.permutation(lab_len)])
    p_value = (np.sum(true_score < perm_scores) + 1) / float(iterations + 1)
    return p_value


# Evaluate your trained model on GEDI images
def visualize_model(
        live_ims,
        dead_ims,
        model_file,
        output_folder,
        num_channels,
        smooth_iterations=50,
        untargeted=False,
        viz='none',
        per_timepoint=True):
    """Train an SVM for your dataset on GEDI-model encodings."""
    config = GEDIconfig()
    if live_ims is None:
        raise RuntimeError(
            'You need to supply a directory path to the live images.')
    if dead_ims is None:
        raise RuntimeError(
            'You need to supply a directory path to the dead images.')
        
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    live_files = glob(os.path.join(live_ims, '*%s' % config.raw_im_ext))
    dead_files = glob(os.path.join(dead_ims, '*%s' % config.raw_im_ext))
    combined_labels = np.concatenate((
        np.ones(len(live_files)),
        np.zeros(len(dead_files))))
    label_dictionary = {0: 'Dead', 1: 'Live'}
#    combined_labels = np.concatenate((
#        np.zeros(len(live_files)),
#        np.ones(len(dead_files))))
    combined_files = np.concatenate((live_files, dead_files))
    if len(combined_files) == 0:
        raise RuntimeError('Could not find any files. Check your image path.')

    config = GEDIconfig()
    model_file_path = os.path.sep.join(model_file.split(os.path.sep)[:-1])
    meta_file_pointer = os.path.join(
        model_file_path,
        'train_maximum_value.npz')
    if not os.path.exists(meta_file_pointer):
        raise RuntimeError(
            'Cannot find the training data meta file: train_maximum_value.npz'
            'Closest I could find from directory %s was %s.'
            'Download this from the link described in the README.md.'
            % (model_file_path, glob(os.path.join(model_file_path, '*.npz'))))
    meta_data = np.load(meta_file_pointer)

    # Prepare image normalization values
    training_max = np.max(meta_data['max_array']).astype(np.float32)
    training_min = np.min(meta_data['min_array']).astype(np.float32)

    # Find model checkpoints
    ds_dt_stamp = re.split('/', model_file)[-2]
    out_dir = os.path.join(config.results, ds_dt_stamp)

    # Make output directories if they do not exist
    dir_list = [config.results, out_dir]
    [make_dir(d) for d in dir_list]

    # Prepare data on CPU
    images = tf.placeholder(
        tf.float32,
        shape=[None] + config.model_image_size,
        name='images')
    labels = tf.placeholder(
        tf.int64,
        shape=[None],
        name='labels')

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            vgg = vgg16.model_struct(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            vgg.build(
                images,
                output_shape=config.output_shape)

        # Setup validation op
        scores = vgg.fc7
        preds = tf.argmax(vgg.prob, 1)
        activity_pattern = vgg.fc8
        print(activity_pattern.get_shape())
        if not untargeted:
            oh_labels = tf.one_hot(labels, config.output_shape)
            print(oh_labels.get_shape())
            activity_pattern *= oh_labels
        grad_image = tf.gradients(activity_pattern, images)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Loop through each checkpoint then test the entire validation set
    ckpts = [model_file]
    ckpt_yhat, ckpt_y, ckpt_scores = [], [], []
    ckpt_file_array, ckpt_viz_images = [], []
    print('-' * 60)
    print('Beginning evaluation')
    print('-' * 60)

    if config.validation_batch > len(combined_files):
        print('Trimming validation_batch to %s (same as # of files).' % len(combined_files) )
        config.validation_batch = len(combined_files)

    count = 0
    for idx, c in tqdm(enumerate(ckpts), desc='Running checkpoints'):
        dec_scores, yhat, y, file_array, viz_images = [], [], [], [], []
        # Initialize the graph
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(
            tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()))

        # Set up exemplar threading
        saver.restore(sess, c)
        start_time = time.time()
        num_batches = np.floor(len(combined_files) / float(config.validation_batch)).astype(int)
        print('num batches', num_batches)
        print('len combined files', len(combined_files))
        for image_batch, label_batch, file_batch in tqdm(
                image_batcher(
                    start=0,
                    num_batches=num_batches,
                    images=combined_files,
                    labels=combined_labels,
                    config=config,
                    training_max=training_max,
                    training_min=training_min,
                    num_channels=num_channels,
                    per_timepoint=per_timepoint,
                    output_folder=output_folder)):
            feed_dict = {
                images: image_batch,
                labels: label_batch
            }
            it_grads = np.zeros((image_batch.shape))
            sc, tyh = sess.run(
                [scores, preds],
                feed_dict=feed_dict)
            for idx in range(smooth_iterations):
                feed_dict = {
                    images: add_noise(image_batch),
                    labels: label_batch
                }
                it_grad = sess.run(
                    grad_image,
                    feed_dict=feed_dict)
                it_grads += it_grad[0]
            print('it grad shape', np.shape(it_grad))

            it_grads /= smooth_iterations  # Mean across iterations
            it_grads = visualization_function(it_grads, viz)
            print('it grads shape', np.shape(it_grads))

            # Save each grad individually
            print('grad i max', np.max(it_grads))
            print('grad i min', np.min(it_grads))
            it_grads = np.interp(it_grads, [np.min(it_grads), np.max(it_grads)], [0, 65535])
            print('grad i max interp', np.max(it_grads))
            print('grad i min interp', np.min(it_grads))
            it_grads = np.uint16(it_grads)
            print('grad i max uint16', np.max(it_grads))
            print('grad i min uint16', np.min(it_grads))

            for grad_i, pred_i, file_i, label_i in zip(it_grads, tyh, file_batch, label_batch):
                grad_folder = os.path.join(output_folder, 'heatmaps', label_dictionary[label_i])
                if not os.path.exists(grad_folder):
                    os.makedirs(grad_folder)
                out_pointer = os.path.join(
                    grad_folder,
                    file_i.split(os.path.sep)[-1])
                out_pointer = out_pointer.split('.')[0] + '.tif'
#                f = plt.figure()

                imageio.imwrite(out_pointer, grad_i)
#                plt.imshow(grad_i)
#                plt.title('Pred=%s, label=%s' % (pred_i, label_batch))
#                plt.savefig(out_pointer)
#                plt.close(f)

            # Plot a moisaic of the grads
            # if viz == 'none':
            #     pos_grads = normalize(np.maximum(it_grads, 0))
            #     neg_grads = normalize(np.minimum(it_grads, 0))
            #     alpha_mosaic(
            #         image_batch,
            #         pos_grads,
            #         'pos_batch_%s.pdf' % count,
            #         title='Positive gradient overlays.',
            #         rc=1,
            #         cc=len(image_batch),
            #         cmap=plt.cm.Reds)
            #     alpha_mosaic(
            #         image_batch,
            #         neg_grads,
            #         'neg_batch_%s.pdf' % count,
            #         title='Negative gradient overlays.',
            #         rc=1,
            #         cc=len(image_batch),
            #         cmap=plt.cm.Reds)
            # else:
            #     alpha_mosaic(
            #         image_batch,
            #         it_grads,
            #         output_folder + '/batch_%s.pdf' % count,
            #         title='Gradient overlays.',
            #         rc=1,
            #         cc=len(image_batch),
            #         cmap=plt.cm.Reds)
            count += 1

            # Store the results
            dec_scores += [sc]  
            yhat = np.append(yhat, tyh)
            y = np.append(y, label_batch)
            file_array = np.append(file_array, file_batch)
            viz_images += [it_grads]
            
            for _im, file_i, label_i in zip(image_batch, file_batch, label_batch):
                '''Saves cropped input images for comparison'''
#                print(np.unique(_im))
                print('shape', np.shape(_im))
                print('max _im', np.max(_im))
                print('min _im', np.min(_im))
                _im = np.uint8(_im * 255)
                print(type(_im))
                print(_im.dtype)
                print('max _im', np.max(_im))
                print('min _im', np.min(_im))
                crop_folder = os.path.join(output_folder, 'cropped', label_dictionary[label_i])
                if not os.path.exists(crop_folder):
                    os.makedirs(crop_folder)
                out_pointer = os.path.join(
                    crop_folder,
                    file_i.split(os.path.sep)[-1])
                out_pointer = out_pointer.split('.')[0] + '.tif'
                imageio.imwrite(out_pointer, _im)

        ckpt_yhat.append(yhat)
        ckpt_y.append(y)
        ckpt_scores.append(dec_scores)
        ckpt_file_array.append(file_array)
        ckpt_viz_images.append(viz_images)
        print ('Batch %d took %.1f seconds' % (idx, time.time() - start_time))
    sess.close()
    ckpt_viz_images = np.squeeze(ckpt_viz_images)
    # Save everything
    np.savez(
        os.path.join(out_dir, 'validation_accuracies'),
        ckpt_yhat=ckpt_yhat,
        ckpt_y=ckpt_y,
        ckpt_scores=ckpt_scores,
        ckpt_names=ckpts,
        combined_files=ckpt_file_array,
        ckpt_viz_images=ckpt_viz_images)

    print('ckpt_y', ckpt_y)
    print('ckpt_yhat',ckpt_yhat)
    print('ckpt_viz_image shape', np.shape(ckpt_viz_images))
    print('ckpt_file_array', ckpt_file_array)
    ckpt_file_array = ckpt_file_array[0]
    ckpt_y = ckpt_y[0]
    ckpt_yhat = ckpt_yhat[0]
     # Save images
#    save_images(
#        y= ckpt_y,
#        yhat= ckpt_yhat,
#        viz=ckpt_viz_images,
#        files=ckpt_file_array,
#        output_folder=output_folder,
#        target=1,
#        label_dict={
#            'live': 1,
#            'dead': 0
#        })


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--live_ims",
        type=str,
        dest="live_ims",
        default='/mnt/data/ScientistLiveDead/BSLive',
        help="Directory containing your Live .tiff images.")
    parser.add_argument(
        "--dead_ims",
        type=str,
        dest="dead_ims",
        default='/mnt/data/ScientistLiveDead/BSDead',
        help="Directory containing your Dead .tiff images.")
    parser.add_argument(
        "--model_file",
        type=str,
        dest="model_file",
        default = '/home/jlamstein/Documents/pretrained_weights/trained_gedi_model/model_58600.ckpt-58600',
        help="Folder containing your trained CNN's checkpoint files.")
#        default='/Users/nickjermey/GEDI/models/trained_gedi_model/model_58600.ckpt-58600',
    parser.add_argument(
        "--untargeted",
        dest="untargeted",
        action='store_false',
        help='Visualize overall saliency instead of features related to the most likely category.')
    parser.add_argument(
        "--per_timepoint",
        dest="per_timepoint",
        action='store_true',
        help='Produce visualizations of every timepoint in a tiff image.')
    parser.add_argument(
        "--smooth_iterations",
        type=int,
        dest="smooth_iterations",
        default=10,
        help='Number of iterations of smoothing for visualizations.')
    parser.add_argument(
        "--num_channels",
        type=int,
        dest="num_channels",
        default=1,
        help='Number of channels to visualize.')
    parser.add_argument(
        "--visualization",
        type=str,
        dest="viz",
        default='sum_abs',
        help='Number of iterations of smoothing for visualizations.')
    parser.add_argument(
        "--output_folder",
        type=str,
        dest="output_folder",
        default='/mnt/data/ScientistLiveDead/gradient_images',
        help='Folder to save the visualizations.')

    args = parser.parse_args()
    visualize_model(**vars(args))
