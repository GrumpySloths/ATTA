from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import logging
import tempfile
import os
import numpy as np
from scipy.misc import imread, imsave

import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import random
import nni

slim = tf.contrib.slim
args = None
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)
logger = logging.getLogger('ATTA_AutoML')




#获取参数
def get_params():
    parser=argparse.ArgumentParser()
    # parser.add_argument("--cuda_device",type=str,default="0")
    parser.add_argument("--exp_id",type=str,default="default_Su")
    parser.add_argument("--batch_size",type=int,default=10)
    parser.add_argument("--max_epsilon",type=float,default=16.0)
    parser.add_argument("--num_iter",type=int,default=5)
    parser.add_argument("--num_epoch",type=int,default=10)
    parser.add_argument("--conv1",type=int,default=3)
    parser.add_argument("--conv2",type=int,default=3)
    parser.add_argument("--conv3",type=int,default=3)
    parser.add_argument("--momentum",type=float,default=1.0)
    parser.add_argument("--alpha1",type=float,default=0.5)
    parser.add_argument("--alpha2",type=float,default=0.5)
    parser.add_argument("--beta",type=float,default=0.5)
    parser.add_argument("--gamma",type=float,default=0.5)
    parser.add_argument("--image_width",type=int,default=299)
    parser.add_argument("--image_height",type=int,default=299)
    parser.add_argument("--source_model",type=str,default="InceptionV3")
    parser.add_argument("--checkpoint_path",type=str,default='./models')
    parser.add_argument("--input_dir",type=str,default='./dev_data/val_rs')
    parser.add_argument("--label_file",type=str,default='./dev_data/val_rs.csv')
    parser.add_argument("--output_dir",type=str,default='./outputs')
    parser.add_argument("--learning_rate",type=float,default=0.001)

    args,_=parser.parse_known_args()
    return args

# tf.args["DEFINE_string"]('source_model', 'InceptionV3', 'source model to generate adversarial samples')

# tf.args["DEFINE_string"]('checkpoint_path', './models',
#                        'Path to checkpoint for pretained models.')
# tf.args["DEFINE_string"]('input_dir', './dev_data/val_rs',
#                                     'Input directory with images.')
# tf.args["DEFINE_string"]('label_file', './dev_data/val_rs.csv',
#                                     'Input directory with images.')

# tf.args["DEFINE_string"]('output_dir', './outputs',
#                        'Output directory with images.')

# tf.args["DEFINE_string"]("cuda_device",'0',"the cuda to select")

# tf.args["DEFINE_string"]('exp_id', 'default_Su', 'expid')

# tf.args["DEFINE_integer"]('batch_size', 10, 'How many images process at one time.')

# tf.args["DEFINE_float"]('max_epsilon', 16.0, 'max epsilon.')

# tf.args["DEFINE_integer"]('num_iter', 5, 'max iteration.')
# tf.args["DEFINE_integer"]('num_epoch', 40, 'max epoch to train attack network.')

# tf.args["DEFINE_integer"]('conv1', 16, 'conv1 layer of attack network')
# tf.args["DEFINE_integer"]('conv2', 3, 'conv2 layer of attack network')
# tf.args["DEFINE_integer"]('conv3', 0, 'conv3 layer of attack network')

# tf.args["DEFINE_float"]('momentum', 1.0, 'momentum about the model.')

# tf.args["DEFINE_float"]('alpha1', 0.6, 'alpha in the training loss')
# tf.args["DEFINE_float"]('alpha2', 0.4, 'alpha in the training loss')
# tf.args["DEFINE_float"]('gamma', 0.8, 'gamma in the fool loss')

# tf.args["DEFINE_integer"](
#     'image_width', 299, 'Width of each input images.')

# tf.args["DEFINE_integer"](
#     'image_height', 299, 'Height of each input images.')



# os.environ['CUDA_VISIBLE_DEVICES'] = args["cuda_device"]

def load_images(input_dir, label_file, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    labels = []
    idx = 0
    raw_labels = load_labels(label_file)
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        labels.append(raw_labels[filepath.split('/')[-1]])
        idx += 1
        if idx == batch_size:
            yield filenames, images, labels
            filenames = []
            labels = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images, labels

def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


def graph(x_adv, y, x_max, x_min, grad):
    eps = 2.0 * args["max_epsilon"] / 255.0
    num_iter = args["num_iter"]
    alpha = eps / num_iter
    momentum = args["momentum"]
    num_classes = 1001
    source_model = args["source_model"]

    x_adv = x_adv + momentum * alpha * grad

    with tf.variable_scope('attack_network', reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.conv2d(x_adv, 3, kernel_size=(args["conv1"], args["conv1"]))
        x = tf.nn.leaky_relu(x)
        x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(args["conv2"], args["conv2"]))
        if args["conv3"] > 0:
            x = tf.nn.leaky_relu(x_trans)
            x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(args["conv3"], args["conv3"]))
    if source_model == 'InceptionV3':
        with tf.variable_scope('adv1', reuse=tf.AUTO_REUSE):
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, end_points = inception_v3.inception_v3(
                    x_adv, num_classes=num_classes, is_training=False)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv, end_points_adv = inception_v3.inception_v3(
                x_trans, num_classes=num_classes, is_training=False)
    elif source_model == 'resnet_v2':
        with tf.variable_scope('adv1', reuse=tf.AUTO_REUSE):
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, end_points = resnet_v2.resnet_v2_101(
                    x_adv, num_classes=num_classes, is_training=False)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_adv, end_points_adv = resnet_v2.resnet_v2_101(
                x_trans, num_classes=num_classes, is_training=False)
    elif source_model == 'InceptionV4':
        with tf.variable_scope('adv1', reuse=tf.AUTO_REUSE):
            with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
                logits, end_points = inception_v4.inception_v4(
                    x_adv, num_classes=num_classes, is_training=False)
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_adv, end_points_adv =inception_v4.inception_v4(
                x_trans, num_classes=num_classes, is_training=False)
    elif source_model == 'InceptionResnetV2':
        with tf.variable_scope('adv1', reuse=tf.AUTO_REUSE):
            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2.inception_resnet_v2(
                    x_adv, num_classes=num_classes, is_training=False)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_adv, end_points_adv = inception_resnet_v2.inception_resnet_v2(
                x_trans, num_classes=num_classes, is_training=False)


    else:
        print('source_model error!!')
    # define fool loss

    one_hot = tf.one_hot(y, num_classes)
    loss_adv = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits_adv))

    one_hot = tf.one_hot(y, num_classes)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits))
    # loss_fool = - args["gamma"] * loss_adv - loss
    loss_fool=loss+args["gamma"]*loss_adv
    # update adv image
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    noise = optimizer.compute_gradients(loss_fool, var_list=[x_adv])[0][0]
 
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x_adv=x_adv-alpha*tf.sign(noise)
    # x_adv = x_adv + alpha * tf.sign(noise)
    x_adv = tf.clip_by_value(x_adv, x_min, x_max)
    


    return x_adv, noise,loss_fool,x_trans


def train_graph(x_input, x_adv, y):
    momentum = args["momentum"]
    num_classes = 1001
    source_model = args["source_model"]
    if source_model == 'densenet121':
        num_classes = 1000

    with tf.variable_scope('attack_network', reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.conv2d(x_adv, 3, kernel_size=(args["conv1"], args["conv1"]),padding="SAME")
        x = tf.nn.leaky_relu(x)
        x_trans_adv = tf.contrib.layers.conv2d(x, 3, kernel_size=(args["conv2"], args["conv2"]),padding="SAME")
        if args["conv3"] > 0:
            x = tf.nn.leaky_relu(x_trans_adv)
            x_trans_adv = tf.contrib.layers.conv2d(x, 3, kernel_size=(args["conv3"], args["conv3"]),padding="SAME")

    with tf.variable_scope('attack_network', reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.conv2d(x_input, 3, kernel_size=(args["conv1"], args["conv1"]),padding="SAME")
        x = tf.nn.leaky_relu(x)
        x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(args["conv2"], args["conv2"]),padding="SAME")
        if args["conv3"] > 0:
            x = tf.nn.leaky_relu(x_trans)
            x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(args["conv3"], args["conv3"]),padding="SAME")
    if source_model == 'InceptionV3':
        with tf.variable_scope('adv2'):
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits_adv, end_points_adv = inception_v3.inception_v3(
                    x_trans_adv, num_classes=num_classes, is_training=False)
        with tf.variable_scope('adv3'):
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits_trans, end_points_trans = inception_v3.inception_v3(
                    x_trans, num_classes=num_classes, is_training=False)
    elif source_model == 'resnet_v2':
        with tf.variable_scope('adv2'):
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits_adv, end_points_adv = resnet_v2.resnet_v2_101(
                    x_trans_adv, num_classes=num_classes, is_training=False)
        with tf.variable_scope('adv3'):
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits_trans, end_points_trans = resnet_v2.resnet_v2_101(
                    x_trans, num_classes=num_classes, is_training=False)

    else:
        print('source_model error!!')
    # define fool loss
    # pred = tf.argmax(end_points_adv['Predictions'], 1)
    one_hot = tf.one_hot(y, num_classes)
    loss2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits_adv))

    # pred = tf.argmax(end_points_trans['Predictions'], 1)
    one_hot = tf.one_hot(y, num_classes)
    loss3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits_trans))

    loss = args["alpha1"] * tf.norm(x_adv - x_trans_adv, ord=2) + loss2 + args["alpha2"] * loss3
    # update model
    optimizer = tf.train.AdamOptimizer(learning_rate=args["learning_rate"])
    train_op = optimizer.minimize(loss)
    return loss, train_op, x_trans


def main(args):
    #构建映射
    model_checkpoint_map = {
    'InceptionV3': os.path.join(args["checkpoint_path"], 'inception_v3.ckpt'),
    'adv1/InceptionV3': os.path.join(args["checkpoint_path"], 'trans1_inception_v3.ckpt'),
    'adv2/InceptionV3': os.path.join(args["checkpoint_path"], 'trans2_inception_v3.ckpt'),
    'adv3/InceptionV3': os.path.join(args["checkpoint_path"], 'trans3_inception_v3.ckpt'),
    'adv4/InceptionV3': os.path.join(args["checkpoint_path"], 'trans4_inception_v3.ckpt'),
    'AdvInceptionV3': os.path.join(args["checkpoint_path"], 'adv_inception_v3_rename.ckpt'),
    'Ens3AdvInceptionV3': os.path.join(args["checkpoint_path"], 'ens3_adv_inception_v3_rename.ckpt'),
    'Ens4AdvInceptionV3': os.path.join(args["checkpoint_path"], 'ens4_adv_inception_v3_rename.ckpt'),
    'InceptionV4': os.path.join(args["checkpoint_path"], 'inception_v4.ckpt'),
    'InceptionResnetV2': os.path.join(args["checkpoint_path"], 'inception_resnet_v2_2016_08_30.ckpt'),
    'EnsAdvInceptionResnetV2': os.path.join(args["checkpoint_path"], 'ens_adv_inception_resnet_v2_rename.ckpt'),
    # resnet
    'resnet_v2': os.path.join(args["checkpoint_path"], 'resnet_v2_101.ckpt'),
    'adv1/resnet_v2': os.path.join(args["checkpoint_path"], 'trans1_resnet_v2_101.ckpt'),
    'adv2/resnet_v2': os.path.join(args["checkpoint_path"], 'trans2_resnet_v2_101.ckpt'),
    'adv3/resnet_v2': os.path.join(args["checkpoint_path"], 'trans3_resnet_v2_101.ckpt'),
    # densenet
    'densenet121': os.path.join(args["checkpoint_path"], 'tf-densenet121.ckpt'),
    'adv1/densenet121': os.path.join(args["checkpoint_path"], 'trans1_tf-densenet121.ckpt'),
    'adv2/densenet121': os.path.join(args["checkpoint_path"], 'trans2_tf-densenet121.ckpt'),
    'adv3/densenet121': os.path.join(args["checkpoint_path"], 'trans3_tf-densenet121.ckpt'),
}
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].

    f2l = load_labels(args["label_file"])
    eps = 2 * args["max_epsilon"] / 255.0
    batch_shape = [args["batch_size"], args["image_height"], args["image_width"], 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(args["output_dir"])

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_adv = tf.placeholder(tf.float32, shape=batch_shape)
        y = tf.placeholder(tf.int64, shape=[args["batch_size"]])
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        grad=tf.placeholder(tf.float32,shape=batch_shape)
        num_classes=1001

        print("正在构建内循环图")
        x_adv2,noise,loss_fool,x_trans_debug=graph(x_input,y,x_max,x_min,grad)

        print("正在构建外循环图")
        # loss, train_op, x_trans = train_graph(x_input, x_adv, y)
        loss,train_op,x_trans = train_graph(x_input, x_adv, y)
        #加载评估模型
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False,reuse=tf.AUTO_REUSE)
        pred_v3 = tf.argmax(end_points_v3['Predictions'], 1)
        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope=args["source_model"]))
        s2 = tf.train.Saver(slim.get_model_variables(scope='adv1/' + args["source_model"]))
        s3 = tf.train.Saver(slim.get_model_variables(scope='adv2/' + args["source_model"]))
        s4 = tf.train.Saver(slim.get_model_variables(scope='adv3/' + args["source_model"]))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        logger.debug("模型加载完毕")
        with tf.Session(config=config) as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            s1.restore(sess, model_checkpoint_map[args["source_model"]])
            s2.restore(sess, model_checkpoint_map['adv1/' + args["source_model"]])
            s3.restore(sess, model_checkpoint_map['adv2/' + args["source_model"]])
            s4.restore(sess, model_checkpoint_map['adv3/' + args["source_model"]])
            s5.restore(sess, model_checkpoint_map['InceptionV3'])

            # idx = 0
            # l2_diff = 0
            for step in range(args["num_epoch"]):
                idx=0
                loss_avg = 0
                for filenames, images, labels in load_images(args["input_dir"], args["label_file"], batch_shape):
                    idx = idx + 1
                    adv_grad=np.zeros(shape=batch_shape)
                    adv_images=images
                    for i  in range(args["num_iter"]):
                        adv_images,adv_grad,adv_loss,adv_images_debug=sess.run([x_adv2,noise,loss_fool,x_trans_debug],feed_dict={x_input:adv_images,grad:adv_grad,y:labels})
                    trans_loss,_,trans_images= sess.run([loss,train_op,x_trans],
                                                feed_dict={x_input: images, y: labels, x_adv: adv_images})
                    # if step == args["num_epoch"] - 1:
                    #     # tf.reset_default_graph()
                    #     adv_images_sample=sess.run(x_adv_sample,feed_dict={x_input:adv_images,y:labels})
                    #     save_images(adv_images_sample, filenames, adv_dir)
                        # save_images(trans_images, filenames, trans_dir)
                logger.debug("step={} loss={}".format(step, loss_avg / 1000))
                # print("step={} loss={}".format(step, loss_avg / 1000))
            idx=0
            success_count=0
            for filenames, images, labels in load_images(args["input_dir"], args["label_file"], batch_shape):
                
                idx = idx + 1
                logger.debug("start the i={} eval".format(idx))
                adv_grad=np.zeros(shape=batch_shape)
                adv_images=images
                for i  in range(args["num_iter"]):
                    adv_images,adv_grad,adv_loss,adv_images_debug=sess.run([x_adv2,noise,loss_fool,x_trans_debug],feed_dict={x_input:adv_images,grad:adv_grad,y:labels})
                # print("start the i={} eval".format(idx))
                v3=sess.run(pred_v3,feed_dict={x_input:adv_images})
                for filename,l_v3 in zip(filenames,v3):
                    label=f2l[filename]
                    if l_v3 != label:
                        success_count+=1
            logger.debug("Attack success Rate for Inception_V3:",success_count/1000)
            nni.report_final_result(success_count/1000)
            # print("Attack success Rate for Inception_V4:",success_count/1000)
                # save_images(adv_images_sample, filenames, adv_dir)
            # diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
            # l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))


if __name__ == '__main__':
    # tf.app.run()
    # args=vars(get_params())
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    args=vars(get_params())
    args.update(tuner_params)
    main(args)