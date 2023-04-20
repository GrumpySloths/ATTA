from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from imageio import imread,imsave

import tensorflow as tf
# from scipy.misc import imread, imsave


from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import random

slim = tf.contrib.slim

tf.flags.DEFINE_string("cuda_device",'2',"the cuda to select")

tf.flags.DEFINE_string('exp_id', 'debug', 'expid')

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 8, 'max iteration.')
tf.flags.DEFINE_integer('num_epoch', 13, 'max epoch to train attack network.')

tf.flags.DEFINE_integer('conv1', 16, 'conv1 layer of attack network')
tf.flags.DEFINE_integer('conv2', 3, 'conv2 layer of attack network')
tf.flags.DEFINE_integer('conv3', 0, 'conv3 layer of attack network')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_float('alpha1', 0.8441259307379171, 'alpha in the training loss')
tf.flags.DEFINE_float('alpha2', 0.6989752957059422, 'alpha in the training loss')
tf.flags.DEFINE_float('gamma', 0.03445142957980907, 'gamma in the fool loss')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_string('source_model', 'InceptionV3', 'source model to generate adversarial samples')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')
tf.flags.DEFINE_string('input_dir', './dev_data/val_rs',
                                    'Input directory with images.')
tf.flags.DEFINE_string('label_file', './dev_data/val_rs.csv',
                                    'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './outputs',
                       'Output directory with images.')


FLAGS = tf.flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_para_map={
    "InceptionV3":[inception_v3.inception_v3_arg_scope(),inception_v3.inception_v3],
    "InceptionV4":[inception_v4.inception_v4_arg_scope(),inception_v4.inception_v4],
    "resnet_v2":[resnet_v2.resnet_arg_scope(),resnet_v2.resnet_v2_101],
    "InceptionResnetV2":[inception_resnet_v2.inception_resnet_v2_arg_scope(),inception_resnet_v2.inception_resnet_v2]
}
model_checkpoint_map = {
    'InceptionV3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv1/InceptionV3': os.path.join(FLAGS.checkpoint_path, 'trans1_inception_v3.ckpt'),
    'adv2/InceptionV3': os.path.join(FLAGS.checkpoint_path, 'trans2_inception_v3.ckpt'),
    'adv3/InceptionV3': os.path.join(FLAGS.checkpoint_path, 'trans3_inception_v3.ckpt'),
    'AdvInceptionV3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'Ens3AdvInceptionV3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'Ens4AdvInceptionV3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'InceptionV4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'adv1/InceptionV4': os.path.join(FLAGS.checkpoint_path, 'trans1_inception_v4.ckpt'),
    'adv2/InceptionV4': os.path.join(FLAGS.checkpoint_path, 'trans2_inception_v4.ckpt'),
    'adv3/InceptionV4': os.path.join(FLAGS.checkpoint_path, 'trans3_inception_v4.ckpt'),
    'InceptionResnetV2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'adv1/InceptionResnetV2': os.path.join(FLAGS.checkpoint_path, 'trans1_inception_resnet_v2_2016_08_30.ckpt'),
    'adv2/InceptionResnetV2': os.path.join(FLAGS.checkpoint_path, 'trans2_inception_resnet_v2_2016_08_30.ckpt'),
    'adv3/InceptionResnetV2': os.path.join(FLAGS.checkpoint_path, 'trans3_inception_resnet_v2_2016_08_30.ckpt'),
    'EnsAdvInceptionResnetV2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    # resnet
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt'),
    'adv1/resnet_v2': os.path.join(FLAGS.checkpoint_path, 'trans1_resnet_v2_101.ckpt'),
    'adv2/resnet_v2': os.path.join(FLAGS.checkpoint_path, 'trans2_resnet_v2_101.ckpt'),
    'adv3/resnet_v2': os.path.join(FLAGS.checkpoint_path, 'trans3_resnet_v2_101.ckpt'),

}


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
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
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
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    source_model = FLAGS.source_model

    x_adv = x_adv + momentum * alpha * grad
    scope,model=model_para_map[source_model]
    with tf.variable_scope('attack_network', reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.conv2d(x_adv, 3, kernel_size=(FLAGS.conv1, FLAGS.conv1))
        x = tf.nn.leaky_relu(x)
        x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(FLAGS.conv2, FLAGS.conv2))
        if FLAGS.conv3 > 0:
            x = tf.nn.leaky_relu(x_trans)
            x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(FLAGS.conv3, FLAGS.conv3))

    with tf.variable_scope('adv1', reuse=tf.AUTO_REUSE):
        with slim.arg_scope(scope):
            logits, end_points = model(
                x_adv, num_classes=num_classes, is_training=False)
    with slim.arg_scope(scope):
        logits_adv, end_points_adv = model(
            x_trans, num_classes=num_classes, is_training=False)

    # define fool loss

    one_hot = tf.one_hot(y, num_classes)
    loss_adv = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits_adv))

    one_hot = tf.one_hot(y, num_classes)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits))
    # loss_fool = - FLAGS.gamma * loss_adv - loss
    loss_fool=loss+FLAGS.gamma*loss_adv
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
    num_classes = 1001
    source_model = FLAGS.source_model
    scope,model=model_para_map[source_model]

    with tf.variable_scope('attack_network', reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.conv2d(x_adv, 3, kernel_size=(FLAGS.conv1, FLAGS.conv1))
        x = tf.nn.leaky_relu(x)
        x_trans_adv = tf.contrib.layers.conv2d(x, 3, kernel_size=(FLAGS.conv2, FLAGS.conv2))
        if FLAGS.conv3 > 0:
            x = tf.nn.leaky_relu(x_trans_adv)
            x_trans_adv = tf.contrib.layers.conv2d(x, 3, kernel_size=(FLAGS.conv3, FLAGS.conv3))

    with tf.variable_scope('attack_network', reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.conv2d(x_input, 3, kernel_size=(FLAGS.conv1, FLAGS.conv1))
        x = tf.nn.leaky_relu(x)
        x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(FLAGS.conv2, FLAGS.conv2))
        if FLAGS.conv3 > 0:
            x = tf.nn.leaky_relu(x_trans)
            x_trans = tf.contrib.layers.conv2d(x, 3, kernel_size=(FLAGS.conv3, FLAGS.conv3))

    with tf.variable_scope('adv2'):
        with slim.arg_scope(scope):
            logits_adv, end_points_adv = model(
                x_trans_adv, num_classes=num_classes, is_training=False)
    with tf.variable_scope('adv3'):
        with slim.arg_scope(scope):
            logits_trans, end_points_trans = model(
                x_trans, num_classes=num_classes, is_training=False)



    one_hot = tf.one_hot(y, num_classes)
    loss2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits_adv))

    
    one_hot = tf.one_hot(y, num_classes)
    loss3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits_trans))
    # loss=
    
    loss = FLAGS.alpha1 * tf.norm(x_adv - x_input, ord=2) + loss2 + FLAGS.alpha2 * loss3
    # update model
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss)
    # return loss, train_op, x_trans
    return loss,train_op,x_trans


 
def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels(FLAGS.label_file)
    eps = 2 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)
    record=[]

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_adv = tf.placeholder(tf.float32, shape=batch_shape)
        y = tf.placeholder(tf.int64, shape=[FLAGS.batch_size])
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        grad=tf.placeholder(tf.float32,shape=batch_shape)


        print("正在构建内循环图")
        x_adv2,noise,loss_fool,x_trans_debug=graph(x_input,y,x_max,x_min,grad)

        print("正在构建外循环图")
        # loss, train_op, x_trans = train_graph(x_input, x_adv, y)
        loss,train_op,x_trans = train_graph(x_input, x_adv, y)

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope=FLAGS.source_model))
        s2 = tf.train.Saver(slim.get_model_variables(scope='adv1/' + FLAGS.source_model))
        s3 = tf.train.Saver(slim.get_model_variables(scope='adv2/' + FLAGS.source_model))
        s4 = tf.train.Saver(slim.get_model_variables(scope='adv3/' + FLAGS.source_model))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            s1.restore(sess, model_checkpoint_map[FLAGS.source_model])
            s2.restore(sess, model_checkpoint_map['adv1/' + FLAGS.source_model])
            s3.restore(sess, model_checkpoint_map['adv2/' + FLAGS.source_model])
            s4.restore(sess, model_checkpoint_map['adv3/' + FLAGS.source_model])

            for step in range(FLAGS.num_epoch):
                idx = 0
                loss_avg = 0
                trans_dir = os.path.join(FLAGS.output_dir, "trans-{}".format(FLAGS.exp_id))
                adv_dir = os.path.join(FLAGS.output_dir, "adv-{}".format(FLAGS.exp_id))
                check_or_create_dir(trans_dir)
                check_or_create_dir(adv_dir)
                for filenames, images, labels in load_images(FLAGS.input_dir, FLAGS.label_file, batch_shape):
                    idx = idx + 1
                    print("start the i={} attack".format(idx))
                    adv_grad=np.zeros(shape=batch_shape)
                    adv_images=images
                    for i  in range(FLAGS.num_iter):
                        adv_images,adv_grad,adv_loss,adv_images_debug=sess.run([x_adv2,noise,loss_fool,x_trans_debug],feed_dict={x_input:adv_images,grad:adv_grad,y:labels})
                        print("after %d step,the loss is %.4f"%(i,adv_loss))
                        # adv_images,noise_adv,loss_fool=sess.run([x_adv,noise,loss_fool],feed_dict={x_input:adv_images,y:labels,grad:noise_adv})
                        # loss_fool_avg+=loss_fool
                    
                    # adv_images = sess.run(x_adv, feed_dict={x_input: images, y: labels})
                    trans_loss,_,trans_images= sess.run([loss,train_op,x_trans],
                                                feed_dict={x_input: images, y: labels, x_adv: adv_images})
                    # adv_dir_debug="./outputs/debug/step%d/batch%d"%(step,idx)
                    # trans_dir_debug="./outputs/debug/step%d/batch%d_trans"%(step,idx)

                    # check_or_create_dir(adv_dir_debug)
                    # check_or_create_dir(trans_dir_debug)
                    # save_images(adv_images_debug, filenames, adv_dir_debug)
                    # save_images(trans_images, filenames, trans_dir_debug)

                    # print("保存第%d 批次图片成功"%idx)
                    # print("trans_loss:",trans_loss)
                    record.append(trans_loss)
                    # if step == FLAGS.num_epoch - 1:
                    #     save_images(adv_images, filenames, adv_dir)
                    #     save_images(trans_images, filenames, trans_dir)
                    # trans_loss, _, trans_images = sess.run([loss, train_op, x_trans],
                    #                             feed_dict={x_input: images, y: labels, x_adv: adv_images})
                    loss_avg = loss_avg + trans_loss

                print("step={} loss={}".format(step, loss_avg / 1000))
               # print("trans_loss",record)
            idx=0
            for filenames, images, labels in load_images(FLAGS.input_dir, FLAGS.label_file, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))
                adv_grad=np.zeros(shape=batch_shape)
                adv_images=images
                for i  in range(FLAGS.num_iter):
                    adv_images,adv_grad,adv_loss,adv_images_debug=sess.run([x_adv2,noise,loss_fool,x_trans_debug],feed_dict={x_input:adv_images,grad:adv_grad,y:labels})
                    # print("after %d step,the loss is %.4f"%(i,adv_loss))
                save_images(adv_images, filenames, adv_dir)
            print("save success")
       


        


if __name__ == '__main__':
    tf.app.run()