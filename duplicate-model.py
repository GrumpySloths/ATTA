import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
checkpoint_path = './models'

model_checkpoint_map = {
    'InceptionV3': os.path.join(checkpoint_path, 'inception_v3.ckpt'),
    'adv1/InceptionV3': os.path.join(checkpoint_path, 'trans1_inception_v3.ckpt'),
    'adv2/InceptionV3': os.path.join(checkpoint_path, 'trans2_inception_v3.ckpt'),
    'adv3/InceptionV3': os.path.join(checkpoint_path, 'trans3_inception_v3.ckpt'),
    'adv4/InceptionV3': os.path.join(checkpoint_path, 'trans4_inception_v3.ckpt'),

    # resnet
    'resnet_v2': os.path.join(checkpoint_path, 'resnet_v2_101.ckpt'),
    'adv1/resnet_v2': os.path.join(checkpoint_path, 'trans1_resnet_v2_101.ckpt'),
    'adv2/resnet_v2': os.path.join(checkpoint_path, 'trans2_resnet_v2_101.ckpt'),
    'adv3/resnet_v2': os.path.join(checkpoint_path, 'trans3_resnet_v2_101.ckpt'),
    # Inceptionv4
    'InceptionV4': os.path.join(checkpoint_path, 'inception_v4.ckpt'),
    'adv1/InceptionV4': os.path.join(checkpoint_path, 'trans1_inception_v4.ckpt'),
    'adv2/InceptionV4': os.path.join(checkpoint_path, 'trans2_inception_v4.ckpt'),
    'adv3/InceptionV4': os.path.join(checkpoint_path, 'trans3_inception_v4.ckpt'),
    #InceptionResnetV2
    'InceptionResnetV2': os.path.join(checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'adv1/InceptionResnetV2': os.path.join(checkpoint_path, 'trans1_inception_resnet_v2_2016_08_30.ckpt'),
    'adv2/InceptionResnetV2': os.path.join(checkpoint_path, 'trans2_inception_resnet_v2_2016_08_30.ckpt'),
    'adv3/InceptionResnetV2': os.path.join(checkpoint_path, 'trans3_inception_resnet_v2_2016_08_30.ckpt'),

}

def rename(checkpoint_dir, checkpoint_dir_new, add_prefix):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.train.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.train.load_variable(checkpoint_dir, var_name)
            # Set the new name
            new_name = add_prefix + var_name
            print('Renaming %s to %s.' % (var_name, new_name))
            # Rename the variable
            var = tf.Variable(var, name=new_name)
        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_dir_new)

# rename(model_checkpoint_map['InceptionV3'], model_checkpoint_map['adv4/InceptionV3'], 'adv4/')

# rename(model_checkpoint_map['InceptionV3'], model_checkpoint_map['adv1/InceptionV3'], 'adv1/')
# rename(model_checkpoint_map['InceptionV3'], model_checkpoint_map['adv2/InceptionV3'], 'adv2/')
# rename(model_checkpoint_map['InceptionV3'], model_checkpoint_map['adv3/InceptionV3'], 'adv3/')

# rename(model_checkpoint_map['resnet_v2'], model_checkpoint_map['adv1/resnet_v2'], 'adv1/')
# rename(model_checkpoint_map['resnet_v2'], model_checkpoint_map['adv2/resnet_v2'], 'adv2/')
# rename(model_checkpoint_map['resnet_v2'], model_checkpoint_map['adv3/resnet_v2'], 'adv3/')

# rename(model_checkpoint_map['densenet121'], model_checkpoint_map['adv1/densenet121'], 'adv1/')
# rename(model_checkpoint_map['densenet121'], model_checkpoint_map['adv2/densenet121'], 'adv2/')
# rename(model_checkpoint_map['densenet121'], model_checkpoint_map['adv3/densenet121'], 'adv3/')
rename(model_checkpoint_map['InceptionV4'], model_checkpoint_map['adv1/InceptionV4'], 'adv1/')
rename(model_checkpoint_map['InceptionV4'], model_checkpoint_map['adv2/InceptionV4'], 'adv2/')
rename(model_checkpoint_map['InceptionV4'], model_checkpoint_map['adv3/InceptionV4'], 'adv3/')

rename(model_checkpoint_map['InceptionResnetV2'], model_checkpoint_map['adv1/InceptionResnetV2'], 'adv1/')
rename(model_checkpoint_map['InceptionResnetV2'], model_checkpoint_map['adv2/InceptionResnetV2'], 'adv2/')
rename(model_checkpoint_map['InceptionResnetV2'], model_checkpoint_map['adv3/InceptionResnetV2'], 'adv3/')