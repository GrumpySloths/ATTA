import numpy as np
from numpy import pi
import math
from scipy.misc import imread, imresize, imsave
import tensorflow as tf
from scipy.fftpack import dct, idct, rfft, irfft
import os
import pandas as pd
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
tf.flags.DEFINE_string('EXP_ID', 'ni-fgsm', 'method for experiment')
tf.flags.DEFINE_string('source_model', 'InceptionV3', 'source model to generate adversarial samples')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', '/home/niujh/SI-NI-FGSM-master/outputs/di_fgsm_resnet_v2',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './logs', 'Output directory with images.')

FLAGS = tf.flags.FLAGS

slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

checkpoint_path = './models'
model_checkpoint_map = {
    'InceptionV3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'AdvInceptionV3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'Ens3AdvInceptionV3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'Ens4AdvInceptionV3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'InceptionV4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'InceptionResnetV2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'EnsAdvInceptionResnetV2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l

T = np.array([
		[0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
		[0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
		[0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
		[0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
		[0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
		[0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
		[0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
		[0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]
	])

""
Jpeg_def_table = np.array([
	[16, 11, 10, 16, 24, 40, 51, 61],
	[12, 12, 14, 19, 26, 58, 60, 55],
	[14, 13, 16, 24, 40, 57, 69, 56],
	[14, 17, 22, 29, 51, 87, 80, 62],
	[18, 22, 37, 56, 68, 109, 103, 77],
	[24, 36, 55, 64, 81, 104, 113, 92],
	[49, 64, 78, 87, 103, 121, 120, 101],
	[72, 92, 95, 98, 112, 100, 103, 99],
])

""
num = 8
q_table = np.ones((num,num))*30
q_table[0:4,0:4] = 25
# print(q_table)

def dct2 (block):
	return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
	return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def rfft2 (block):
	return rfft(rfft(block.T).T)
def irfft2(block):
	return irfft(irfft(block.T).T)


def FD_jpeg_encode(input_matrix):
	output = []
	input_matrix = input_matrix*255
	
	n = input_matrix.shape[0]
	h = input_matrix.shape[1]
	w = input_matrix.shape[2]
	c = input_matrix.shape[3]
	horizontal_blocks_num = w / num
	output2=np.zeros((c,h, w))
	output3=np.zeros((n,3,h, w))	
	vertical_blocks_num = h / num
	n_block = np.split(input_matrix,n,axis=0)
	for i in range(1, n):
		c_block = np.split(n_block[i],c,axis =3)
		j=0
		for ch_block in c_block:
			vertical_blocks = np.split(ch_block, vertical_blocks_num,axis = 1)
			k=0
			for block_ver in vertical_blocks:
				hor_blocks = np.split(block_ver,horizontal_blocks_num,axis = 2)
				m=0
				for block in hor_blocks:
					block = np.reshape(block,(num,num))
					block = dct2(block)
                    # quantization
					table_quantized = np.matrix.round(np.divide(block, q_table))
					table_quantized = np.squeeze(np.asarray(table_quantized))
                    # de-quantization
					table_unquantized = table_quantized*q_table
					IDCT_table = idct2(table_unquantized)
					if m==0:
						output=IDCT_table
					else:
						output=np.concatenate((output,IDCT_table),axis=1)
					# if m==0:
					# 	output=IDCT_table
					# else:			
 					#     output = np.concatenate((output,IDCT_table),axis=1)
					m+=1
				if k==0:
					output1=output
				else:				
					output1 = np.concatenate((output1,output),axis=0)
				k=k+1
			output2[j] = output1
			j=j+1
		output3[i] = output2     
       
	output3 = np.transpose(output3,(0,2,1,3))
	output3 = np.transpose(output3,(0,1,3,2))
	output3 = output3/255
	output3 = np.clip(np.float32(output3),0.0,1.0)
	return output3

def load_images(input_dir,batch_shape):
	images=np.zeros(batch_shape)
	temp=np.zeros((1,224,224,3))
	filenames=[]
	idx=0
	batch_size=batch_shape[0]
	for filepath in tf.gfile.Glob(os.path.join(input_dir,"*")):
		with tf.gfile.Open(filepath,"rb") as f:
			image=imread(f,mode="RGB").astype(np.float)
			image=imresize(image,size=(224,224))/255.0
			temp[0]=image
			# image=image.unsqueeze(0)
			image=FD_jpeg_encode(temp)
			image=imresize(temp[0],size=(299,299))/255.0

			# print("debug point:",image.shape)
		images[idx,:,:,:]=image*2.0-1.0
		# images[idx,:,:,:]=image

		filenames.append(os.path.basename(filepath))
		idx+=1
		if idx==batch_size:
			yield filenames,images
			filenames=[]
			images=np.zeros(batch_shape)
			idx=0
	if idx>0:
		yield filenames,images

# def load_images(input_dir, batch_shape):
#     """Read png images from input directory in batches.
#     Args:
#       input_dir: input directory
#       batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
#     Yields:
#       filenames: list file names without path of each image
#         Lenght of this list could be less than batch_size, in this case only
#         first few images of the result are elements of the minibatch.
#       images: array with all images from this batch
#     """
#     images = np.zeros(batch_shape)
#     filenames = []
#     idx = 0
#     batch_size = batch_shape[0]

#     for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
#         with tf.gfile.Open(filepath, 'rb') as f:
#             image = imread(f, mode='RGB').astype(np.float) / 255.0
#         # Images for inception classifier are normalized to be in [-1, 1] interval.
#         images[idx, :, :, :] = image * 2.0 - 1.0
#         filenames.append(os.path.basename(filepath))
#         idx += 1
#         if idx == batch_size:
#             yield filenames, images
#             filenames = []
#             images = np.zeros(batch_shape)
#             idx = 0
#     if idx > 0:
#         yield filenames, images

def save_results(model_name, success_count):
    file_path = os.path.join(FLAGS.output_dir, FLAGS.EXP_ID + '.csv')
    if os.path.exists(file_path):
        result = pd.read_csv(file_path, index_col=0)
    else:
        result = pd.DataFrame(columns=['InceptionV3', 'AdvInceptionV3', 'Ens3AdvInceptionV3',
                    'Ens4AdvInceptionV3', 'InceptionV4', 'InceptionResnetV2',
                    'EnsAdvInceptionResnetV2', 'resnet_v2'], 
                        index=['InceptionV3', 'InceptionV4', 'InceptionResnetV2', 'resnet_v2'])
    # save
    for i in range(len(model_name)):
        result.loc[FLAGS.source_model, model_name[i]] = success_count[i] / 1000.
    result.to_csv(file_path)

def main():
	f2l = load_labels('./dev_data/val_rs.csv')
	batch_shape=[50,299,299,3]
	num_classes=1001

	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.Graph().as_default():
		x_input=tf.placeholder(tf.float32,shape=batch_shape)
		with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
			logits,end_points=inception_v3.inception_v3(
				x_input,num_classes=num_classes,is_training=False,scope="InceptionV3"
			)
		# with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
		# 	_,end_points=inception_resnet_v2.inception_resnet_v2(
		# 		x_input,num_classes=num_classes,is_training=False,scope="EnsAdvInceptionResnetV2"
		# 	)
		pred=tf.argmax(end_points["Predictions"],1)
		s1=tf.train.Saver(slim.get_model_variables(scope="InceptionV3"))
		# s1=tf.train.Saver(slim.get_model_variables(scope="EnsAdvInceptionResnetV2"))

		
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			s1.restore(sess,model_checkpoint_map['InceptionV3'])
			# s1.restore(sess,model_checkpoint_map['EnsAdvInceptionResnetV2'])

			idx=0
			result=0
			for filenames, images in load_images(FLAGS.input_dir, batch_shape):
				idx+=1
				print("running %d attack"%idx)
				# images=FD_jpeg_encode(images)
				# images=imresize(images,size=(299,299))
				v3=sess.run(pred,feed_dict={x_input:images})
				# print(v3.shape)
				for filename ,label in zip(filenames,v3):
					true_label=f2l[filename]
					if true_label!=label:
						result+=1
				# for filename, label in zip(filenames, labels):
                #         true_label=f2l[filename]
                #         if true_label != label:
                #             result+=1
			print("the fianl result:",result/1000)
    # input_dir = './outputs'
	



if __name__ == "__main__":
	main()


















