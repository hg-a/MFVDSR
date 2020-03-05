from MODEL import MFVDSR
from scipy import misc
from PIL import Image
from utils import psnr, get_img_list, get_test_image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob, os, re, scipy.io, pickle, time, re, math, argparse, scipy, scipy.misc

# ============================================================
# all dataset is --- 'Set5', 'Set14', 'B100', 'Urban100'
# select epoch
c_epoch = 60
# ============================================================


DATA_PATH = "./data/test/"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

all_dataset_list = ['Set5', 'Set14', 'B100', 'Urban100']


def test_VDSR_with_sess(ckpt_path, data_path, sess, weights, epo, li):

	folder_list = glob.glob(os.path.join(data_path, li))
	saver.restore(sess, ckpt_path)

	for folder_path in folder_list:
		img_list = get_img_list(folder_path)
		for i in range(len(img_list)):
			input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
			input_y = input_list[0]
			gt_y = gt_list[0]

			img_vdsr_y = sess.run([output_tensor], 
			feed_dict={input_tensor: 
			np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
			img_vdsr_y = np.resize(img_vdsr_y, 
								(input_y.shape[0], input_y.shape[1]))
			
			weights_ = sess.run([weights], 
			feed_dict={input_tensor:
			np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
			psnr_bicub = psnr(input_y, gt_y, scale_list[0])
			psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])

			if img_list[i][0][-6] == "\\":
				fn = int(img_list[i][0][-5]) + 1
			else:
				fn = int(img_list[i][0][-6:-4]) + 1


			if((i+1) % 3 == 1):
				scipy.misc.imsave("./gen_test_image/x2/%s/vdsr_no.%s_%0.3f.bmp" % (li, fn, psnr_vdsr), img_vdsr_y)

			if((i+1) % 3 == 2):
				scipy.misc.imsave("./gen_test_image/x3/%s/vdsr_no.%s_%0.3f.bmp" % (li, fn, psnr_vdsr), img_vdsr_y)
			
			if((i+1) % 3 == 0):
				scipy.misc.imsave("./gen_test_image/x4/%s/vdsr_no.%s_%0.3f.bmp" % (li, fn, psnr_vdsr), img_vdsr_y)

			'''
			if((i+1) % 3 == 1):
				scipy.misc.imsave("./gen test image/x2/%s/vdsr_no.%s_.bmp" % (li, fn), gt_y)

			if((i+1) % 3 == 2):
				scipy.misc.imsave("./gen test image/x3/%s/vdsr_no.%s_.bmp" % (li, fn), gt_y)
			
			if((i+1) % 3 == 0):
				scipy.misc.imsave("./gen test image/x4/%s/vdsr_no.%s_.bmp" % (li, fn), gt_y)
			'''
			

def test_VDSR(epoch, ckpt_path, data_path):
	with tf.Session() as sess:
		test_VDSR_with_sess(epoch, ckpt_path, data_path, sess)
if __name__ == '__main__':
	with tf.Session() as sess:
		input_tensor  			= tf.placeholder(tf.float32, 
									shape=(1, None, None, 1))
		shared_model = tf.make_template('shared_model', MFVDSR)
		output_tensor, weights 	= shared_model(input_tensor)
		saver = tf.train.Saver(weights)
		tf.initialize_all_variables().run()

		for li in all_dataset_list:
			cp =  c_epoch - 1
			cpa = c_epoch*3364
			model_ckpt = "checkpoints\\VDSR_const_clip_0.01_epoch_%03d"%cp
			model_ckpt = model_ckpt + ".ckpt-%d"%cpa
			test_VDSR_with_sess(model_ckpt, DATA_PATH, sess, weights, c_epoch, li)