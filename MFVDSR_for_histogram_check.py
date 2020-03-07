import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL_for_parameter_check import MFVDSR
import matplotlib.pyplot as plt
from utils import get_image_batch, get_train_list

IMG_SIZE = (41, 41)
BATCH_SIZE = 64
BASE_LR = 0.0001
LR_RATE = 1
LR_STEP_SIZE = 60
MAX_EPOCH = 60

DATA_PATH = "./data/train/train/"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

TEST_DATA_PATH = "./data/test/"

if __name__ == '__main__':
	train_list = get_train_list(DATA_PATH)

	train_input  	= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
	train_gt  		= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

	shared_model = tf.make_template('shared_model', MFVDSR)
	train_output, weights, beforeReLU_check_tensor, afterReLU_check_tensor, eve_tensor, inpdata 	= shared_model(train_input)
	loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4
	tf.summary.scalar("loss", loss)

	global_step 	= tf.Variable(0, trainable=False)
	learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
	tf.summary.scalar("learning rate", learning_rate)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')#tf.train.MomentumOptimizer(learning_rate, 0.9)
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)

	shuffle(train_list)
	config = tf.ConfigProto()

	start_time = time.time()

	with tf.Session(config=config) as sess:
		if not os.path.exists('logs'):
			os.mkdir('logs')
		merged = tf.summary.merge_all()
		file_writer = tf.summary.FileWriter('logs', sess.graph)
		

		tf.initialize_all_variables().run()

		'''
		if model_path:
			print ("restore model...")
			saver.restore(sess, model_path)
			print ("Done")
		'''

		for epoch in range(0, MAX_EPOCH):
			for step in range(len(train_list)//BATCH_SIZE):
				offset = step*BATCH_SIZE
				input_data, gt_data, cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE, IMG_SIZE)
				feed_dict = {train_input: input_data, train_gt: gt_data}
				_,l,output,lr, g_step, beforecheck, aftercheck, evecheck = sess.run([opt, loss, train_output, learning_rate, global_step, beforeReLU_check_tensor, afterReLU_check_tensor, eve_tensor], feed_dict=feed_dict)
				print("[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr))			
					
				offset = []
				for i in range(-4000,4000):
					offset.append(i*0.001)

				offsetlast = []
				for i in range(-1000,3000):
					offsetlast.append(i*0.001)

				if step <= 5 and epoch == 0:
					for i in range(0,19):
						beforecheck[i] = beforecheck[i].reshape(6885376)
						plt.clf()
						plt.hist(beforecheck[i],offset)
						plt.ylim(ymax = 15000)
						savepath = "./parameter check/histogram/1epo_%2d"%(i+1)
						savepath = savepath + "layer_before_ReLU_%2dstep.png"%step
						plt.savefig(savepath, format='png')
							
						aftercheck[i] = aftercheck[i].reshape(6885376)
						plt.clf()
						plt.hist(aftercheck[i],offset)
						plt.ylim(ymax = 15000)
						savepath = "./parameter check/histogram/1epo_%2d"%(i+1)
						savepath = savepath + "layer_after_ReLU_%2dstep.png"%step
						plt.savefig(savepath, format='png')
							
					output = output.reshape(107584)
					plt.clf()
					plt.hist(output,offsetlast)
					plt.ylim(ymax = 500)
					savepath = "./parameter check/histogram/1epo_20layer_output_%2dstep.png"%step
					plt.savefig(savepath, format='png')

				if step <= 5 and epoch == 59:
					for i in range(0,19):
		
						beforecheck[i] = beforecheck[i].reshape(6885376)
						plt.clf()
						plt.hist(beforecheck[i],offset)
						plt.ylim(ymax = 15000)
						savepath = "./parameter check/histogram/60epo_%2d"%(i+1)
						savepath = savepath + "layer_before_ReLU_%2dstep.png"%step
						plt.savefig(savepath, format='png')
						
							
						aftercheck[i] = aftercheck[i].reshape(6885376)
						plt.clf()
						plt.hist(aftercheck[i],offset)
						plt.ylim(ymax = 15000)
						savepath = "./parameter check/histogram/60epo_%2d"%(i+1)
						savepath = savepath + "layer_after_ReLU_%2dstep.png"%step
						plt.savefig(savepath, format='png')
							
					output = output.reshape(107584)
					plt.clf()
					plt.hist(output,offsetlast)
					plt.ylim(ymax = 500)
					savepath = "./parameter check/histogram/60epo_20layer_output_%2dstep.png"%step
					plt.savefig(savepath, format='png')


				del input_data, gt_data, cbcr_data

			saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)
	end_time = time.time()
	training_time = end_time - start_time
	print(training_time)