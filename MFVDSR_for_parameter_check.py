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
import bisect

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

def makecountlist(obser_tensor, startv, endv, layernumber, step, epo, state):
	rangelist = []
	for i in range(startv, endv):
		rangelist.append(i)
		rangev = [rangev*0.001 for rangev in rangelist if rangev % 5 == 0 and rangev % 2 == 1]
	countlist = [0]*(len(rangev)-1)
	for i in range(0, (len(rangev)-1)):
		vdown = bisect.bisect(obser_tensor, rangev[i])
		vup = bisect.bisect(obser_tensor, rangev[i+1])
		count = vup - vdown
		countlist[i] = count

	savepath = "./parameter check/distribution/%2d"%epo				
	savepath = savepath + "epo_%2d"%(layernumber+1)
	savepath = savepath + "layer_" + state + "_%2dstep.txt"%step
	distribution_file = open(savepath,'w')
	for co in countlist:
		distribution_file.write(str(co))
		distribution_file.write("\n")
	distribution_file.close()


def weight_dist_countlist(obser_tensor, startv, endv, layernumber, step, epo):
	rangelist = []
	for i in range(startv, endv):
		rangelist.append(i)
		rangev = [rangev*0.001 for rangev in rangelist if rangev % 5 == 0 and rangev % 2 == 1]
	countlist = [0]*(len(rangev)-1)
	for i in range(0, (len(rangev)-1)):
		vdown = bisect.bisect(obser_tensor, rangev[i])
		vup = bisect.bisect(obser_tensor, rangev[i+1])
		count = vup - vdown
		countlist[i] = count

	savepath = "./parameter check/distribution/weight_distribution_%2d"%epo	
	savepath = savepath + "epo_%2d"%(layernumber+1)
	savepath = savepath + "layer_%2dstep.txt"%step
	distribution_file = open(savepath,'w')
	for co in countlist:
		distribution_file.write(str(co))
		distribution_file.write("\n")
	distribution_file.close()

if __name__ == '__main__':
	train_list = get_train_list(DATA_PATH)
	start_time = time.time()

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

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)

	shuffle(train_list)
	config = tf.ConfigProto()

	with tf.Session(config=config) as sess:
		#TensorBoard open log with "tensorboard --logdir=logs"
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
			#for step in range(len(train_list)//BATCH_SIZE):
			for step in range(len(train_list)//BATCH_SIZE):
				offset = step*BATCH_SIZE
				input_data, gt_data, cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE, IMG_SIZE)
				feed_dict = {train_input: input_data, train_gt: gt_data}
				_,l,output,lr, g_step, beforecheck, aftercheck, checkweights, evetensor, inp = sess.run([opt, loss, train_output, learning_rate, global_step, beforeReLU_check_tensor, afterReLU_check_tensor, weights, eve_tensor, inpdata], feed_dict=feed_dict)
				print("[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr))

				if step <= 5 and epoch == 0:
					
					for i in range(0,19):
						obser_tensor = beforecheck[i].reshape(6885376)
						obser_tensor.sort()
						makecountlist(obser_tensor, -4525, 4526, i, step, epoch, "before_ReLU")
						obser_tensor = aftercheck[i].reshape(6885376)
						obser_tensor.sort()
						makecountlist(obser_tensor, -4525, 4526, i, step, epoch, "after_ReLU")
							
					obser_tensor = inp.reshape(107584)
					obser_tensor.sort()
					makecountlist(obser_tensor, -4525, 4526, 0, step, epoch, "input")
					obser_tensor = output.reshape(107584)
					obser_tensor.sort()
					makecountlist(obser_tensor, -4525, 4526, 19, step, epoch, "final_output")
					obser_evetensor = evetensor.reshape(107584)
					obser_evetensor.sort()
					makecountlist(obser_evetensor, -4525, 4526, 19, step, epoch, "just_before_apply_skipconnection_before_output")
					

					obser_tensor = checkweights[0].reshape(576)
					obser_tensor.sort()
					weight_dist_countlist(obser_tensor, -1505, 1506, 0, step, epoch)
					for i in range(1,5):
						obser_tensor = checkweights[i*2].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					for i in range(5,10):
						obser_tensor = checkweights[i*2+1].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					for i in range(11,16):
						obser_tensor = checkweights[i*2].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					for i in range(16,20):
						obser_tensor = checkweights[i*2+1].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					obser_tensor = checkweights[41].reshape(576)
					obser_tensor.sort()
					weight_dist_countlist(obser_tensor, -1505, 1506, 19, step, epoch)

				
				if step <= 5 and epoch == 59:
					for i in range(0,19):
						obser_tensor = beforecheck[i].reshape(6885376)
						obser_tensor.sort()
						makecountlist(obser_tensor, -4525, 4526, i, step, epoch, "before_ReLU")
						obser_tensor = aftercheck[i].reshape(6885376)
						obser_tensor.sort()
						makecountlist(obser_tensor, -4525, 4526, i, step, epoch, "after_ReLU")

					obser_tensor = inp.reshape(107584)
					obser_tensor.sort()
					makecountlist(obser_tensor, -4525, 4526, 0, step, epoch, "input")
					obser_tensor = output.reshape(107584)
					obser_tensor.sort()
					makecountlist(obser_tensor, -4525, 4526, 19, step, epoch, "final_output")
					obser_evetensor = evetensor.reshape(107584)
					obser_evetensor.sort()
					makecountlist(obser_evetensor, -4525, 4526, 19, step, epoch, "just_before_apply_skipconnection_before_output")


					obser_tensor = checkweights[0].reshape(576)
					obser_tensor.sort()
					weight_dist_countlist(obser_tensor, -1505, 1506, 0, step, epoch)
					for i in range(1,5):
						obser_tensor = checkweights[i*2].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					for i in range(5,10):
						obser_tensor = checkweights[i*2+1].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					for i in range(11,16):
						obser_tensor = checkweights[i*2].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					for i in range(16,20):
						obser_tensor = checkweights[i*2+1].reshape(36864)
						obser_tensor.sort()
						weight_dist_countlist(obser_tensor, -1505, 1506, i, step, epoch)
					obser_tensor = checkweights[41].reshape(576)
					obser_tensor.sort()
					weight_dist_countlist(obser_tensor, -1505, 1506, 19, step, epoch)
				
						
				del input_data, gt_data, cbcr_data

			#f.close()
		saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)
	end_time = time.time()
	training_time = end_time - start_time
	print(training_time)