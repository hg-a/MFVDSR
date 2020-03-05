from MODEL import MFVDSR
import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.io
from TEST import test_VDSR
import matplotlib.pyplot as plt
from utils import get_image_batch, get_train_list

IMG_SIZE = (41, 41)
BATCH_SIZE = 64
BASE_LR = 0.0001
LR_RATE = 0.1
LR_STEP_SIZE = 40
MAX_EPOCH = 60

DATA_PATH = "./data/train/"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

TEST_DATA_PATH = "./data/test/"

if __name__ == '__main__':
	train_list = get_train_list(DATA_PATH)
	
	train_input  	= tf.placeholder(tf.float32, 
	shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
	train_gt  		= tf.placeholder(tf.float32, 
	shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

	shared_model = tf.make_template('shared_model', MFVDSR)
	train_output, weights 	= shared_model(train_input)
	loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4
	loss = loss/BATCH_SIZE
	tf.summary.scalar("loss", loss)

	global_step 	= tf.Variable(0, trainable=False)
	learning_rate 	= tf.train.exponential_decay(BASE_LR, 
	global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, 
	staircase=True)

	tf.summary.scalar("learning rate", learning_rate)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, 
	beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)

	shuffle(train_list)
	config = tf.ConfigProto()

	with tf.Session(config=config) as sess:
		if not os.path.exists('logs'):
			os.mkdir('logs')
		merged = tf.summary.merge_all()
		file_writer = tf.summary.FileWriter('logs', sess.graph)

		tf.initialize_all_variables().run()

		'''
		if model_ckpt:
			print ("restore model...")
			saver.restore(sess, model_ckpt)
			print ("Done")
		'''
		start_time = time.time()
		
		for epoch in range(0, MAX_EPOCH):
			for step in range(len(train_list)//BATCH_SIZE):
				offset = step*BATCH_SIZE
				input_data, gt_data, cbcr_data = get_image_batch(train_list, 
				offset, BATCH_SIZE, IMG_SIZE)
				feed_dict = {train_input: input_data, train_gt: gt_data}
				_,l,output,lr, g_step, scalsummary = sess.run([opt, loss, train_output, learning_rate, global_step, merged], feed_dict=feed_dict)
				print("[epoch %2.4f] loss %.4f\t lr %.5f"%
				(epoch+(float(step)*BATCH_SIZE/len(train_list)), 
				np.sum(l), lr))
				file_writer.add_summary(scalsummary, g_step)

				del input_data, gt_data, cbcr_data
			if (epoch % 1) == 0:
				saver.save(sess, 
				"./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" 
				% epoch ,global_step=global_step)
	end_time = time.time()
	training_time = end_time - start_time
	print("training time :", training_time)