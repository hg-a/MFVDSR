import os, glob, re, argparse
import numpy as np
import math
import scipy.io

def psnr(target, ref, scale):
	target_data = np.array(target)
	target_data = target_data[scale:-scale, scale:-scale]

	ref_data = np.array(ref)
	ref_data = ref_data[scale:-scale, scale:-scale]
	
	diff = ref_data - target_data
	diff = diff.flatten('C')
	rmse = math.sqrt(np.mean(diff ** 2.))
	return 20*math.log10(1.0/rmse)

def get_train_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	print (len(l))
	l = [f for f in l if re.search(r"^\d+.mat$", os.path.basename(f))]
	print (len(l))
	train_list = []
	for f in l:
		if os.path.exists(f):
			if os.path.exists(f[:-4]+"_2.mat"): 
				train_list.append([f, f[:-4]+"_2.mat"])
			if os.path.exists(f[:-4]+"_3.mat"): 
				train_list.append([f, f[:-4]+"_3.mat"])
			if os.path.exists(f[:-4]+"_4.mat"): 
				train_list.append([f, f[:-4]+"_4.mat"])
	return train_list

def get_image_batch(train_list,offset,batch_size,IMG_SIZE):
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	cbcr_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([batch_size, IMG_SIZE[1], IMG_SIZE[0], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([batch_size, IMG_SIZE[1], IMG_SIZE[0], 1])
	return input_list, gt_list, np.array(cbcr_list)

def get_img_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	l = [f for f in l if re.search(r"^\d+.mat$", os.path.basename(f))]
	train_list = []
	for f in l:
		if os.path.exists(f):
			if os.path.exists(f[:-4]+"_2.mat"): 
				train_list.append([f, f[:-4]+"_2.mat", 2])
			if os.path.exists(f[:-4]+"_3.mat"): 
				train_list.append([f, f[:-4]+"_3.mat", 3])
			if os.path.exists(f[:-4]+"_4.mat"): 
				train_list.append([f, f[:-4]+"_4.mat", 4])
	return train_list

def get_test_image(test_list, offset, batch_size):
	target_list = test_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	scale_list = []
	for pair in target_list:
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		if "img_2" in mat_dict : 	
			input_img = mat_dict["img_2"]
		if "img_3" in mat_dict : 	
			input_img = mat_dict["img_3"]
		if "img_4" in mat_dict : 	
			input_img = mat_dict["img_4"]
		#else: continue
		gt_img = scipy.io.loadmat(pair[0])['img_raw']
		input_list.append(input_img)
		gt_list.append(gt_img)
		scale_list.append(pair[2])
	return input_list, gt_list, scale_list