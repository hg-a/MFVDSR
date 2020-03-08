from MODEL import MFVDSR
from utils import psnr, get_img_list, get_test_image
import numpy as np
import tensorflow as tf
import glob, os, time, argparse

# ============================================================
# all dataset is --- 'Set5', 'Set14', 'B100', 'Urban100'
# select epoch
start_epo = 60
end_epo = 60
# ============================================================


DATA_PATH = "./data/test/test/"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

all_dataset_list = ['Set5', 'Set14', 'B100', 'Urban100']

def test_VDSR_with_sess(ckpt_path, data_path, sess, weights, epo, li):
	if li is 'Set5':
		n_ = 5
	elif li is 'Set14':
		n_ = 14
	elif li is 'B100':
		n_ = 100
	elif li is 'Urban100':
		n_ = 100

	folder_list = glob.glob(os.path.join(data_path, li))
	saver.restore(sess, ckpt_path)

	totalpsnr_bicub = totalpsnr_vdsr = 0
	totalx2_bicub = totalx2_vdsr = 0
	totalx3_bicub = totalx3_vdsr = 0
	totalx4_bicub = totalx4_vdsr = 0
	for folder_path in folder_list:
		psnr_list = []
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
			
			psnr_bicub = psnr(input_y, gt_y, scale_list[0])
			psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
			totalpsnr_bicub += psnr_bicub
			totalpsnr_vdsr += psnr_vdsr
			average_bicub = totalpsnr_bicub/(3*n_)
			average_vdsr = totalpsnr_vdsr/(3*n_)

			if((i+1) % 3 == 1):
				totalx2_bicub += psnr_bicub
				totalx2_vdsr += psnr_vdsr
				averagex2_bicub = totalx2_bicub/n_
				averagex2_vdsr = totalx2_vdsr/n_

			if((i+1) % 3 == 2):
				totalx3_bicub += psnr_bicub
				totalx3_vdsr += psnr_vdsr
				averagex3_bicub = totalx3_bicub/n_
				averagex3_vdsr = totalx3_vdsr/n_
			
			if((i+1) % 3 == 0):
				totalx4_bicub += psnr_bicub
				totalx4_vdsr += psnr_vdsr
				averagex4_bicub = totalx4_bicub/n_
				averagex4_vdsr = totalx4_vdsr/n_

		
		psnr_file_savepath = "./parameter check/psnr save/psnr_epochs_%2depochs_%s.txt"%(epo,li)
		psnr_file = open(psnr_file_savepath,'w')
		
		psnr_file.write("PSNR: bicubic -> all")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%average_bicub))
		psnr_file.write("\n")
		psnr_file.write("PSNR: V D S R -> all")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%average_vdsr))
		psnr_file.write("\n")
		psnr_file.write("PSNR: bicubic -> x2")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%averagex2_bicub))
		psnr_file.write("\n")
		psnr_file.write("PSNR: V D S R -> x2")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%averagex2_vdsr))
		psnr_file.write("\n")			
		psnr_file.write("PSNR: bicubic -> x3")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%averagex3_bicub))
		psnr_file.write("\n")
		psnr_file.write("PSNR: V D S R -> x3")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%averagex3_vdsr))
		psnr_file.write("\n")			
		psnr_file.write("PSNR: bicubic -> x4")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%averagex4_bicub))
		psnr_file.write("\n")
		psnr_file.write("PSNR: V D S R -> x4")
		psnr_file.write("\n")
		psnr_file.write(str("%0.2f"%averagex4_vdsr))
		psnr_file.close()

if __name__ == '__main__':
	if not os.path.exists('parameter check'):
		os.mkdir('parameter check')
	if not os.path.exists('parameter check/psnr save'):
		os.mkdir('parameter check/psnr save')
	with tf.Session() as sess:
		input_tensor  			= tf.placeholder(tf.float32, 
									shape=(1, None, None, 1))
		shared_model = tf.make_template('shared_model', MFVDSR)
		output_tensor, weights 	= shared_model(input_tensor)
		saver = tf.train.Saver(weights)
		tf.initialize_all_variables().run()

		for li in all_dataset_list:
			for i in range(start_epo, end_epo+1):
				cp =  i - 1
				cpa = i*3364
				model_ckpt = "checkpoints\\VDSR_const_clip_0.01_epoch_%03d"%cp
				model_ckpt = model_ckpt + ".ckpt-%d"%cpa
				test_VDSR_with_sess(model_ckpt, DATA_PATH, sess, weights, i, li)
