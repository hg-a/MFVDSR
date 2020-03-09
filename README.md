# MFVDSR

The code is built on [tensorflow-vdsr](https://github.com/Jongchan/tensorflow-vdsr).   
Environment : window10, GeForce RTX 2080   

## version
Python = 3.5.1   
tensorflow = 1.14.0   
tensorflow-gpu = 1.14.0   
scipy = 1.2.2   
Pillow = 3.3.1   

## generate data
training and test dataset are available [here](https://drive.google.com/open?id=1PMcy0J6NEpYb8AvDohw_0ceBIsjdRa8W)   
checkpoint is available [here](https://drive.google.com/drive/folders/1VprfcEtB7OpabL3g5NT5KjH_YCZNxK2q?usp=sharing)

put 91, 291 folder in './data/train'   
put Set5, Set14, B100, Urban100 folder in './data/test'

### generate training data
run 'aug_train.m' in './data/train'
### generate test data
run 'aug_test.m' in './data/test'


## file list
```shell
#MFVDSR_for_parameter_check.py  : Collect feature maps and weight parameters
#MFVDSR_for_training.py         : MFVDSR training file
#MODEL.py                       : MFVDSR Network
#MODEL_for_parameter_check.py   : MFVDSR Network for parameter check
#TEST_for_make_result_image.py  : Generate test images of test set
#TEST_for_test_all_dataset.py   : PSNR calculation for all test dataset
#TEST_for_test_onlyx2bicubic.py : PSNR calculation for x2 bicubic of a specific test dataset
#utils.py                       : PSNR calculation and data processing
```

## folder list
```shell
#checkpoints      : Folder where checkpoint is stored during training time
#data             : Training / Test Dataset
#parameter check  : Folder where parameters and PSNR calculation results are stored
```