# Spot-the-ball
A program that plays the spot the football game used by BOTB and British Newspapers. 


## Unet - Final.py

This code trains a model to pick up areas of distortion where the ball has been removed from the original images.
Pictures submitted must have their height and width be divisible by 32 i.e (h%32 ==0, w%32 == 0).

Required for training is a folder of masks and training images. A mask will be a black background with a white area to show where the ball used to be , the training image is that of your typical spot the ball challenge image. 

In example images: 
gt_mask.png : Ground truth mask
image7.png : Training Image
pr_mask : Mask shozwing predicted location of ball

Lines 17-24 specify the file path to your pictures and the associated masks.

Lines 207-210 specify the hyperparameters used:

ENCODER = 'resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' 
DEVICE = 'cuda'

Encoder and Encoder weights can be found in Semantic Segmentation models by https://github.com/qubvel/segmentation_models.pytorch

Uncomment Lines 392 - EOF to save a predicted mask.

## Gan_training.py
Run gan training with the same masks and images as for Unet - Final.py


## Combine_models.py

Use this if you have created a mask using Unet - Final.py. 

Inputs are in line 105: undoctor('imagetest2.jpg','masktest2.jpg')
where imagetest2 and masktest2 are  the image and predicted mask respectively.

## Combined Complete 

Line 242 : mask = find_ball('image7.png')

Line 245: undoctor('image7.jpg','mask.jpg')

Change 'image7' to the name of your image to output the location of a ball and generate a ball in that location all in one go. Note Unet - Final.py must have been run to create ./best_model.pth.



