# Spot-the-ball
A program that plays the spot the football game used by BOTB and British Newspapers. 


#Unet- Final

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

#
