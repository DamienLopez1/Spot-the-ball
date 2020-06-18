
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import albumentations as albu
import torch
import segmentation_models_pytorch as smp

DATA_DIR = 'D:/705/UNet/Pytorch-UNet-master/data/'


#Loading Data
x_train_dir = os.path.join(DATA_DIR, 'trainimgs')
y_train_dir = os.path.join(DATA_DIR, 'trainmasks')

x_valid_dir = os.path.join(DATA_DIR, 'validimgs')
y_valid_dir = os.path.join(DATA_DIR, 'validmasks')

x_test_dir = os.path.join(DATA_DIR, 'testimgs')
y_test_dir = os.path.join(DATA_DIR, 'testmsks')


# Function to view images
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        #plt.imshow((image* 255).astype(np.uint8))
    plt.show()

# ### Dataloader

from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
        
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        
        self.img_names = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir,img_name) for img_name in self.img_names]
        
        self.mask_names = os.listdir(masks_dir)
        self.masks_fps= [os.path.join(masks_dir,mask_name) for mask_name in self.mask_names]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#resnext 50 takes RGB
        mask = cv2.imread(self.masks_fps[i],0)#remove the 0 to use really_best_model
        
        
        mask = np.round(mask/255)
        
        mask = mask[...,np.newaxis]#adds a batch #delete this line to use really_best_model
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)



# Visualize an example of the dataset

dataset = Dataset(x_train_dir, y_train_dir)

image, mask = dataset[0] # get some sample
#mask  = mask * 255
visualize(
    image=image, 
    cars_mask=mask.squeeze(),
)


# ### Augmentations
#Apply data augmentation

#  - horizontal flip
#  - affine transforms
#  - perspective transforms
#  - brightness/contrast/colors manipulations
#  - image bluring and sharpening
#  - gaussian noise
#  - random crops
# 

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        #albu.Blur(blur_limit=7, always_apply= True,p = 0.8),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 1024)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


#### Visualize resulted augmented images and masks

augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
)

# same image with different random transforms
for i in range(3):
    image, mask = augmented_dataset[1]
    visualize(image=image, mask=mask.squeeze())


# ## Create model and train

ENCODER = 'resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' 
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


#Create train and validation dataloaders

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
   
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
   
)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

#Diceloss and Adam Optimizer
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


# create epoch runners 

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)



# train model for 10 epochs
train_tracking = []
valid_tracking = []

trainlosslist = []
validlosslist = []

min_score = 1.2


#**********************
#to run this code, segmentation models, train.py must be modified to return the loss value
for i in range(0, 20):
    
    print('\nEpoch: {}'.format(i))
    
    train_tracking = []
    valid_tracking = []
    train_logs,train_tracking = train_epoch.run(train_loader)
    
    valid_logs,valid_tracking = valid_epoch.run(valid_loader)
    
    #train_logs = train_epoch.run(train_loader)
    
    #valid_logs = valid_epoch.run(valid_loader)

    
    for i in range(len(train_tracking)):
        trainlosslist.append(list(train_tracking[i].values())[0])
    
    for i in range(len(valid_tracking)):
        validlosslist.append(list(valid_tracking[i].values())[0])

    
    plt.plot(trainlosslist)
    plt.plot(validlosslist)
    # If validation score is the best then train it
    if min_score > valid_logs['dice_loss']:
        min_score = valid_logs['dice_loss']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')



# ## Test best saved model

best_model = torch.load('./Resnet model 20 Epochs.pth')
#best_model = torch.load('./best_model.pth')

# create test dataset

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
   
)


test_dataloader = DataLoader(test_dataset)


# evaluate model on test set

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs,testloss = test_epoch.run(test_dataloader)


# ## Visualize predictions

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir,
)


image = test_dataset_vis[0][0] # get some sample

visualize(
    image=image,
)

for i in range(10):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    mask_vis =  test_dataset_vis[n][1].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
   
    
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    
    
    pr_mask = pr_mask[0].cpu().numpy().reshape(512,1024)
    
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )
    
#Code used to Save output images
'''
    image = cv2.resize(image_vis,(1024,512))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./image" + str(i) + ".png", image)

    pr_mask = cv2.resize(pr_mask,(1000,485))
    gt_mask = cv2.resize(gt_mask,(1000,485))

    cv2.imwrite("./pr_mask" + str(i)  + ".png", pr_mask*255)
    cv2.imwrite("./gt_mask" + str(i)  + ".png", gt_mask*255)
'''

