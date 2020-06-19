import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageOps, ImageDraw
from resizeimage import resizeimage
from torchvision import transforms
import torch.nn as nn

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


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)




def undoctor(image, mask):
    """
    :param image: doctored image
    :param mask: mask indicating distortion
    :return: undoctored image
    """
    #Convert U-Net mask shape to box; used to generate elipse and resize GAN output
    mask = cv2.imread(mask)
    image = cv2.imread(image)
    new_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h = len(new_mask)
    w = len(new_mask[0])
    y_coords = []
    x_coords = []
    for i in range(h):
        for j in range(w):
            if new_mask[i][j] == 255:
                y_coords.append(i)
                x_coords.append(j)

    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords) #Converts mask shape into box
    w1 = x2 - x1
    h1 = y2 - y1

    #create elliptical mask
    ellipt_mask = Image.new('RGB', (w, h), 'black')
    ball_large = Image.new('RGB', (w, h), 'black')
    im = Image.new('RGB', (w1, h1), 'black')
    draw = ImageDraw.Draw(im)
    draw.ellipse((0, 0, w1, h1), fill='white') #Creates elliptical mask used to insert ball
    ellipt_mask.paste(im, (x1, y1))
    ellipt_mask = np.asarray(ellipt_mask)
    ellipt_mask1 = Image.fromarray(ellipt_mask)
    plt.imshow(ellipt_mask1)
    plt.show()

    # Create Football
    gen = Generator(ngpu=0)
    gen.load_state_dict(torch.load('Generator_Weights/gen200'))
    noise = torch.randn(1, 100, 1, 1)
    gan_football = gen(noise)
    gan_football = torch.squeeze(gan_football)

    #reshape to fit area of distortion
    ball = transforms.ToPILImage()(gan_football)
    ball = resizeimage.resize_cover(ball, (w1, h1))
    ball_large.paste(ball, (x1, y1))
    ball_large = np.asarray(ball_large)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #use elliptical mask to overlay ball onto image
    image[np.where(ellipt_mask == 255)] = ball_large[np.where(ellipt_mask == 255)]
    image = Image.fromarray(image)

    plt.imshow(image)
    plt.show()


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


#Diceloss and Adam Optimizer
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


best_model = torch.load('./best_model.pth')

def find_ball(image):
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
        
    pr_mask = pr_mask[0].cpu().numpy().reshape(512,1024)
    
    pr_mask = cv2.resize(pr_mask,(1000,485))
    
    cv2.imwrite("./pr_mask.png", pr_mask*255)

mask = find_ball('image7.png')


undoctor('image7.jpg','mask.jpg')

