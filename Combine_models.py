from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

from resizeimage import resizeimage

from torchvision import transforms
import torch
import torch.nn as nn


#%%
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


undoctor('imagetest2.jpg','masktest2.jpg')

