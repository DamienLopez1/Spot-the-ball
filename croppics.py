import PIL
from PIL import Image
import pandas as pd
import os
#%%
#get coordinates
df = pd.read_csv('coords-valid.csv')
a = df['0']
b = df['1']
c = df['2']
d = df['3']
y_all = pd.concat([a, b, c, d], axis=1)
print(len(y_all))
print(a)

list_of_coords = y_all.values.tolist()
print(list_of_coords)
#%%
#enlarge bounding boxes
def enlargebbox():
    for i in range(len(list_of_coords)):
        r = 0.1
        a = list_of_coords[i][0]
        b = list_of_coords[i][1]
        c = list_of_coords[i][2]
        d = list_of_coords[i][3]
        list_of_coords[i][0] = a + (r * (a-c) * 0.5)
        list_of_coords[i][1] = b + (r * (b-d) * 0.5)
        list_of_coords[i][2] = c - (r * (a-c) * 0.5)
        list_of_coords[i][3] = d - (r * (b-d) * 0.5)
    return list_of_coords

new_coords  = enlargebbox()
print(new_coords)
#%% Convert to tuple for input to crop code

for i in range(len(new_coords)):
    new_coords[i] = tuple(new_coords[i])
new_coords = tuple(new_coords)
print((new_coords))

#%%
#generate list of valid img names
'''
valid_images = pd.read_csv("coords-valid.csv")
list_of_valids = valid_images['Img_No'].tolist()
list_of_names = []
for i in list_of_valids:
    list_of_names.append(str(i) + ".jpg")
print(list_of_names)
print(len(list_of_names))
'''
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)
list_of_img = sorted_alphanumeric(os.listdir(r'C:\Users\TEMP\OneDrive - City, University of London\Undoctored Images'))

print(list_of_img)
print(len(list_of_img))
print(len(new_coords))

#%%


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open('C:/Users/TEMP/OneDrive - City, University of London/Undoctored Images/' + str(image_path))
    cropped_image = image_obj.crop(coords)
    cropped_image.save('croppedimg/crops/' + str(saved_location))


#%%
for i in range(len(list_of_img)):
    crop(list_of_img[i], new_coords[i], list_of_img[i])