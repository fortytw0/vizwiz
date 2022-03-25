from skimage import io, transform
import glob
import os
import json
import numpy as np

img_dir = 'data/images/'
image_shape = (320, 240, 3)

print(glob.glob(os.path.join(img_dir, '*.jpg')))

def preprocess(image) : 
    return transform.resize(image, image_shape)

def read_image(image_name, split, ifpreprocess=True) : 

    '''
    Pre-processing automatically normalizes image. 
    '''

    image = io.imread(os.path.join(img_dir, split, image_name))

    if ifpreprocess : 
        image = preprocess(image)
    
    return image

if __name__ == '__main__' : 

    image_name = 'VizWiz_train_00000333.jpg'
    image = read_image(image_name, 'train', ifpreprocess=False)
    print('Reading image without pre-processing : ')
    print('Image Shape' , image.shape)
    print('Min Pixel Value : ' , np.min(image))
    print('Max Pixel Value : ' , np.max(image))

    print('###')

    image_name = 'VizWiz_train_00000333.jpg'
    image = read_image(image_name, 'train')
    print('Reading image with pre-processing : ')
    print('Image Shape' , image.shape)
    print('Min Pixel Value : ' , np.min(image))
    print('Max Pixel Value : ' , np.max(image))

