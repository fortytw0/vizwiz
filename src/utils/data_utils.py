import json
from urllib import parse
import os
import wget
from skimage import io
from tqdm import tqdm
import time

data_dir = 'data/images'
annotation_dir = 'data/annotation'

image_data_url = 'https://vizwiz.cs.colorado.edu//VizWiz_visualization_img/'

train_labels_url = 'https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations/train.json'
val_labels_url = 'https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations/val.json'
test_labels_url = 'https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations/test.json'


def download_annotations() : 

    wget.download(train_labels_url, annotation_dir)
    wget.download(val_labels_url, annotation_dir)
    wget.download(test_labels_url, annotation_dir)

def fetch_image(image_name) :

    image_url = parse.urljoin(image_data_url, image_name)
    return io.imread(image_url)


def download_images() : 

    splits = ['train', 'val', 'test']

    for split in splits : 

        print('Downloading train data...')
        time.sleep(2)

        json_file = os.path.join(annotation_dir, '{}.json'.format(split))

        with open(json_file) as f :
            annotations = json.load(f)

        num_annotations = len(annotations)

        for i in  tqdm(range(num_annotations)) : 

            image = fetch_image(annotations[i]['image'])
            image_save_path = os.path.join(data_dir, split, annotations[i]['image'])
            io.imsave(image_save_path, image, check_contrast=False)


if __name__ == '__main__' : 

    # image_name = 'VizWiz_train_00000000.jpg'
    # image = fetch_image(image_name)
    # print(type(image))
    # print(image)

    download_images()


