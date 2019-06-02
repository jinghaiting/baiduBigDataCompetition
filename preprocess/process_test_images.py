import sys
sys.path.append('/home/LiZhongYu/data/jht/competition_classification')
import os
from skimage import io
import numpy as np
from tqdm import tqdm
from paths import test_image_dir, test_ids_npy, test_images_npy


def get_ids():
    image_ids = [image_name.rsplit('.', maxsplit=1)[0] for image_name in os.listdir(test_image_dir)
                        if image_name.lower().endswith('.jpg')]
    image_ids.sort()
    return image_ids


def load_ids():
    if os.path.exists(test_ids_npy):
        image_ids = np.load(test_ids_npy)
    else:
        image_ids = get_ids()
        np.save(test_ids_npy, image_ids)
    return image_ids

def get_images_by_id():
    images = []
    image_ids = load_ids()

    for image_id in tqdm(image_ids):
        image_name = '%s.jpg' % image_id
        img_path = os.path.join(test_image_dir, image_name)
        image = io.imread(img_path)
        images.append(image)

    images = np.stack(images).astype(np.uint8)  # 最终的shape为（N,100,100,3）,需要这句   #u无符号，8指明0到255

    return images

def load_images_by_id():
    if os.path.exists(test_images_npy):
        images = np.load(test_images_npy)
    else:
        images = get_images_by_id()
        np.save(test_images_npy, images)
    return images

def main():

    images = load_images_by_id()

if __name__ == '__main__':
    main()
