from skimage import io
import os
from paths import *
from paths import root_dir, mkdir_if_not_exist, train_ids_npy, train_image_aug_dir
import numpy as np
from tqdm import tqdm

# 标签对应的one-hot编码
label_map = {
    '001': [1, 0, 0, 0, 0, 0, 0, 0, 0],
    '002': [0, 1, 0, 0, 0, 0, 0, 0, 0],
    '003': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    '004': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    '005': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    '006': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    '007': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    '008': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    '009': [0, 0, 0, 0, 0, 0, 0, 0, 1]
}


# load id to npy

## 分割文件名，得到id
def get_ids():
    # 注释的这些跟下面直接用image_ids = [.........]一个效果

    # image_ids = []        ##注意之前用的是mage_ids = list(),用这个之后打印出来的形状不是（N,）一行,而是(N,1)一列
    # for image_name in os.listdir(datasets_dir):
    #     if image_name.lower().endswith('.jpg'):
    #         image_id = image_name.rsplit('.', maxsplit=1)[0]
    #         image_ids.append(image_id)

    image_ids = [image_name.rsplit('.', maxsplit=1)[0] for image_name in os.listdir(train_image_aug_dir)
                 if image_name.lower().endswith('.jpg')]
    image_ids.sort()
    return image_ids


##从  ids_npy_filename中读取id,若里面没有存，则先把get_ids(),把得到的ids存入
def load_ids():
    if os.path.exists(train_ids_npy):
        image_ids = np.load(train_ids_npy)
    else:
        image_ids = get_ids()
        np.save(train_ids_npy, image_ids)
    return image_ids


# load image based on id_npy

def get_images_labels_by_id():
    images = []
    labels = []

    image_ids = load_ids()
    for image_id in tqdm(image_ids):
        image_name = '%s.jpg' % image_id
        img_path = os.path.join(train_image_aug_dir, image_name)
        image = io.imread(img_path)
        images.append(image)

        label = image_id.rsplit('_', maxsplit=1)[1]
        label = label_map[label]
        labels.append(label)

    images = np.stack(images).astype(np.uint8)  # 最终的shape为（N,100,100,3）,需要这句   #u无符号，8指明0到255
    labels = np.stack(labels, axis=0)
    return images, labels


def load_images_labels_by_id():
    if os.path.exists(train_images_npy) and os.path.exists(train_labels_npy):
        images = np.load(train_images_npy)
        labels = np.load(train_labels_npy)
    else:
        images, labels = get_images_labels_by_id()
        np.save(train_images_npy, images)
        np.save(train_labels_npy, labels)
    return images, labels


def main():

    images, labels = load_images_labels_by_id()



if __name__ == '__main__':
    main()
