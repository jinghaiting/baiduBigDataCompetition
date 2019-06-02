import os
import inspect

def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))


origin_data_dir = os.path.join(root_dir, 'origin_data')
train_image_aug_dir = os.path.join(origin_data_dir, 'train_image_aug')
train_image_dir = os.path.join(origin_data_dir, 'train_image')
test_image_dir = os.path.join(origin_data_dir, 'test_image')

test_visit_dir = os.path.join(origin_data_dir, 'test_visit')

cached_dir = os.path.join(root_dir, 'cache')
mkdir_if_not_exist(dir_list=[cached_dir])

train_images_npy = os.path.join(cached_dir, 'train_images.npy')
train_labels_npy = os.path.join(cached_dir, 'train_labels.npy')
test_images_npy = os.path.join(cached_dir, 'test_images.npy')


train_ids_npy = os.path.join(cached_dir, 'train_images_ids.npy')
train_visit_dir = os.path.join(origin_data_dir, 'train_visit.npy')
train_visits_origin_npy = os.path.join(cached_dir, 'train_visits_origin.npy')
train_visits_274_npy = os.path.join(cached_dir, 'train_visits_274.npy')
train_visits_224_npy = os.path.join(cached_dir, 'train_visits_224.npy')

test_ids_npy = os.path.join(cached_dir, 'test_images_ids.npy')
test_visit_dir = os.path.join(origin_data_dir, 'test_visit.npy')
test_visits_origin_npy = os.path.join(cached_dir, 'test_visits_origin.npy')
test_visits_274_npy = os.path.join(cached_dir, 'test_visits_274.npy')
test_visits_224_npy = os.path.join(cached_dir, 'test_visits_224.npy')



model_path = os.path.join(root_dir, 'model', 'model.h5')

result_data_path = os.path.join(root_dir, 'result_data.txt')

visits_274_new_feature_npy = os.path.join(cached_dir, 'visits_274_new_feature.npy')
visits_224_new_feature_npy = os.path.join(cached_dir, 'visits_224_new_feature.npy')