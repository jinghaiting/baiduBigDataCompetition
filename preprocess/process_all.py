from preprocess.process_test_images import load_images_by_id
from preprocess.process_train__images_labels import load_images_labels_by_id
from preprocess.process_visits import main as main_process_visits

def main():
    images_train, labels = load_images_labels_by_id()
    images_test = load_images_by_id()
    main_process_visits()

if __name__ == '__main__':
    main()