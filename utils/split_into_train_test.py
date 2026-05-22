import os
import random
import shutil
import timeit
import argparse

def random_split(root_dir, splited_dir, split_ratio=0.8):
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')

    # create train & test folder
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(os.path.join(train_dir, 'images')):
        os.makedirs(os.path.join(train_dir, 'images'))
    if not os.path.exists(os.path.join(train_dir, 'labels')):
        os.makedirs(os.path.join(train_dir, 'labels'))

    # get all the images & labels filenames
    filenames = list(set([filename.split('_p')[0] for filename in os.listdir(os.path.join(splited_dir, 'images'))]))

    # shuffle the finenames
    random.shuffle(filenames)

    # calculate the train/test set files number
    num_train = int(len(filenames) * split_ratio)


    # move the train set
    for i in range(num_train):
        pre_image_src = os.path.join(splited_dir, 'images', filenames[i]+'_pre_disaster.tif')
        post_image_src = os.path.join(splited_dir, 'images', filenames[i]+'_post_disaster.tif')
        building_label_src = os.path.join(splited_dir, 'labels', filenames[i]+'_pre_disaster.json')
        damage_label_src = os.path.join(splited_dir, 'labels', filenames[i]+'_post_disaster.json')


        shutil.move(pre_image_src, os.path.join(train_dir, 'images', filenames[i]+'_pre_disaster.tif'))
        shutil.move(post_image_src, os.path.join(train_dir, 'images', filenames[i]+'_post_disaster.tif'))
        shutil.move(building_label_src, os.path.join(train_dir, 'labels', filenames[i]+'_pre_disaster.json'))
        shutil.move(damage_label_src, os.path.join(train_dir, 'labels', filenames[i]+'_post_disaster.json'))

    # then the rest should be the test set
    os.rename(splited_dir, test_dir)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    parser = argparse.ArgumentParser(
        description=
        """""")

    parser.add_argument('--root_path',
                        required=True,
                        metavar="/path/to/xBD",
                        help='Path to parent dataset directory "xBD"')
    parser.add_argument('--data_path',
                        required=True,
                        metavar="/path/to/xBD/tier1",
                        help='Path to parent dataset directory "xBD"')
    parser.add_argument('--split_ratio',
                        type=float,
                        required=True,
                        metavar="train set ratio",
                        help='Path to parent dataset directory "xBD"')
   
    args = parser.parse_args()

    random_split(args.root_path, args.data_path, args.split_ratio)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))