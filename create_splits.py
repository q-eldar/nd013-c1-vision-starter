import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    test_ratio = 0.20
    data_dir += '/waymo'
    tfrecord_dir = data_dir + '/training_and_validation'
    if(os.path.exists(tfrecord_dir)):
        print(f'using {tfrecord_dir} as data_dir')
    else:
        print(f'using {data_dir} as data_dir')
        tfrecord_dir = data_dir

    tfrecord_files = [fname for fname in glob.glob(f'{tfrecord_dir}/*.tfrecord')]
    
    np.random.shuffle(tfrecord_files)
    train_files, val_files = np.split(tfrecord_files, [int(len(tfrecord_files)*(1-test_ratio))])
    print(f'numver of train files: {len(train_files)}')
    print(f'number of val files: {len(val_files)}')
    train = os.path.join(data_dir,'train')
    for f in train_files:
        shutil.move(f, train)

    val = os.path.join(data_dir,'val')
    for f in val_files:
        shutil.move(f, val)
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)