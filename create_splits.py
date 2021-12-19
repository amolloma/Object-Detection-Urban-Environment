import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
        # creates a list of files in the source directory
    files = os.listdir(source)
    
    # randomly shuffle files
    random.shuffle(files)
    
    #calculate index to slice files based on percentage
    split_percent = .85
    idx = int(len(files)*split_percent)
    
    #locations for dumping the split files
    train_folder = "/home/workspace/data/waymo/train/"
    val_folder = "/home/workspace/data/waymo/val/"

    # looping through source file list and splitting
    for i, filename in enumerate(files):
        if i < idx:
            shutil.move(os.path.join(source, filename), train_folder)
        else:
            shutil.move(os.path.join(source, filename), val_folder)      
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)