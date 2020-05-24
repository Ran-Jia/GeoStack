# coding=utf-8
import os, sys, re

from .base import DATA_PATH, INPUT_PATH, OUTPUT_PATH, FEATURE_PATH



def yes_no_input():
    while True:
        choice = input("Can new directories for input and output data be created? [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False

# check if path exsits
if not os.path.exists(DATA_PATH):
    # check if making directories
    if yes_no_input():

        print('making directory {}'.format(DATA_PATH))
        os.makedirs(DATA_PATH)

        print('making directory {}'.format(INPUT_PATH))
        os.makedirs(INPUT_PATH)

        print('making directory {}'.format(OUTPUT_PATH))
        os.makedirs(OUTPUT_PATH)

        print('making directory {}'.format(FEATURE_PATH))
        os.makedirs(FEATURE_PATH)

    else:
        print('not making directories...')



