# https://opensource.com/article/17/2/python-tricks-artists
from os import listdir
from PIL import Image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir')
    args = parser.parse_args()
    for filename in listdir(f'{args.dir}/'):
        if filename.endswith('.png'):
            try:
                img = Image.open(f'{args.dir}/'+filename) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print(filename) # print out the names of corrupt files