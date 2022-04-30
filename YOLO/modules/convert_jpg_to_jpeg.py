from PIL import Image
import argparse
import os

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('--origin', type=dir_path)
args = parser.parse_args()

directory = args.origin

print(directory)
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        im = Image.open(directory+'/'+filename)
        name=directory+'/'+filename[:-4]+'.jpeg'
        im.save(name)
        print(os.path.join(directory, filename))
        os.remove(directory + '/' + filename)
    elif filename.endswith(".png"):
        im = Image.open(directory+'/'+filename)
        rgb_im = im.convert('RGB')
        name=directory+'/'+filename[:-4]+'.jpeg'
        rgb_im.save(name)
        print(os.path.join(directory, filename))
        os.remove(directory + '/' + filename)
    else:
        continue