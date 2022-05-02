import sys
import os
p = os.path.abspath('..')
sys.path.insert(1, p)
# p = os.path.abspath('/home/dodalpaga/.local/lib/python3.8/site-packages')
# sys.path.insert(1, p)
# p = os.path.abspath('/usr/lib/python38.zip')
# sys.path.insert(1, p)

from yolov5.run_model import run_model
from yolov5.run_inference import inference
import modules.filters as filters


import argparse, glob, time
import pandas as pd
from tqdm import tqdm
import cv2
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', None)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
def file_choices(choices,fname):
    ext = os.path.splitext(fname)[1][1:]
    if ext not in choices:
       parser.error("file doesn't end with one of {}".format(choices))
    return fname

parser = argparse.ArgumentParser()
parser.add_argument('--origin', type=dir_path)
parser.add_argument('--weights',type=lambda s:file_choices(("pt"),s))
parser.add_argument('--conf_thres',type=float)
args = parser.parse_args()

origin = args.origin
weights_path = args.weights
conf_thres = args.conf_thres

images_path = glob.glob(os.path.join(origin+'/*.jpeg'))
labels_path = glob.glob(os.path.join(origin+'/*.txt'))

print("Il y a",len(images_path),"images de test")
print("Voici un extrait :")
print(pd.DataFrame(images_path).head(5))

if len(images_path)>0:
    os.system("rm -r ./"+origin+"/Images_predites")
    os.system("rm -r ./"+origin+"/Prediction")
    os.system("mkdir Prediction")
    os.system("mkdir Images_predites")

    model = run_model(weights=weights_path)
    times = []

    os.system('mkdir {}/Filtered/'.format(origin))
    for img_path in tqdm(images_path):
        start = time.time()
        # Applying filter
        img = filters.filter_V1(img_path)
        filtered_img_path = origin+"/Filtered/"+img_path.split("/")[-1]
        cv2.imwrite(filtered_img_path,img)
        # Making inference
        inference(source=img_path,conf_thres=conf_thres,model=model);
        times.append(time.time()-start)
    print("\n Le temps moyen d'inference est :",round(sum(times)/len(times),2),"s")
    os.system("mv Prediction "+origin)

    font = cv2.FONT_HERSHEY_PLAIN
    # prediction_paths = glob.glob(os.path.join(origin+'/Prediction/*.txt'))
    prediction_paths = glob.glob(os.path.join(origin+'/*.jpeg'))
    for prediction_path in tqdm(prediction_paths):
        # Loading image and label
        # name = prediction_path.split("/")[-1][:-4]
        name = prediction_path.split("/")[-1][:-5]
        img_path = origin+"/"+name+".jpeg"
        # print(img_path)
        label_associated = name+".txt"
        # Loading image
        img = cv2.imread(img_path)
        if (os.path.isfile(origin+'/'+label_associated) == True):
            # Displaying the labels
            file = open(origin+'/'+label_associated, 'r')
            Lines = file.readlines()
            # Strips the newline character
            for line in Lines:
                x = int((float(line.split(" ")[1]) - float(line.split(" ")[3])/2 ) * img.shape[1])
                y = int((float(line.split(" ")[2]) - float(line.split(" ")[4])/2 ) * img.shape[0])
                w = int((float(line.split(" ")[3])) * img.shape[1])
                h = int((float(line.split(" ")[4])) * img.shape[0])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                cv2.putText(img, "Label", (x, y - 10), font, 1, (255,0,0), 2)
        if (os.path.isfile(origin+'/Prediction/'+label_associated) == True):
            # Displaying the predictions
            file = open(origin+'/Prediction/'+label_associated, 'r')
            Lines = file.readlines()
            # Strips the newline character
            for line in Lines:
                x = int((float(line.split(" ")[1]) - float(line.split(" ")[3])/2 ) * img.shape[1])
                y = int((float(line.split(" ")[2]) - float(line.split(" ")[4])/2 ) * img.shape[0])
                w = int((float(line.split(" ")[3])) * img.shape[1])
                h = int((float(line.split(" ")[4])) * img.shape[0])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2,)
                cv2.putText(img, "Prediction", (x, y - 10), font, 1, (0,0,255), 2)
        cv2.imwrite('./Images_predites/'+name+"_predite.jpeg", img) # Aperçu sur l'image originale
    
    
    os.system("mv ./Images_predites ./"+origin)
else:
    print("No jpeg images in directory")