import cv2
import numpy as np

def gray_rescaled(img_path):
    # RESIZE
    img = cv2.imread(img_path)
    width = 640
    height = 640
    dsize = (width, height)
    img = cv2.resize(img, dsize, interpolation = cv2.INTER_CUBIC)
    # img = cv2.edgePreservingFilter(img, sigma_s=0, sigma_r=1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def filter_V1(img_path):
    # RESIZE
    img = cv2.imread(img_path)
    width = 640
    height = 640
    dsize = (width, height)
    img = cv2.resize(img, dsize)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def display_labels(img,img_path):
    font = cv2.FONT_HERSHEY_PLAIN
    file = open(img_path[:-5]+'.txt', 'r')
    Lines = file.readlines()
    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        x = int((float(line.split(" ")[1]) - float(line.split(" ")[3])/2 ) * img.shape[1])
        y = int((float(line.split(" ")[2]) - float(line.split(" ")[4])/2 ) * img.shape[0])
        w = int((float(line.split(" ")[3])) * img.shape[1])
        h = int((float(line.split(" ")[4])) * img.shape[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(img, "Label", (x, y - 10), font, 0.4, (255,0,0), 2)