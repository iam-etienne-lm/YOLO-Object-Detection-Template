# Index

## Project layout

    Dataset/
        Dataset/ # Folder in which you place images and labels associated you want to make inference on
        DatasetAugmented/ # Dataset *8 using orientation/flip/transpose/inverse transformations
        DatasetAugmented_VX/ # Filtered dataset using X filter
    Inference/
        Testset/ # Batch of labelised images for inference and evaluation
        mAP/ # Library for mAP computing
        yolov5/ # Library for handling yolov5 computation
    modules/ # Contains filters, scripts to change images format and requirements
    Training/
        yolov5/ # Library for handling yolov5 computation

## Requirements

- `pip install modules/requirements.txt` - Install requirements

## Notebooks

    Dataset/
    - `Bbox_augmentation.ipynb` - Get 3 transformations + all inverse images in separate subfolders
    - `Filter_images.ipynb` - Apply filters and save images in separate folder with Train / Val / Test split for training
    - `Training.ipynb` - Training a Yolo algorithm
    Inference/
    - `Inference.py` - Creates images predictions given a weights file and a testset
    - `Evaluate.py` - Evaluates the last inference done on a specific testset
    Training/
    - `Training.ipynb` - Colab notebook for training yolo algorithm

## GitLab-CI

```powershell
stages:
  - inference_run

import_pytorch_ngc:
  stage: inference_run
  tags:
    - nano
  image: nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
  before_script:
    - pip3 install --upgrade setuptools pip
    - pip3 install -U -r modules/requirements.txt
    - python3 modules/convert_jpg_to_jpeg.py --origin Testset
  script:
    - python3 Inference.py --origin Testset --weights yolov5/runs/train/Z_V5_yoloS/weights/best.pt --conf_thres 0.1
    - python3 Evaluate.py --origin Testset
    - python3 Inference.py --origin Testset --weights yolov5/runs/train/Z_V5_yoloL/weights/best.pt --conf_thres 0.1
    - python3 Evaluate.py --origin Testset
    - python3 Inference.py --origin Testset --weights yolov5/runs/train/X_V5_yoloS/weights/best.pt --conf_thres 0.1
    - python3 Evaluate.py --origin Testset
    - python3 Inference.py --origin Testset --weights yolov5/runs/train/Y_V5_yoloS/weights/best.pt --conf_thres 0.1
    - python3 Evaluate.py --origin Testset
    - python3 Inference.py --origin Testset --weights yolov5/runs/train/GAMMA_V5_yoloS/weights/best.pt --conf_thres 0.1
    - python3 Evaluate.py --origin Testset
```

# LabelImg

---

LabelImg is a graphical image annotation tool.

It is written in Python and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, the format used
by <a href="http://www.image-net.org/" >ImageNET</a>. Besides, it also supports YOLO and CreateML formats.

<center>

![image](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)

![image](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo.jpg)

</center>

<a href="https://youtu.be/p0nR2YsCY_U">Watch a demo video </a>

## Installation

Get from PyPI but only python3.0 or above
This is the simplest (one-command) install method on modern Linux distributions such as Ubuntu and Fedora.

```bash
pip3 install labelImg
labelImg
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

Build from source

Linux/Ubuntu/Mac requires at least <a href="https://www.python.org/getit/">Python 2.6 </a> and has been tested with <a href="https://www.riverbankcomputing.com/software/pyqt/intro">PyQt 4.8</a>. However, <a href="https://www.python.org/getit/">Python 3 or above </a> and <a href="https://pypi.org/project/PyQt5/">PyQt5</a> are strongly recommended.

### Ubuntu Linux

Python 3 + Qt5

```bash
sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

### Max OS

Python 3 + Qt5

```bash
brew install qt  # Install qt-5.x.x by Homebrew
brew install libxml2

or using pip

pip3 install pyqt5 lxml # Install qt and lxml by pip

make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

Python 3 Virtualenv (Recommended)

Virtualenv can avoid a lot of the QT / Python version issues

```bash
brew install python3
pip3 install pipenv
pipenv run pip install pyqt5==5.15.2 lxml
pipenv run make qt5py3
pipenv run python3 labelImg.py
[Optional] rm -rf build dist; python setup.py py2app -A;mv "dist/labelImg.app" /Applications
```

Note: The Last command gives you a nice .app file with a new SVG Icon in your /Applications folder. You can consider using the script: build-tools/build-for-macos.sh

### Windows

Install <a href="https://www.python.org/downloads/windows/">Python</a>,<a href="https://www.riverbankcomputing.com/software/pyqt/download5">PyQt5</a> and install <a href="http://lxml.de/installation.html">lxml</a>.

Open cmd and go to the labelImg directory

```bash
pyrcc4 -o libs/resources.py resources.qrc
For pyqt5, pyrcc5 -o libs/resources.py resources.qrc

python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

If you want to package it into a separate EXE file

```bash
Install pyinstaller and execute:

pip install pyinstaller
pyinstaller --hidden-import=pyqt5 --hidden-import=lxml -F -n "labelImg" -c labelImg.py -p ./libs -p ./
```

### Windows + Anaconda

Download and install <a href="https://www.anaconda.com/download/#download">Anaconda</a> (Python 3+)

Open the Anaconda Prompt and go to the labelImg directory

```bash
conda install pyqt=5
conda install -c anaconda lxml
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

Use Docker

```bash
docker run -it \
--user $(id -u) \
-e DISPLAY=unix$DISPLAY \
--workdir=$(pwd) \
--volume="/home/$USER:/home/$USER" \
--volume="/etc/group:/etc/group:ro" \
--volume="/etc/passwd:/etc/passwd:ro" \
--volume="/etc/shadow:/etc/shadow:ro" \
--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
tzutalin/py2qt4

make qt4py2;./labelImg.py
```

You can pull the image which has all of the installed and required dependencies. <a href="https://youtu.be/nw1GexJzbCI">Watch a demo video </a>

## Usage

### Steps (YOLO)

```md
1. In `data/predefined_classes.txt` define the list of classes that will be used
   for your training (for us it will only contain "Scratch")
2. Build and launch using the instructions above.
3. Right below "Save" button in the toolbar, click "PascalVOC" button to switch to YOLO format.
4. You may use Open/OpenDIR to process single or multiple images. When finished with a single image, click save.
```

A txt file of YOLO format will be saved in the same folder as your image with same name.<br>
A file named "classes.txt" is saved to that folder too. "classes.txt" defines the list of class names that <br>
your YOLO label refers to.

**Note:**

```md
- Your label list shall not change in the middle of processing a list of images.
  When you save an image, classes.txt will also get updated, while previous annotations will not be updated.

- You shouldn't use "default class" function when saving to YOLO format, it will not be referred.

- When saving as YOLO format, "difficult" flag is discarded.

Create pre-defined classes
```

You can edit the
<a href="https://github.com/tzutalin/labelImg/blob/master/data/predefined_classes.txt">data/predefined_classes.txt </a>
to load pre-defined classes

### Annotation visualization

```md
1. Copy the existing lables file to same folder with the images. The labels file name must be same with image file name.
2. Click File and choose 'Open Dir' then Open the image folder.
3. Select image in File List, it will appear the bounding box and label for all objects in that image.

(Choose Display Labels mode in View to show/hide lablels)
```

## Annotation format

Your data is going to be normalized (between 0 and 1), so whatever the scale of your image is the annotation still fits it.
Your annotations will have this format :

```md
Class_number x_center_box y_center_box width_box height_box
```

**Important : Here we have only 1 classe so verify that Class_number = 0**<br>
If Class_number ≠ 0, verify the predefined classes in the <a href="https://github.com/tzutalin/labelImg/blob/master/data/predefined_classes.txt">settings</a>

## Hotkeys

| Shortcut         | Effect                                   |
| ---------------- | ---------------------------------------- |
| Ctrl+u           | Load all of the images from a directory  |
| Ctrl + r         | Change the default annotation target dir |
| Ctrl + s         | Save                                     |
| Ctrl + d         | Copy the current label and rect box      |
| Ctrl + Shift + d | Delete the current image                 |
| Space            | Flag the current image as verified       |
| w                | Create a rect box                        |
| d                | Next image                               |
| a                | Previous image                           |
| del              | Delete the selected rect box             |

**Verify Image:**

When pressing space, the user can flag the image as verified, a green background will appear.
This is used when creating a dataset automatically, the user can then through all the pictures and flag them instead of annotate them.

**Difficult:**

The difficult field is set to 1 indicates that the object has been annotated as "difficult", for example, an object which is clearly visible but difficult to recognize without substantial use of context.
According to your deep neural network implementation, you can include or exclude difficult objects during training.

How to reset the settings

```md
In case there are issues with loading the classes, you can either:

1. From the top menu of the labelimg click on Menu/File/Reset All
2. Remove the ".labelImgSettings.pkl" from your home directory. In Linux and Mac you can do:
   rm ~/.labelImgSettings.pkl
```

License

<a href="https://github.com/tzutalin/labelImg/blob/master/LICENSE">Free software: MIT license </a>

Citation: Tzutalin. LabelImg. Git code (2015). <a href="https://github.com/tzutalin/labelImg">https://github.com/tzutalin/labelImg</a>

Related and additional tools

1. <a href="https://github.com/tzutalin/ImageNet_Utils">ImageNet Utils </a> to
   download image, create a label text for machine learning, etc
2. <a href="https://hub.docker.com/r/tzutalin/py2qt4">Use Docker to run labelImg </a>
3. <a href="https://github.com/tensorflow/models/blob/4f32535fe7040bb1e429ad0e3c948a492a89482d/research/object_detection/g3doc/preparing_inputs.md#generating-the-pascal-voc-tfrecord-files">Generating the PASCAL VOC TFRecord files </a>
4. <a href="https://www.elegantthemes.com/">App Icon based on Icon by Nick Roach (GPL) </a>
5. <a href="https://tzutalin.blogspot.com/2019/04/set-up-visual-studio-code-for-python-in.html">Setup python development in vscode </a>
6. <a href="https://code.ihub.org.cn/projects/260/repository/labelImg">The link of this project on iHub platform </a>
7. <a href="https://github.com/tzutalin/labelImg/tree/master/tools">Convert annotation files to CSV format or format for Google Cloud AutoML </a>

# Dataset

---

Get started with a folder of images which you labeled using <a href=/labeling>LabelImg</a> :

    TRAIN_DATASET/ # Folder in which you place all images and labels associated you watn to make inference on.
        classes.txt # Text file with only "Scratch" in it

        capture0.jpeg  # Image 0
        capture0.txt  # Label 0 associated to image 0 (format : "0 | x | y | w | h")

        capture1.jpeg  # Image 1
        capture1.txt  # Label 1 associated to image 1 (format : "0 | x | y | w | h")

        # ...
        # ...

## Data Augmentation

You will first deal with the file **Bbox_augmentation.ipynb**

Change the path in cell 2 to the folder containing the images and the labels you want to be augmented :

```python
# Chemin du dossier qui contient les images et les bounding boxs
PATH = 'TRAIN_DATASET'+"/"
```

You can now execute the whole script, it will automatically create **4 subfolders**
in which you will find the corresponding images and labels transformed.

    TRAIN_DATASET/ # Folder in which you place all images and labels associated you watn to make inference on.
        HorizontalFlip/ # Folder with horizontal transformation
            capture0_H.jpeg  # Flipped horizontal
            capture0_H.txt  # Flipped horizontal
        Invert/ # Folder with all other images inverted
            capture0_I.jpeg # Same
            capture0_I.txt # Same
            capture0_H_I.jpeg  # Flipped horizontal
            capture0_H_I.txt  # Flipped horizontal
            capture0_V_I.jpeg  # Flipped vertical
            capture0_V_I.txt  # Flipped vertical
            capture0_T_I.jpeg  # Transpose
            capture0_T_I.txt  # Transpose
        Transpose/ # Folder with transposed transformation
            capture0_T.jpeg  # Transpose
            capture0_T.txt  # Transpose
        VerticalFlip/ # Folder with vertical transformation
            capture0_V.jpeg  # Flipped vertical
            capture0_V.txt  # Flipped vertical

        classes.txt # Same

        capture0.jpeg # Same
        capture0.txt # Same

        # ...
        # ...

**At the end, if you do the 4 transformations you will have 8 times the amount of images and labels you had to begin with.**

## Filtering

Then, you will have to apply a filter on the images (same one as in production)
on the training images so the model has the same root for the training as for the inference.

First, gather all the images and labels that you augmented _(if you want them)_ in 1 folder :

    TRAIN_DATASET/ # Folder in which you place all images and labels associated you watn to make inference on.
        classes.txt # Text file with only "Scratch" in it

        capture0.jpeg  # Original
        capture0.txt  # Original
        capture0_H.jpeg  # Flipped horizontal
        capture0_H.txt  # Flipped horizontal
        capture0_V.jpeg  # Flipped vertical
        capture0_V.txt  # Flipped vertical
        capture0_T.jpeg  # Transpose
        capture0_T.txt  # Transpose

        # Note : note taking the inverted images and labels here for instance

        capture1.jpeg  # Original
        capture1.txt  # Original
        capture1_H.jpeg  # Flipped horizontal
        # ...
        # ...

Then you can open the file **Filter_images.ipynb** :<br>

1. Make sure you change in cell 2 the **origin** of your dataset and the **filter_version** you want to use.<br>
2. You can find de filters in the file **/utils/filters.py** or in the <a href=/cod>Codification Section</a>

## Output folder for training

Assume that you choose **TRAIN_DATASET** as origin and V5 as **filter_version**.<br>
You will end up with a new folder like this :

    TRAIN_DATASET_V5/
        Trainset/
        Testset/
        Valset/
        config.yml

Here is what the config.yml must contain :

```yml
train: ../TRAIN_DATASET_V5/Trainset # Path to Trainset
val: ../TRAIN_DATASET_V5/Valset # Path to Valset
nc: 1 # Only 1 class
names: ["Scratch"] # Name of the class
```

#### Now your data is officially shaped to be given to a Yolo model.

<div style="color:red">
Note that the Testset isn't taken in consideration by Yolo. This batch of the dataset is kept aside in order to evaluate the model.
</div>

# Training

---

1. Choose a model from n to x :

| Model                      | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| -------------------------- | --------------------- | ----------------------- | ------------------ | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n]                  | 640                   | 28.0                    | 45.7               | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| [YOLOv5s]                  | 640                   | 37.4                    | 56.8               | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| [YOLOv5m]                  | 640                   | 45.4                    | 64.1               | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| [YOLOv5l]                  | 640                   | 49.0                    | 67.3               | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| [YOLOv5x]                  | 640                   | 50.7                    | 68.9               | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                            |                       |                         |                    |                              |                               |                                |                    |
| [YOLOv5n6]                 | 1280                  | 36.0                    | 54.4               | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| [YOLOv5s6]                 | 1280                  | 44.8                    | 63.7               | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| [YOLOv5m6]                 | 1280                  | 51.3                    | 69.3               | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| [YOLOv5l6]                 | 1280                  | 53.7                    | 71.3               | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| [YOLOv5x6]<br>+ [TTA][tta] | 1280<br>1536          | 55.0<br>**55.8**        | 72.7<br>**72.7**   | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |

## Locally

```bash
python yolov5/train.py
    --data TRAIN_DATASET_V5/config.yaml # Path to the "config.yml" of your Train/Val set
    --weights yolov5/models/yolov5s.pt # Default yolov5s
            # Just replace the "s" with the model you want to use
            # Note : you can also use custom weights in this path (ex pre-trained models)
    --batch 32 # Default 16
    --epochs 300 # Default 300
```

## Using Google Colab

1. Copy the whole project to your Google Drive<br>
2. Open the **Training.ipynb** file.<br>
3. Adapt the first 6 cells paths to match your Drive account

You can then modify the _training line_ as explained in [the previous paragraph](#locally)

## Debugging

Here is what the output should look like if the training launched successfully :

```python
Train: weights=yolov5/models/yolov5s.pt, cfg=, data=TRAIN_DATASET_V5/config.yaml,
hyp=yolov5/data/hyps/hyp.scratch.yaml, epochs=300, batch_size=32, imgsz=640, rect=False,
resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None,
image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False,
workers=8, project=yolov5/runs/train, name=exp, exist_ok=False, quad=False, linear_lr=False,
label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, local_rank=-1, entity=None, upload_dataset=False,
bbox_interval=-1, artifact_alias=latest

Github: skipping check (not a git repository), for updates see https://github.com/ultralytics/yolov5

Hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0,
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0,
fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0

Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)

TensorBoard: Start with 'tensorboard --logdir yolov5/runs/train', view at http://localhost:6006/

Overriding model.yaml nc=80 with nc=1

                 from  n    params  module                                  arguments
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]
  2                -1  1     18816  models.common.C3                        [64, 64, 1]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]
  4                -1  2    115712  models.common.C3                        [128, 128, 2]
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]
  6                -1  3    625152  models.common.C3                        [256, 256, 3]
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  models.common.Concat                    [1]
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  models.common.Concat                    [1]
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]
 19          [-1, 14]  1         0  models.common.Concat                    [1]
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]
 22          [-1, 10]  1         0  models.common.Concat                    [1]
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]
 24      [17, 20, 23]  1     16182  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 270 layers, 7022326 parameters, 7022326 gradients, 15.8 GFLOPs

Transferred 343/349 items from yolov5/models/yolov5s.pt
Scaled weight_decay = 0.0005
Optimizer: SGD with parameter groups 57 weight (no decay), 60 weight, 60 bias
Albumentations: Blur(always_apply=False, p=0.01, blur_limit=(3, 7)), MedianBlur(always_apply=False,
p=0.01, blur_limit=(3, 7)), ToGray(always_apply=False, p=0.01), CLAHE(always_apply=False, p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))

Train: Scanning '/content/drive/MyDrive/Atos/yolov5/../TRAIN_DATASET_V5/Trainset.cache' images and labels...
    3237 found, 5 missing, 30 empty, 0 corrupt: 100% 3242/3242 [00:00<?, ?it/s]
Train: New cache created: /Users/dorianvoydie/Documents/team2_pytorch/yolov5/../TRAIN_DATASET_V5/Trainset.cache
Val: Scanning '/content/drive/MyDrive/Atos/yolov5/../TRAIN_DATASET_V5/Valset.cache' images and labels...
    812 found, 0 missing, 6 empty, 0 corrupt: 100% 812/812 [00:00<?, ?it/s]
Val: New cache created: /Users/dorianvoydie/Documents/team2_pytorch/yolov5/../TRAIN_DATASET_V5/Valset.cache

Plotting labels to yolov5/runs/train/exp/labels.jpg...

AutoAnchor: 4.11 anchors/target, 0.976 Best Possible Recall (BPR).
Anchors are a poor fit to dataset ⚠️, attempting to improve...
AutoAnchor: WARNING: Extremely small objects found. 4 of 292 labels are < 3 pixels in size.
AutoAnchor: Running kmeans for 9 anchors on 292 points...
AutoAnchor: Evolving anchors with Genetic Algorithm: fitness = 0.6477: 100%|█| 1
AutoAnchor: thr=0.25: 0.9795 best possible recall, 3.54 anchors past thr
AutoAnchor: n=9, img_size=640, metric_all=0.252/0.651-mean/best, past_thr=0.455-mean: 15,23, 46,24, 55,70, 27,150, 158,83, 64,239, 297,51, 120,207, 338,309
AutoAnchor: Original anchors better than new anchors. Proceeding with original anchors.
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to yolov5/runs/train/exp


Starting training for 300 epochs...

```

I advise you to check that most of your data has been taken in account by the model in this section of the debugging :

```bash
Train: Scanning '/content/drive/MyDrive/Atos/yolov5/../TRAIN_DATASET_V5/Trainset.cache' images and labels...
    3237 found, 5 missing, 30 empty, 0 corrupt: 100% 3242/3242 [00:00<?, ?it/s]
Train: New cache created: /Users/dorianvoydie/Documents/team2_pytorch/yolov5/../TRAIN_DATASET_V5/Trainset.cache
Val: Scanning '/content/drive/MyDrive/Atos/yolov5/../TRAIN_DATASET_V5/Valset.cache' images and labels...
    812 found, 0 missing, 6 empty, 0 corrupt: 100% 812/812 [00:00<?, ?it/s]
Val: New cache created: /Users/dorianvoydie/Documents/team2_pytorch/yolov5/../TRAIN_DATASET_V5/Valset.cache
```

## Output data

Let's say your training ended.

As shown in the [debugging](#debugging) at the end, your experiment will remain in the folder :

```bash
yolov5/runs/train/
```

It will have a name like **exp**, **exp1** or **expX** according to the number of training you launched.<br>
As soon as a training has ended, i advise you to rename this folder as you wish.

## Displaying Training Metrics

The displaying of the training metrics is the easiest part. I advise you to use <a href="https://www.tensorflow.org/tensorboard/get_started">**Tensorboard**</a> :

```python
# Load the TensorBoard notebook extension
%load_ext tensorboard
%tensorboard --logdir yolov5/runs/train --host localhost --port 8088
```

Notice that Tensorboard will have access to all training experiments (folders) you have done in _yolov5/runs/train_<br>
By renaming the experiments you will have a nicer view on tensorboard as you can filter the experiments you want to show.

_If you are using a notebook the interface should be displayed in the output of the cell.<br>
If it is not, check **localhost:8000** in your favorite browser._

# Inference.py

---

## Command

```bash
python Inference.py
  --origin [FOLDER] # Replace FOLDER by your /Testset containing images and labels
  --weights [WEIGHTS_FILE] # Get your weights best.pt file from the "yolov5/runs/train/exp" folder
  --conf_thres [CONF_THRES] # I advise using 0.1
```

## Debugging

```bash
        ____       __            __  _                    __
       / __ \___  / /____  _____/ /_(_)___  ____     ____/ /__     _________ ___  ____  __________  _____
      / / / / _ \/ __/ _ \/ ___/ __/ / __ \/ __ \   / __  / _ \   / ___/ __ `/ / / / / / / ___/ _ \/ ___/
     / /_/ /  __/ /_/  __/ /__/ /_/ / /_/ / / / /  / /_/ /  __/  / /  / /_/ / /_/ / /_/ / /  /  __(__  )
    /_____/\___/\__/\___/\___/\__/_/\____/_/ /_/   \__,_/\___/  /_/   \__,_/\__, /\__,_/_/   \___/____/
                                                                           /____/
```

Il y a 20 images de test<br>
Voici un extrait :

0 Testset/capture12.jpeg<br>
1 Testset/capture6.jpeg<br>
2 Testset/capture7.jpeg<br>
3 Testset/capture13.jpeg<br>
4 Testset/capture14.jpeg<br>

```bash
                  ____                           __  _                __  ___          __     __
                 /  _/___ ___  ____  ____  _____/ /_(_)___  ____ _   /  |/  /___  ____/ /__  / /
                 / // __ `__ \/ __ \/ __ \/ ___/ __/ / __ \/ __ `/  / /|_/ / __ \/ __  / _ \/ /
               _/ // / / / / / /_/ / /_/ / /  / /_/ / / / / /_/ /  / /  / / /_/ / /_/ /  __/ /
              /___/_/ /_/ /_/ .___/\____/_/   \__/_/_/ /_/\__, /  /_/  /_/\____/\__,_/\___/_/
                           /_/                           /____/
```

Fusing layers...<br>
Model Summary: 213 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs<br>
Loading Model took : 0.2018270492553711<br>

```bash
     ______                            __  _                                    ___      __  _
    / ____/___  ____ ___  ____  __  __/ /_(_)___  ____ _   ____  ________  ____/ (_)____/ /_(_)___  ____  _____
   / /   / __ \/ __ `__ \/ __ \/ / / / __/ / __ \/ __ `/  / __ \/ ___/ _ \/ __  / / ___/ __/ / __ \/ __ \/ ___/
  / /___/ /_/ / / / / / / /_/ / /_/ / /_/ / / / / /_/ /  / /_/ / /  /  __/ /_/ / / /__/ /_/ / /_/ / / / (__  )
  \____/\____/_/ /_/ /_/ .___/\__,_/\__/_/_/ /_/\__, /  / .___/_/   \___/\__,_/_/\___/\__/_/\____/_/ /_/____/
                      /_/                      /____/  /_/

           ______            ____   __  __                   __          __    __        ____   ___
          / ____/___  ____  / __/  / /_/ /_  ________  _____/ /_  ____  / /___/ /  _    / __ \ <  /
         / /   / __ \/ __ \/ /_   / __/ __ \/ ___/ _ \/ ___/ __ \/ __ \/ / __  /  (_)  / / / / / /
        / /___/ /_/ / / / / __/  / /_/ / / / /  /  __(__  ) / / / /_/ / / /_/ /  _    / /_/ / / /
        \____/\____/_/ /_/_/     \__/_/ /_/_/   \___/____/_/ /_/\____/_/\__,_/  (_)   \____(_)_/
```

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00, 2.05it/s]

### Le temps moyen d'inference est : 0.49 s

```bash
                      _____             _                _
                     / ___/____ __   __(_)___  ____ _   (_)___ ___  ____ _____ ____  _____
                     \__ \/ __ `/ | / / / __ \/ __ `/  / / __ `__ \/ __ `/ __ `/ _ \/ ___/
                    ___/ / /_/ /| |/ / / / / / /_/ /  / / / / / / / /_/ / /_/ /  __(__  )
                   /____/\__,_/ |___/_/_/ /_/\__, /  /_/_/ /_/ /_/\__,_/\__, /\___/____/
                                            /____/                     /____/
```

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:02<00:00, 8.65it/s]

# Evaluate.py

---

## Command

```bash
python Evaluate.py --origin [FOLDER]
```

**Important : Delete the "classes.txt" file from the FOLDER if there is one.<br>
Keeping it will give you <u>wrong results</u>.**

## Debugging

```bash
rm: ./mAP/input/detection-results/backup: is a directory
rm: ./mAP/input/ground-truth/backup: is a directory
rm: ./mAP/input/ground-truth/backup_no_matches_found: is a directory
./Testset/Prediction/
Conversion completed!
Conversion completed!
total ground-truth files: 20
total detection-results files: 20

No backup required for /Users/dorianvoydie/Documents/team2_pytorch/mAP/input/ground-truth
No backup required for /Users/dorianvoydie/Documents/team2_pytorch/mAP/input/detection-results
total intersected files: 20
Intersection completed!

91.17% = Scratch AP

mAP = 91.17%
```
