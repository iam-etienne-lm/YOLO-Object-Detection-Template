import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

""" IMPORTS """
import os
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.datasets import LoadImages
from utils.general import (check_requirements, check_img_size, increment_path, scale_coords, non_max_suppression, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import select_device
""" -------- """


def inference(
    source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
    imgsz=[640, 640],  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt=True,  # save results to *.txt
    save_conf=True,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    line_thickness=3,  # bounding box thickness (pixels)
    half=False,  # use FP16 half-precision inference
    model=None,
    project=ROOT / '..',  # 'runs/detect' save results to project/name
    name='Prediction',  # save results to project/name
    exist_ok=True,  # existing project/name ok, do not increment
    ):
    if model==None:
        print("You need to insert a model to make an inference")
        return
    # check_requirements(exclude=('tensorboard', 'thop'))
    device = select_device(device)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        
        ################ ORIGINAL
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
            # Stream results
            im0 = annotator.result()
    return