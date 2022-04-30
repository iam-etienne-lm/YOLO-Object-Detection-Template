import time

""" IMPORTS """
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.torch_utils import select_device
""" -------- """

def run_model(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # check_requirements(exclude=('tensorboard', 'thop'))
    # Load model
    device = select_device(device)
    start = time.time()
    from models.common import DetectMultiBackend
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    print('Loading Model took :', str(time.time() - start))
    return(model)