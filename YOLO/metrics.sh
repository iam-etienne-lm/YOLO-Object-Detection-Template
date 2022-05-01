source ../venv/bin/activate
which python3
echo "--------"
which pip
# echo "--------"
# pip install -r yolov5/requirements.txt
echo "--------"
nohup python3 -m tensorboard.main --logdir ./yolov5/runs/train &