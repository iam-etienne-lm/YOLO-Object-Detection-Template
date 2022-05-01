source ../venv/bin/activate
which python3
echo "--------"
which pip
# echo "--------"
# pip install -r yolov5/requirements.txt
echo "--------"
python3 Inference.py --origin ../uploads/Testset --weights ../uploads/weights.pt --conf_thres $1
echo "--------"
sudo chmod -R 777 /var/www/html