<?php
header("Location: http://localhost:6006");
// $output = shell_exec("cd ../../YOLO && bash metrics.sh");
$output = shell_exec("cd ../../YOLO && source ../../venv/bin/activate && nohup python3 -m tensorboard.main --logdir ./yolov5/runs/train &");
echo $output;
?>