<?php
header("Location: http://localhost:6006");
$output = shell_exec("cd ../../YOLO && xdg-open http://localhost:6006 && nohup ../venv/bin/python3 -m tensorboard.main --logdir ./yolov5/runs/train &");
echo $output;
?>