<?php
header("Location: http://localhost:5000/stream");
$output = shell_exec("../../venv/bin/python3 ../../flask-video-stream-master/Detection.py");
echo $output;
?>