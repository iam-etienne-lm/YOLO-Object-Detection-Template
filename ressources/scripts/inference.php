<?php
header("Location: ../html/interface.php");
$output = shell_exec("cd ../../YOLO && bash inference.sh ".$_GET["conf_thres"]);
echo $output;
?>