<?php
header("Location: ../html/interface.php");
echo $_GET["conf_threshold"];
$output = shell_exec("cd ../../YOLO && bash inference.sh ".$_GET["conf_threshold"]);
echo $output;
?>