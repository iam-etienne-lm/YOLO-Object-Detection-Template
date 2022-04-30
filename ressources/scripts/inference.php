<?php
header("Location: ./interface.php");
$output = shell_exec("cd ../../Inference && bash inference.sh ".$_GET["conf_thres"]);
echo $output;
?>