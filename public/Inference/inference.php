<?php
header("Location: ../ressources/html/interface.php");
$output = shell_exec("bash inference.sh ".$_GET["conf_thres"]);
echo $output;
?>