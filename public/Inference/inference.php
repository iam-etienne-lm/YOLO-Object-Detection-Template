<?php
header("Location: ../interface.php");
$output = shell_exec("bash inference.sh");
echo $output;
?>