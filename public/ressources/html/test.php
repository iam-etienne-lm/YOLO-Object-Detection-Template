<?php
echo "<h3> Inferences </h3>";

foreach (glob("../../Inference/Testset/Images_predites/*.jpeg") as $filename)  {
    echo "<p>$filename</p>";
    echo "<img src='$filename' alt='$filename' style='width:400px ; height:auto'/>";
}
?>