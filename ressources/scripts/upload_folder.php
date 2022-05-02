<?php
header("Location: ../html/interface.php");
if (isset($_POST['sub'])) {
    if (count($_FILES['upload']['name']) > 0) {
     for ($i=0; $i<count($_FILES['upload']['name']); $i++) {
      $tmpFilePath = $_FILES['upload']['tmp_name'][$i];
      if ($tmpFilePath != "") {
        $shortname = $_FILES['upload']['name'][$i];
        // $filePath = "/var/www/html/YOLO/Testset/".$_FILES['upload']['name'][$i];
        $filePath = "../../uploads/Testset/".$_FILES['upload']['name'][$i];
        if (move_uploaded_file($tmpFilePath, $filePath)) {
          $files[] = $shortname;                    
        }
      }
    }
  }
  echo "<h1>Uploaded:</h1>";    
  if(is_array($files)){
    echo "<ul>";
    foreach($files as $file){
      echo "<li>$file</li>";
    }
    echo "</ul>";
  }
  $output = shell_exec("cd ../../YOLO/modules &&  python3 ../../YOLO/modules/convert_jpg_to_jpeg.py --origin ../../uploads/Testset && sudo chmod 777 -R /var/www/html");
  echo $output;
}
?>