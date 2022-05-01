<?php
header("Location: ../html/interface.php");

// Clear only all files in "Testset" folder
$files = glob('../../uploads/Testset/*'); // get all file names
foreach($files as $file){ // iterate files
    echo $file . "<br />"; // display file name
    if(is_file($file)) {
        echo $file . "<br />"; // display file name
        unlink($file); // delete file
    }
}

// Clear only all folders in "Testset" folder
$folders = glob('../../uploads/Testset/*'); // get all files names
foreach($folders as $folder){ // iterate files
  if (is_dir($folder)) {
    $objects = scandir($folder);
    foreach ($objects as $object) {
      if ($object != "." && $object != "..") {
        if (filetype($folder."/".$object) == "dir") 
            rrmdir($folder."/".$object); 
        else unlink   ($folder."/".$object);
      }
    }
    reset($objects);
    rmdir($folder);
  }
}
?>