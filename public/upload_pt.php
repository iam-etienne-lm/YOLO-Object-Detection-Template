<?php
    // header("Location: interface.html");
    $uploadfile = '/var/www/html/uploads/weights.pt';

    echo "<p>";
    echo "Moving : ";
    echo $_FILES['userfile']['tmp_name'];
    echo "<p>";
    echo "to : ";
    echo $uploadfile;
    echo "<p>";
    echo "I am : ";
    echo exec('whoami'); 
    echo "<p>";

    if (move_uploaded_file($_FILES['userfile']['tmp_name'], $uploadfile)) {
        echo "File is valid, and was successfully uploaded.\n";
    } else {
        echo "Upload failed";
    }

    echo "</p>";
    echo '<pre>';
    echo 'Here is some more debugging info:';
    echo "</p>";
    print_r($_FILES);
?>