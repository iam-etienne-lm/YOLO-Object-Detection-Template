<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AWS Project | Big Data</title>
    <link rel="icon" href="img/logo.png" type="image/png" />
    <link
      type="text/css"
      rel="stylesheet"
      href="/public/ressources/css/interface.css"
    />
    
    <script src="https://kit.fontawesome.com/0c87a70838.js"></script>
  </head>

  <body>
    <div class="navbar">
      <ul>
        <li>
          <a class="active" href="/public/index.php"
            ><i class="fa fa-fw fa-home"></i> Home</a
          >
        </li>
        <li>
          <a href="/public/ressources/html/interface.php"
            ><i class="fas fa-share-alt"></i> Interface</a
          >
        </li>
        <li>
          <a href="/public/ressources/onnx/index_onnx.php"
            ><i class="fas fa-share-alt"></i> ONNX</a
          >
        </li>
        <li>
          <a href="/public/ressources/html/doc.php"
            ><i class="fas fa-book"></i> Documentation</a
          >
        </li>
      </ul>
    </div>

    <div class="content">
      <div class="form-cols form-cols-bis">
        <div class="separation">
          <div class="middle-text">Upload .pt file</div>
          <form
            enctype="multipart/form-data"
            action="upload_pt.php"
            method="POST"
          >
            <input class="choose" name="userfile" type="file" />
            <div class="wrap">
              <button class="button" type="submit">Upload</button>
            </div>
          </form>
        </div>
        <div class="loader" id="loader">
          <img
            width="50%"
            height="50%"
            src="/public/ressources/Circle-Loading.svg"
            alt="loading"
          />
        </div>
        <div class="separation">
          <div class="middle-text">Upload images (multiple)</div>
          <form
            action="upload_folder.php"
            enctype="multipart/form-data"
            method="post"
          >
            <input
              class="choose"
              id="upload"
              name="upload[]"
              type="file"
              multiple="multiple"
              accept="image/*.jpeg, image/*.jpg, image/*.png"
            />
            <div class="wrap">
              <button class="button" name="sub" type="submit">Upload</button>
            </div>
          </form>
        </div>
      </div>

      <form method="GET" action="/public/Inference/inference.php">
        <button type="submit" class="btn btn-4 btn-sep icon-dwnl">
          Compute Inferences
        </button>
      </form>
    </div>

    <div><?php
    echo "<h2 style='color:white'> Inferences </h2>";

    foreach (glob("../../Inference/Testset/Images_predites/*.jpeg") as $filename)  {
        echo "<p style='color:white'>$filename</p>";
        echo "<img src='$filename' alt='$filename' style='width:800px ; height:auto'/>";
    }
    ?></div>

    <div class="footer">By Dorian VOYDIE</div>
  </body>
</html>
