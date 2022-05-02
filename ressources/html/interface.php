<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AWS Project | Big Data</title>
    <link rel="icon" href="../logo.png" type="image/png" />
    <link
      type="text/css"
      rel="stylesheet"
      href="/ressources/css/interface.css"
    />
    
    <script src="https://kit.fontawesome.com/0c87a70838.js"></script>
  </head>

  <body>
    <div class="navbar">
      <ul>
        <li>
          <a class="active" href="/index.php"
            ><i class="fa fa-fw fa-home"></i> Home</a
          >
        </li>
        <li>
          <a href="/ressources/html/training.php"
            ><i class="fas fa-share-alt"></i> Training</a
          >
        </li>
        <li>
          <a href="/ressources/html/interface.php"
            ><i class="fas fa-share-alt"></i> Inference</a
          >
        </li>
        <li>
          <a href="/ressources/onnx/index_onnx.php"
            ><i class="fas fa-share-alt"></i> ONNX</a
          >
        </li>
        <li>
          <a href="/ressources/html/doc.php"
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
            action="../scripts/upload_pt.php"
            method="POST"
          >
            <input class="choose" name="userfile" type="file" />
            <div class="wrap">
              <button class="button" type="submit">Upload</button>
            </div>
          </form>
        </div>
        <div class="separation">
          <div class="middle-text">Upload images (multiple)</div>
          <form
            action="../scripts/upload_folder.php"
            enctype="multipart/form-data"
            method="post"
          >
            <input
              class="choose"
              id="upload"
              name="upload[]"
              type="file"
              multiple="multiple"
              accept="image/*.jpeg , image/*.jpg"
            />
            <div class="wrap">
              <button class="button" name="sub" type="submit">Upload</button>
            </div>
          </form>
        </div>
      </div>
        

      <div class="form-cols">
        <form method="GET" action="/ressources/scripts/inference.php">
          <div style="white-space: pre;">
            <label
              for="conf_threshold"
              style="
                color: white;
                font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande',
                  'Lucida Sans Unicode', Geneva, Verdana, sans-serif;
                margin: 15px;
              "
              >Choose a confidence threshold</label>
            <input id="conf_threshold" name="conf_threshold" type="number" step="0.05" max="1" min="0" value="0.4"/>
          </div>
            <button type="submit" class="btn btn-1 btn-sep icon-network">
              Inference
            </button>
        </form>
        <form method="GET" action="/ressources/scripts/metrics.php" target="_blank">
          <button type="submit" class="btn btn-2 btn-sep icon-metrics">
            Show Metrics
          </button>
        </form>
        <form method="GET" action="/ressources/scripts/clear_cache.php">
          <button type="submit" class="btn btn-3 btn-sep icon-clear">
            Clear Cache
          </button>
        </form>
      </div>
    </div>

    <div>
      <?php
    echo "<h2 style='color:white'> Inferences </h2>";
    foreach (glob("../../uploads/Testset/Images_predites/*.jpeg") as $filename)  {
        echo "</br>";
        echo "<img src='$filename' alt='$filename' style='width:800px ; height:auto'/>";
    }
    ?></div>

    <div class="footer">By Dorian VOYDIE</div>
  </body>
</html>
