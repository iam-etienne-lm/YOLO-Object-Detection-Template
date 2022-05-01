<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AWS Project | Big Data</title>
    <link rel="icon" href="../logo.png" type="image/png" />
    <link
      type="text/css"
      rel="stylesheet"
      href="/ressources/css/doc.css"
    />
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
    />
    <script src="/ressources/js/script.js"></script>
    <script src="https://kit.fontawesome.com/0c87a70838.js"></script>
  </head>

  <body>
    <div class="navbar">
      <ul>
        <li>
          <a class="active" href="/index.php"
            ><i class="fa fa-fw fa-home"></i>Home</a
          >
        </li>
        <li>
          <a href="/ressources/html/training.php">
            <i class="fas fa-share-alt"></i>Training</a
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
      <h1 id="debug">Debuggage</h1>

      <p>
        Une fois les requirements installés, pour débugger, utilisez votre
        terminal :
      </p>
      <pre>
        <code class="lang-bash">
        source ./venv/bin/activate
        which python3
        which pip
        # pip install -r yolov5/requirements.txt # Normalement déjà installés (pytorch / opencv / ...)
        python3 ./YOLO/YOLO.py --origin ./YOLO/Testset --weights ./uploads/weights.pt --conf_thres 0.4 # Testez l'inférence
        sudo chmod -R 777 /var/www/html # Donner les droits admin au nouveaux fichiers
        </code>
      </pre>
    </div>

    <div class="footer">By Dorian VOYDIE</div>
  </body>
</html>
