<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AWS Project | Big Data</title>
    <link rel="icon" href="ressources/logo.png" type="image/png" />
    <link
      type="text/css"
      rel="stylesheet"
      href="{{ url_for('static',filename='styles/index.css') }}"
    />
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
    />
    <script src="https://kit.fontawesome.com/0c87a70838.js"></script>
  </head>

  <body>
    <div class="navbar">
      <ul>
        <li>
          <a class="active" href="http://localhost/index.php"
            ><i class="fa fa-fw fa-home"></i> Home</a
          >
        </li>
        <li>
          <a href="http://localhost/ressources/html/training.php"
            ><i class="fas fa-share-alt"></i> Training</a
          >
        </li>
        <li>
          <a href="http://localhost/ressources/html/interface.php"
            ><i class="fas fa-share-alt"></i> Inference</a
          >
        </li>
        <li>
          <a href="http://localhost/ressources/onnx/index_onnx.php"
            ><i class="fas fa-share-alt"></i> ONNX</a
          >
        </li>
        <li>
          <a href="http://localhost/ressources/html/doc.php"
            ><i class="fas fa-book"></i> Documentation</a
          >
        </li>
      </ul>
    </div>
    <div class="content">
      <h1 id="title">Lancer la webcam</h1>

      <p>Pour lancer la webcam, ouvrez votre terminal, dans le dossier du projet tapez :</p>
      <pre>
        <code class="lang-bash">
          /venv/bin/python3 /flask-video-stream-master/Detection.py
        </code>
      </pre>

      <p>Pour arrêter la webcam, ouvrez votre terminal, dans le dossier du projet tapez :</p>
      <pre>
        <code class="lang-bash">
          sudo /venv/bin/python3 /flask-video-stream-master/Kill_process.py
        </code>
      </pre>



    </div>
    <img src="{{ url_for('video_feed') }}">

    <div class="footer">By Dorian VOYDIE</div>
  </body>
</html>
