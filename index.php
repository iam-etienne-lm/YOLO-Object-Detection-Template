<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AWS Project | Big Data</title>
    <link rel="icon" href="img/logo.png" type="image/png" />
    <link
      type="text/css"
      rel="stylesheet"
      href="/public/ressources/css/index.css"
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
          <a class="active" href="/public/index.php"
            ><i class="fa fa-fw fa-home"></i> Home</a
          >
        </li>
        <li>
          <a href="/public/ressources/html/training.php"
            ><i class="fas fa-share-alt"></i> Training</a
          >
        </li>
        <li>
          <a href="/public/ressources/html/interface.php"
            ><i class="fas fa-share-alt"></i> Inference</a
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
      <h1 id="requirements">Requirements</h1>

      <p>Installez de quoi host un site Apache / PHP :</p>
      <pre>
        <code class="lang-bash">
          sudo bash apt-get update
          apt-get install apache2 php libapache2-mod-php
          systemctl restart apache2
        </code>
      </pre>

      <p>
        Modifier la configuration de PHP pour accepter l'upload de fichiers plus
        lourds (8Mo max sinon) :
      </p>
      <pre>
        <code class="lang-bash">
          sudo nano /etc/php/7.4/apache2/php.ini
        </code>
        <p>Remplissez-le comme suit :</p>
        <code class="lang-bash">
          file_uploads = On
          post_max_size = 800M
          upload_max_filesize = 800M
        </code>
      </pre>
      <p>Créer un environnement virtuel python :</p>
      <pre>
        <code class="lang-bash">
          python3 -m venv ./venv # Créer un environnement virtuel 
          source venv/bin/activate 
          which python3 # Cela doit pointer vers le dossier venv/bin 
          which pip # Cela doit pointer vers le dossier venv/bin 
          pip install -r Inference/yolov5/requirements.txt ```
        </code>
      </pre>

      <p>
        Enfin, placez le contenu de ce dossier dans /var/www/html. Puis donnez
        les autorisations root à ce dossier :
      </p>
      <pre>
        <code class="lang-bash">
          sudo chmod 777 -R /var/www/html
        </code>
      </pre>
    </div>

    <div class="footer">By Dorian VOYDIE, Jason DAURAT, Yoann MAAREK</div>
  </body>
</html>
