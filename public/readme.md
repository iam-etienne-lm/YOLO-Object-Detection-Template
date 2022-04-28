# Requirements

Installez de quoi host un site Apache / PHP :

```bash
sudo bash
apt-get update
apt-get install apache2 php libapache2-mod-php
systemctl restart apache2
```

Modifier la configuration de PHP pour accepter l'upload de fichiers plus lours (> 8Mo)

```bash
sudo nano /etc/php/7.4/apache2/php.ini
```

    file_uploads = On
    post_max_size = 800M
    upload_max_filesize = 800M

Créer un environnement virtuel python

```bash
python3 -m venv ./venv # Créer un environnement virtuel dans le dossier "public"
source venv/bin/activate
which python3 # Cela doit pointer vers le dossier venv/bin
which pip # Cela doit pointer vers le dossier venv/bin
pip install -r Inference/yolov5/requirements.txt
```

Enfin, placez le contenu de ce dossier dans /var/www/html
Puis donnez les autorisations root à ce dossier :

```bash
sudo chmod 777 -R /var/www/html
```
